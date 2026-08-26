using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using slm;
using TinyBrain;

const int ContextSize = 3;
const int EmbedDim = 10;
const int HiddenSize = 64;
const double LearningRate = 0.01;
const int Epochs = 30;
const int MaxVocabSize = 1000;
const int MaxTrainStories = 10000;
const int MaxValStories = 200;

var ParamsFile    = Path.Combine(AppContext.BaseDirectory, "parameters.txt");
var TrainingsFile = Path.Combine(AppContext.BaseDirectory, "trainings.txt");

// Parse --epoch N and --prompt <string> from CLI args
int? epochOverride = null;
string promptOverride = null;
for (var i = 0; i < args.Length - 1; i++)
{
    if (args[i] == "--epoch" && int.TryParse(args[i + 1], out var n))
        epochOverride = n;
    if (args[i] == "--prompt")
        promptOverride = args[i + 1];
}

// Derive epoch start from previous training runs recorded in trainings.txt
var epochStart = 0;
if (File.Exists(Path.Combine(AppContext.BaseDirectory, "trainings.txt")))
    foreach (var line in File.ReadAllLines(Path.Combine(AppContext.BaseDirectory, "trainings.txt")))
    {
        var m = System.Text.RegularExpressions.Regex.Match(line, @"epochs=(\d+)");
        if (m.Success) epochStart += int.Parse(m.Groups[1].Value);
    }

// Datasets are read directly from source — too large to copy to output dir
var datasetsDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../datasets"));
var trainCsv = Path.Combine(datasetsDir, "train.csv");
var valCsv = Path.Combine(datasetsDir, "validation.csv");

var trainText = DatasetLoader.LoadText(trainCsv, MaxTrainStories);
var valText = DatasetLoader.LoadText(valCsv, MaxValStories);

var uniqueTokens = Tokenizer.SplitWords(trainText).Distinct().Count();
var tokenizer = new Tokenizer(trainText, MaxVocabSize);
var coverage = (double)(tokenizer.VocabSize - 3) / uniqueTokens;
Console.WriteLine($"Unique tokens: {uniqueTokens}  Vocab size: {tokenizer.VocabSize}  Coverage: {coverage:P1}");

var trainTokens = tokenizer.Encode(trainText);
var valTokens = tokenizer.Encode(valText);
Console.WriteLine($"Train tokens: {trainTokens.Length}  Val tokens: {valTokens.Length}");

var trainData = new TrainingData(trainTokens, ContextSize);
var valData = new TrainingData(valTokens, ContextSize);
Console.WriteLine($"Train pairs: {trainData.Pairs.Length}  Val pairs: {valData.Pairs.Length}");

const int DHead = EmbedDim;   // single-head: head dim == embed dim
var model = new SlmModel(tokenizer.VocabSize, ContextSize, EmbedDim, DHead, HiddenSize);

// Load checkpoint if available (always, so --epoch resumes from existing params)
if (File.Exists(ParamsFile))
{
    Console.WriteLine("Loading saved parameters...");
    var flat = File.ReadAllLines(ParamsFile)
        .Select(l => double.Parse(l, CultureInfo.InvariantCulture)).ToArray();
    model.FlatParameters = flat;
}

var shouldTrain = epochOverride.HasValue || !File.Exists(ParamsFile);
var epochsToRun = epochOverride ?? Epochs;

if (shouldTrain)
{
    var finalTrainLoss = 0.0;
    var finalPerplexity = 0.0;
    var totalElapsed = TimeSpan.Zero;

    Console.WriteLine($"Training — Parameters: {model.FlatParameters.Length}  Pairs: {trainData.Pairs.Length}");
    for (var epoch = 0; epoch < epochsToRun; epoch++)
    {
        var displayEpoch = epochStart + epoch;
        var sw = Stopwatch.StartNew();
        Console.WriteLine($"Epoch {displayEpoch,3}");

        var epochLoss = 0.0;
        var total = trainData.Pairs.Length;
        var updateEvery = Math.Max(1, total / 10000);
        var pairIdx = 0;

        foreach (var (ctx, target) in trainData.Pairs)
        {
            model.ZeroGradients();

            var logits = model.Forward(ctx);
            var probs  = logits.Softmax();
            var loss   = probs.NLL(target);
            epochLoss += loss.Data[0];

            loss.Backpropagation();

            var gn = Math.Sqrt(model.ParameterMatrices.Sum(m => m.GradientNormSquared()));
            var clip = gn > 1.0 ? 1.0 / gn : 1.0;
            foreach (var m in model.ParameterMatrices)
                m.ApplyGradients(LearningRate, clip);

            pairIdx++;
            if (pairIdx % updateEvery == 0 || pairIdx == total || pairIdx == 1)
            {
                var pct = (double)pairIdx / total;
                var filled = (int)(pct * 40);
                var bar = new string('█', filled) + new string('░', 40 - filled);
                Console.Write($"\r  [{bar}] {pct:P0} ({pairIdx}/{total})");
            }
        }
        Console.WriteLine();

        var valLoss = valData.Pairs
            .Select(pair => -Math.Log(SoftmaxRow(model.Forward(pair.Context).Data)[pair.Target] + 1e-10))
            .Average();
        var perplexity = Math.Exp(valLoss);
        sw.Stop();

        finalTrainLoss = epochLoss / trainData.Pairs.Length;
        finalPerplexity = perplexity;
        totalElapsed += sw.Elapsed;

        Console.WriteLine($"Epoch {displayEpoch,3}: time={sw.Elapsed}  train_loss={finalTrainLoss:F4}  val_perplexity={finalPerplexity:F2}");
    }

    File.WriteAllText(ParamsFile,
        model.FlatParameters
            .Aggregate(new StringBuilder(),
                (sb, v) => sb.AppendLine(v.ToString(CultureInfo.InvariantCulture)))
            .ToString());
    Console.WriteLine("Parameters saved.");

    var record = $"{DateTime.Now:yyyy-MM-ddTHH:mm:ss}  epochs={epochsToRun}" +
                 $"  train_loss={finalTrainLoss:F4}  val_perplexity={finalPerplexity:F2}" +
                 $"  elapsed={totalElapsed:hh\\:mm\\:ss}";
    File.AppendAllText(TrainingsFile, record + Environment.NewLine);
    Console.WriteLine($"Training log: {record}");
}

Console.WriteLine("\nGenerating text (20 words):");
var genContext = new int[ContextSize];
if (promptOverride != null)
{
    var promptTokens = tokenizer.Encode(promptOverride);
    // left-pad with BOS if shorter than context window, take tail if longer
    Array.Fill(genContext, Tokenizer.BosIdx);
    var copyLen = Math.Min(promptTokens.Length, ContextSize);
    Array.Copy(promptTokens, promptTokens.Length - copyLen, genContext, ContextSize - copyLen, copyLen);
    Console.WriteLine($"Prompt context: [{string.Join(", ", genContext.Select(idx => tokenizer.Decode([idx])))}]");
}
else
    Array.Fill(genContext, Tokenizer.BosIdx);
var generated = new List<string>();

for (var i = 0; i < 20; i++)
{
    var logits = model.Forward(genContext);
    var probs = SoftmaxRow(logits.Data);
    var next = Multinomial(probs);
    if (next == Tokenizer.EosIdx) break;
    generated.Add(tokenizer.Decode([next]));
    var newContext = new int[ContextSize];
    Array.Copy(genContext, 1, newContext, 0, ContextSize - 1);
    newContext[ContextSize - 1] = next;
    genContext = newContext;
}

var prefix = promptOverride != null ? promptOverride + " " : "";
Console.WriteLine(prefix + string.Join(" ", generated));

// Untracked per-row softmax for inference (no autograd graph)
static double[] SoftmaxRow(double[] data)
{
    var n = data.Length;
    var max = double.NegativeInfinity;
    for (var j = 0; j < n; j++)
        if (data[j] > max) max = data[j];
    var exps = new double[n];
    var sum = 0.0;
    for (var j = 0; j < n; j++) { exps[j] = Math.Exp(data[j] - max); sum += exps[j]; }
    for (var j = 0; j < n; j++) exps[j] /= sum;
    return exps;
}

static int Multinomial(double[] probs)
{
    var r = new Random().NextDouble();
    var cumulative = 0.0;
    for (var i = 0; i < probs.Length; i++)
    {
        cumulative += probs[i];
        if (r < cumulative) return i;
    }
    return probs.Length - 1;
}

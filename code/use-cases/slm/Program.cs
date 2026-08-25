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
const int MaxVocabSize = 500;
const int MaxTrainStories = 2000;
const int MaxValStories = 200;

var ParamsFile = Path.Combine(AppContext.BaseDirectory, "parameters.txt");

// Datasets are read directly from source — too large to copy to output dir
var datasetsDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../datasets"));
var trainCsv = Path.Combine(datasetsDir, "train.csv");
var valCsv = Path.Combine(datasetsDir, "validation.csv");

var trainText = DatasetLoader.LoadText(trainCsv, MaxTrainStories);
var valText = DatasetLoader.LoadText(valCsv, MaxValStories);

var tokenizer = new Tokenizer(trainText, MaxVocabSize);
Console.WriteLine($"Vocab size: {tokenizer.VocabSize}");

var trainTokens = tokenizer.Encode(trainText);
var valTokens = tokenizer.Encode(valText);
Console.WriteLine($"Train tokens: {trainTokens.Length}  Val tokens: {valTokens.Length}");

var trainData = new TrainingData(trainTokens, ContextSize);
var valData = new TrainingData(valTokens, ContextSize);
Console.WriteLine($"Train pairs: {trainData.Pairs.Length}  Val pairs: {valData.Pairs.Length}");

var model = new SlmModel(tokenizer.VocabSize, ContextSize, EmbedDim, HiddenSize);

if (File.Exists(ParamsFile))
{
    Console.WriteLine("Loading saved parameters...");
    var lines = File.ReadAllLines(ParamsFile);
    var flat = lines.Select(l => double.Parse(l, CultureInfo.InvariantCulture)).ToArray();
    model.FlatParameters = flat;
}
else
{
    Console.WriteLine($"Training — Parameters: {model.FlatParameters.Length}  Pairs: {trainData.Pairs.Length}");
    for (var epoch = 0; epoch < Epochs; epoch++)
    {
        var sw = Stopwatch.StartNew();
        Console.WriteLine($"Epoch {epoch,3}");

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
            epochLoss += loss.Data[0, 0];

            loss.Backpropagation();

            var gn = Math.Sqrt(
                model.EmbeddingGradientNormSquared() +
                model.ParameterMatrices.Sum(m => m.GradientNormSquared()));
            var clip = gn > 1.0 ? 1.0 / gn : 1.0;

            foreach (var p in model.EmbeddingParameters)
                p.Data -= LearningRate * p.Gradient * clip;
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

        Console.WriteLine($"Epoch {epoch,3}: time={sw.Elapsed}  train_loss={epochLoss / trainData.Pairs.Length:F4}  val_perplexity={perplexity:F2}");
    }

    File.WriteAllText(ParamsFile,
        model.FlatParameters
            .Aggregate(new StringBuilder(),
                (sb, v) => sb.AppendLine(v.ToString(CultureInfo.InvariantCulture)))
            .ToString());
    Console.WriteLine("Parameters saved.");
}

Console.WriteLine("\nGenerating text (20 words):");
var genContext = new int[ContextSize];
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

Console.WriteLine(string.Join(" ", generated));

// Untracked per-row softmax for inference (no autograd graph)
static double[] SoftmaxRow(double[,] data)
{
    var n = data.GetLength(1);
    var max = double.NegativeInfinity;
    for (var j = 0; j < n; j++)
        if (data[0, j] > max) max = data[0, j];
    var exps = new double[n];
    var sum = 0.0;
    for (var j = 0; j < n; j++) { exps[j] = Math.Exp(data[0, j] - max); sum += exps[j]; }
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

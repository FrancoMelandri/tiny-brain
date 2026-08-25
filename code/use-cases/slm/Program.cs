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
    var parameters = model.Parameters;
    for (var i = 0; i < Math.Min(lines.Length, parameters.Length); i++)
        parameters[i].Data = double.Parse(lines[i], CultureInfo.InvariantCulture);
}
else
{
    Console.WriteLine($"Training Parameters: {model.Parameters.Length} data: {trainData.Pairs.Length}");
    for (var epoch = 0; epoch < Epochs; epoch++)
    {
        var sw = new Stopwatch();
        sw.Start();
        Console.WriteLine($"Epoch {epoch,3}");

        var epochLoss = 0.0;
        var total = trainData.Pairs.Length;
        var updateEvery = Math.Max(1, total / 10000);
        var pairIdx = 0;

        foreach (var (ctx, target) in trainData.Pairs)
        {
            var logits = model.Forward(ctx);
            var probs = logits.Softmax();
            var loss = Operand.Of(0) - probs[target].Log();
            epochLoss += loss.Data;

            loss.Backpropagation();

            var gradNorm = Math.Sqrt(model.Parameters.Sum(p => p.Gradient * p.Gradient));
            var clipScale = gradNorm > 1.0 ? 1.0 / gradNorm : 1.0;
            foreach (var p in model.Parameters)
                p.Data -= LearningRate * p.Gradient * clipScale;

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
            .Select(pair => -Math.Log(SoftmaxProbs(model.Forward(pair.Context))[pair.Target] + 1e-10))
            .Average();
        var perplexity = Math.Exp(valLoss);
        sw.Stop();

        Console.WriteLine($"Epoch {epoch,3}: time: {sw.Elapsed} train_loss={epochLoss / trainData.Pairs.Length:F4}  val_perplexity={perplexity:F2}");
    }

    File.WriteAllText(ParamsFile,
        model.Parameters
            .Aggregate(new StringBuilder(),
                (sb, p) => sb.AppendLine(p.Data.ToString(CultureInfo.InvariantCulture)))
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
    var probs = SoftmaxProbs(logits);
    var next = Multinomial(probs);
    if (next == Tokenizer.EosIdx) break;
    generated.Add(tokenizer.Decode([next]));
    var newContext = new int[ContextSize];
    Array.Copy(genContext, 1, newContext, 0, ContextSize - 1);
    newContext[ContextSize - 1] = next;
    genContext = newContext;
}

Console.WriteLine(string.Join(" ", generated));

// Untracked softmax for generation (no Operand graph created)
static double[] SoftmaxProbs(Operand[] logits)
{
    var maxLogit = logits.Max(l => l.Data);
    var exps = logits.Select(l => Math.Exp(l.Data - maxLogit)).ToArray();
    var sum = exps.Sum();
    return exps.Select(e => e / sum).ToArray();
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

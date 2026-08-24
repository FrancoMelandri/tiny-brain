using System;
using System.Collections.Generic;
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
const int MaxVocabSize = 100;
var ParamsFile = Path.Combine(AppContext.BaseDirectory, "parameters.txt");

var text = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "corpus.txt"));
var tokenizer = new Tokenizer(text, MaxVocabSize);
var tokens = tokenizer.Encode(text);
Console.WriteLine($"Vocab size: {tokenizer.VocabSize}  Tokens: {tokens.Length}");

var trainingData = new TrainingData(tokens, ContextSize);
Console.WriteLine($"Training pairs: {trainingData.Pairs.Length}");

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
    Console.WriteLine("Training...");
    for (var epoch = 0; epoch < Epochs; epoch++)
    {
        var epochLoss = 0.0;

        foreach (var (ctx, target) in trainingData.Pairs)
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
        }

        Console.WriteLine($"Epoch {epoch,3}: avg_loss={epochLoss / trainingData.Pairs.Length:F4}");
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

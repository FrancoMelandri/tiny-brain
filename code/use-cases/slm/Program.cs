using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using slm;
using TinyBrain;

const int ContextSize     = 50;
const int EmbedDim        = 64;
const int HiddenSize      = 128;
const float LearningRate  = 0.02f;
const int Epochs          = 30;
const int MaxVocabSize    = 32000;
const int MaxTrainStories = 1000;
const int MaxValStories   = 200;

var ParamsFile    = Path.Combine(AppContext.BaseDirectory, "parameters.gguf");
var TrainingsFile = Path.Combine(AppContext.BaseDirectory, "trainings.txt");

// Parse CLI args: --epoch N  --prompt <str>  --backend cpu|gpu  --batch-size N  --temperature T  --top-k K
int? epochOverride    = null;
string promptOverride = null;
string backendOverride = null;
int   batchSize   = 32;
float temperature = 1.0f;
int   topK        = 10;
for (var i = 0; i < args.Length - 1; i++)
{
    if (args[i] == "--epoch"       && int.TryParse(args[i + 1], out var n))                    epochOverride   = n;
    if (args[i] == "--prompt")                                                                   promptOverride  = args[i + 1];
    if (args[i] == "--backend")                                                                  backendOverride = args[i + 1].ToLowerInvariant();
    if (args[i] == "--batch-size"  && int.TryParse(args[i + 1],   out var bs) && bs > 0)       batchSize   = bs;
    if (args[i] == "--temperature" && float.TryParse(args[i + 1], out var t)  && t > 0)        temperature = t;
    if (args[i] == "--top-k"       && int.TryParse(args[i + 1],   out var k)  && k > 0)        topK        = k;
}

using IMatrixBackend computeBackend = backendOverride switch
{
    "cpu" => new CpuMatrixBackend(),
    "gpu" => GpuMatrixBackend.TryCreate(verbose: true)
             ?? throw new InvalidOperationException("--backend gpu requested but GPU is not available."),
    null  => GpuMatrixBackend.TryCreate(verbose: true) ?? (IMatrixBackend)new CpuMatrixBackend(),
    _     => throw new ArgumentException($"Unknown --backend value '{backendOverride}'. Use 'cpu' or 'gpu'.")
};
Operand.SetBackend(computeBackend);
Console.WriteLine($"Backend: {computeBackend.GetType().Name}");

// Epoch display counter — sums all "epochs=N" lines in trainings.txt
var epochStart = 0;
if (File.Exists(TrainingsFile))
    foreach (var line in File.ReadAllLines(TrainingsFile))
    {
        var m = System.Text.RegularExpressions.Regex.Match(line, @"epochs=(\d+)");
        if (m.Success) epochStart += int.Parse(m.Groups[1].Value);
    }

var datasetsDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../datasets"));
var trainCsv = Path.Combine(datasetsDir, "train_wikitext.csv");
var valCsv   = Path.Combine(datasetsDir, "validation_wikitext.csv");

// -------------------------------------------------------------------------
// Tokenizer, model and story cursor — first-launch vs resume
// -------------------------------------------------------------------------
var totalStories = DatasetLoader.CountStories(trainCsv);
Console.WriteLine($"Total stories in corpus: {totalStories}");

Tokenizer tokenizer;
SlmModel model;
int storyOffset;

if (File.Exists(ParamsFile))
{
    // Resume: load vocab + story cursor from GGUF
    Console.WriteLine("Loading checkpoint...");
    var (tensors, strKv, u64Kv) = GgufSerializer.ReadWithMetadata(ParamsFile);

    if (strKv.TryGetValue("tokenizer.vocab", out var vocabStr))
    {
        tokenizer = new Tokenizer(vocabStr.Split('\t'));
        Console.WriteLine($"Vocab loaded from GGUF: {tokenizer.VocabSize} tokens");
    }
    else
    {
        // Legacy GGUF without saved vocab — rebuild from full corpus
        Console.WriteLine("No vocab in GGUF, rebuilding from full corpus...");
        var allText = DatasetLoader.LoadText(trainCsv, totalStories);
        tokenizer = new Tokenizer(allText, MaxVocabSize);
        Console.WriteLine($"Vocab built: {tokenizer.VocabSize} tokens");
    }

    storyOffset = u64Kv.TryGetValue("training.story_offset", out var savedOff) ? (int)savedOff : 0;
    Console.WriteLine($"Story offset: {storyOffset}");

    model = new SlmModel(tokenizer.VocabSize, ContextSize, EmbedDim, EmbedDim, HiddenSize);
    model.FlatParameters = tensors.SelectMany(t => t.data).ToArray();
}
else
{
    // First launch: build vocabulary from ALL stories, save initial GGUF
    Console.WriteLine("First launch — building vocabulary from full corpus...");
    var allText   = DatasetLoader.LoadText(trainCsv, totalStories);
    var allWords  = Tokenizer.SplitWords(allText);
    var uniqueCnt = allWords.Distinct().Count();
    tokenizer = new Tokenizer(allText, MaxVocabSize);
    var coverage = (float)(tokenizer.VocabSize - 3) / uniqueCnt;
    Console.WriteLine($"Unique tokens: {uniqueCnt}  Vocab: {tokenizer.VocabSize}  Coverage: {coverage:P1}");

    storyOffset = 0;
    model = new SlmModel(tokenizer.VocabSize, ContextSize, EmbedDim, EmbedDim, HiddenSize);

    // Persist vocab + initial (random) params immediately
    GgufSerializer.Write(ParamsFile, "slm", model.NamedParameterMatrices,
        stringKv: [("tokenizer.vocab", string.Join('\t', tokenizer.Words))],
        uint64Kv: [("training.story_offset", 0UL)]);
    Console.WriteLine("Vocabulary and initial parameters saved to GGUF.");
}

// -------------------------------------------------------------------------
// Training
// -------------------------------------------------------------------------
var shouldTrain  = !File.Exists(ParamsFile) || epochOverride.HasValue;
var epochsToRun  = epochOverride ?? Epochs;

// valData is constant across epochs (small, fixed validation window)
var valData = new TrainingData(tokenizer.Encode(DatasetLoader.LoadText(valCsv, MaxValStories)), ContextSize);

// shouldTrain is now always true on first launch; on resume only if --epoch is given
// Re-evaluate: first launch always trains; resume trains only when --epoch specified
shouldTrain = !File.Exists(ParamsFile) || epochOverride.HasValue;

if (shouldTrain)
{
    var finalTrainLoss  = 0.0f;
    var finalPerplexity = 0.0f;
    var totalElapsed    = TimeSpan.Zero;

    Console.WriteLine($"Training — Parameters: {model.FlatParameters.Length}");

    for (var epoch = 0; epoch < epochsToRun; epoch++)
    {
        var displayEpoch    = epochStart + epoch;
        var effectiveOffset = storyOffset % totalStories;
        var sw              = Stopwatch.StartNew();

        // Load the story window for this epoch
        var epochText  = DatasetLoader.LoadText(trainCsv, MaxTrainStories, effectiveOffset);
        var trainData  = new TrainingData(tokenizer.Encode(epochText), ContextSize);

        Console.WriteLine($"Epoch {displayEpoch,3}  " +
                          $"stories=[{effectiveOffset}..{effectiveOffset + MaxTrainStories}]  " +
                          $"pairs={trainData.Pairs.Length}  batch={batchSize}");

        var epochLoss       = 0.0f;
        var total           = trainData.Pairs.Length;
        var updateEvery     = Math.Max(1, total / 10000);
        var samplesProcessed = 0;

        foreach (var (contexts, targets) in trainData.Batches(batchSize))
        {
            model.ZeroGradients();

            var logits = model.ForwardBatch(contexts);
            var probs  = logits.Softmax();
            var loss   = probs.NLL(targets);
            epochLoss += loss.Data[0] * contexts.Length;

            loss.Backpropagation();

            var gn   = MathF.Sqrt(model.ParameterMatrices.Sum(m => m.GradientNormSquared()));
            var clip = gn > 1.0f ? 1.0f / gn : 1.0f;
            foreach (var m in model.ParameterMatrices)
                m.ApplyGradients(LearningRate, clip);

            samplesProcessed += contexts.Length;
            if (samplesProcessed % updateEvery < batchSize || samplesProcessed >= total)
            {
                var pct    = (double)samplesProcessed / total;
                var filled = (int)(pct * 40);
                var bar    = new string('█', filled) + new string('░', 40 - filled);
                Console.Write($"\r  [{bar}] {pct:P0} ({samplesProcessed}/{total})");
            }
        }
        Console.WriteLine();

        storyOffset += MaxTrainStories;

        var valLoss    = valData.Pairs
            .Select(p =>
            {
                var logitsVal = model.Forward(p.Context);
                Operand.SynchronizeDeviceArray(logitsVal.Data);
                return -MathF.Log(SoftmaxRow(logitsVal.Data)[p.Target] + 1e-6f);
            })
            .Average();
        var perplexity = MathF.Exp(valLoss);
        sw.Stop();

        finalTrainLoss  = epochLoss / samplesProcessed;
        finalPerplexity = perplexity;
        totalElapsed   += sw.Elapsed;

        Console.WriteLine($"Epoch {displayEpoch,3}: time={sw.Elapsed}  " +
                          $"train_loss={finalTrainLoss:F4}  val_perplexity={finalPerplexity:F2}");
    }

    // Save updated params + story cursor + vocab
    GgufSerializer.Write(ParamsFile, "slm", model.NamedParameterMatrices,
        stringKv: [("tokenizer.vocab", string.Join('\t', tokenizer.Words))],
        uint64Kv: [("training.story_offset", (ulong)storyOffset)]);
    Console.WriteLine("Checkpoint saved.");

    var record = $"{DateTime.Now:yyyy-MM-ddTHH:mm:ss}  epochs={epochsToRun}" +
                 $"  train_loss={finalTrainLoss:F4}  val_perplexity={finalPerplexity:F2}" +
                 $"  elapsed={totalElapsed:hh\\:mm\\:ss}";
    File.AppendAllText(TrainingsFile, record + Environment.NewLine);
    Console.WriteLine($"Training log: {record}");
}

var rng = new Random();

// -------------------------------------------------------------------------
// Text generation
// -------------------------------------------------------------------------
Console.WriteLine("\nGenerating text (20 words):");
var genContext = new int[ContextSize];
if (promptOverride != null)
{
    var promptTokens = tokenizer.Encode(promptOverride);
    Array.Fill(genContext, Tokenizer.BosIdx);
    var copyLen = Math.Min(promptTokens.Length, ContextSize);
    Array.Copy(promptTokens, promptTokens.Length - copyLen, genContext, ContextSize - copyLen, copyLen);
    Console.WriteLine($"Prompt: [{string.Join(", ", genContext.Select(idx => tokenizer.Decode([idx])))}]");
}
else
    Array.Fill(genContext, Tokenizer.BosIdx);

var generated = new List<string>();
for (var i = 0; i < 20; i++)
{
    var logits = model.Forward(genContext);
    Operand.SynchronizeDeviceArray(logits.Data);
    var next = Sample(logits.Data, temperature, topK, rng);
    if (next == Tokenizer.EosIdx) break;
    generated.Add(tokenizer.Decode([next]));
    var newContext = new int[ContextSize];
    Array.Copy(genContext, 1, newContext, 0, ContextSize - 1);
    newContext[ContextSize - 1] = next;
    genContext = newContext;
}

var prefix = promptOverride != null ? promptOverride + " " : "";
Console.WriteLine(prefix + string.Join(" ", generated));

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------
static float[] SoftmaxRow(float[] data)
{
    var n   = data.Length;
    var max = float.NegativeInfinity;
    for (var j = 0; j < n; j++) if (data[j] > max) max = data[j];
    var exps = new float[n];
    var sum  = 0.0f;
    for (var j = 0; j < n; j++) { exps[j] = MathF.Exp(data[j] - max); sum += exps[j]; }
    for (var j = 0; j < n; j++) exps[j] /= sum;
    return exps;
}

static int Sample(float[] logits, float temperature, int topK, Random rng)
{
    var n = logits.Length;

    if (topK == 1)
        return Array.IndexOf(logits, logits.Max());  // greedy: argmax on raw logits

    // Apply temperature to logits before softmax
    var scaled = new float[n];
    for (var i = 0; i < n; i++) scaled[i] = logits[i] / temperature;

    var probs = SoftmaxRow(scaled);

    // Keep exactly top-k indices, zero out the rest, renormalise
    var topKSet = Enumerable.Range(0, n)
        .OrderByDescending(i => probs[i])
        .Take(topK)
        .ToHashSet();
    var sum = 0f;
    for (var i = 0; i < n; i++) { if (!topKSet.Contains(i)) probs[i] = 0f; else sum += probs[i]; }
    for (var i = 0; i < n; i++) probs[i] /= sum;

    // Inverse-CDF sample
    var r = (float)rng.NextDouble();
    var cumulative = 0f;
    for (var i = 0; i < n; i++) { cumulative += probs[i]; if (r < cumulative) return i; }
    return n - 1;
}

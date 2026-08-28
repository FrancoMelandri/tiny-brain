using System;
using System.Collections.Generic;
using System.Linq;

namespace slm;

public class TrainingData
{
    public (int[] Context, int Target)[] Pairs { get; }

    public TrainingData(int[] tokens, int contextSize)
    {
        var padded = new int[contextSize + tokens.Length];
        for (var i = 0; i < contextSize; i++)
            padded[i] = Tokenizer.BosIdx;
        tokens.CopyTo(padded, contextSize);

        Pairs = Enumerable.Range(0, tokens.Length)
            .Select(i => (Context: padded[i..(i + contextSize)], Target: padded[i + contextSize]))
            .ToArray();
    }

    public IEnumerable<(int[][] Contexts, int[] Targets)> Batches(int batchSize)
    {
        var buf = new List<(int[] ctx, int tgt)>(batchSize);
        foreach (var (ctx, tgt) in Pairs)
        {
            buf.Add((ctx, tgt));
            if (buf.Count == batchSize)
            {
                yield return (buf.Select(p => p.ctx).ToArray(), buf.Select(p => p.tgt).ToArray());
                buf.Clear();
            }
        }
        if (buf.Count > 0)
            yield return (buf.Select(p => p.ctx).ToArray(), buf.Select(p => p.tgt).ToArray());
    }
}

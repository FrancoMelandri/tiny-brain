using System;
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
}

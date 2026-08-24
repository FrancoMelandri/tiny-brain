using System;
using System.Linq;
using TinyBrain;

namespace slm;

public class EmbeddingTable
{
    private readonly Operand[,] _table;
    private readonly int _vocabSize;
    private readonly int _embedDim;

    public EmbeddingTable(int vocabSize, int embedDim)
    {
        _vocabSize = vocabSize;
        _embedDim = embedDim;
        var rng = new Random();
        _table = new Operand[vocabSize, embedDim];
        for (var i = 0; i < vocabSize; i++)
            for (var j = 0; j < embedDim; j++)
                _table[i, j] = Operand.Of((rng.NextDouble() * 2 - 1) * 0.1);
    }

    public Operand[] Lookup(int tokenIndex)
    {
        var result = new Operand[_embedDim];
        for (var j = 0; j < _embedDim; j++)
            result[j] = _table[tokenIndex, j];
        return result;
    }

    public Operand[] Parameters
        => Enumerable.Range(0, _vocabSize)
            .SelectMany(i => Enumerable.Range(0, _embedDim).Select(j => _table[i, j]))
            .ToArray();
}

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

    // Returns MatrixOperand [1, contextSize*embedDim]; backward writes gradients back to _table entries
    public MatrixOperand LookupFlat(int[] contextIndices)
    {
        var cols = contextIndices.Length * _embedDim;
        var data = new double[cols];
        for (var ci = 0; ci < contextIndices.Length; ci++)
            for (var d = 0; d < _embedDim; d++)
                data[ci * _embedDim + d] = _table[contextIndices[ci], d].Data;

        var table = _table;
        var indices = (int[])contextIndices.Clone();
        var embedDim = _embedDim;

        var result = MatrixOperand.Of(new double[1, cols]); // shape [1, cols]
        Array.Copy(data, result.Data, cols);
        result.SetBackward(grad =>
        {
            for (var ci = 0; ci < indices.Length; ci++)
                for (var d = 0; d < embedDim; d++)
                    table[indices[ci], d].Gradient += grad[ci * embedDim + d];
        });
        return result;
    }

    public void ZeroGradients()
    {
        for (var i = 0; i < _vocabSize; i++)
            for (var j = 0; j < _embedDim; j++)
                _table[i, j].Gradient = 0;
    }

    public double GradientNormSquared()
    {
        var sum = 0.0;
        for (var i = 0; i < _vocabSize; i++)
            for (var j = 0; j < _embedDim; j++)
                sum += _table[i, j].Gradient * _table[i, j].Gradient;
        return sum;
    }
}

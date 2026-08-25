using TinyBrain;

namespace slm;

public class EmbeddingTable
{
    private readonly Operand _table;  // [vocabSize, embedDim]
    private readonly int _embedDim;

    public EmbeddingTable(int vocabSize, int embedDim)
    {
        _embedDim = embedDim;
        _table = Operand.OfRandom(vocabSize, embedDim, 0.1);
    }

    // Returns Operand [1, contextSize*embedDim]
    // _backward accumulates gradients into _table.Gradient
    public Operand LookupFlat(int[] contextIndices)
    {
        var cols = contextIndices.Length * _embedDim;
        var flatData = new double[1, cols];
        for (var ci = 0; ci < contextIndices.Length; ci++)
            for (var d = 0; d < _embedDim; d++)
                flatData[0, ci * _embedDim + d] = _table.Data[contextIndices[ci] * _embedDim + d];

        var tableGrad = _table.Gradient;
        var indices = (int[])contextIndices.Clone();
        var embedDim = _embedDim;

        var result = Operand.Of(flatData);
        result.SetBackward(grad =>
        {
            for (var ci = 0; ci < indices.Length; ci++)
                for (var d = 0; d < embedDim; d++)
                    tableGrad[indices[ci] * embedDim + d] += grad[ci * embedDim + d];
        });
        return result;
    }

    public Operand ParameterMatrix => _table;
    public void ZeroGradients() => _table.ZeroGradient();
    public double GradientNormSquared() => _table.GradientNormSquared();
}

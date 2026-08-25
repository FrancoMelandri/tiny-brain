using System.Linq;
using TinyBrain;

namespace slm;

public class SlmModel
{
    private readonly EmbeddingTable _embedding;
    private readonly MatrixBrain _hiddenBrain;
    private readonly MatrixBrain _outputBrain;

    public SlmModel(int vocabSize, int contextSize, int embedDim, int hiddenSize)
    {
        _embedding = new EmbeddingTable(vocabSize, embedDim);
        _hiddenBrain = new MatrixBrain("slm_h", contextSize * embedDim, [hiddenSize], ActivationType.Tanh);
        _outputBrain = new MatrixBrain("slm_o", hiddenSize, [vocabSize], ActivationType.None);
    }

    public MatrixOperand Forward(int[] contextIndices)
    {
        var input = _embedding.LookupFlat(contextIndices);
        var hidden = _hiddenBrain.Forward(input);
        return _outputBrain.Forward(hidden);
    }

    public void ZeroGradients() => _embedding.ZeroGradients();

    public Operand[] EmbeddingParameters => _embedding.Parameters;

    public double EmbeddingGradientNormSquared() => _embedding.GradientNormSquared();

    public MatrixOperand[] ParameterMatrices
        => [.._hiddenBrain.ParameterMatrices, .._outputBrain.ParameterMatrices];

    // Flat sequence of all parameter values for save/load: embedding row-major, then matrix brain elements
    public double[] FlatParameters
    {
        get
        {
            var emb = _embedding.Parameters.Select(p => p.Data);
            var mats = ParameterMatrices.SelectMany(m =>
            {
                return m.Data;
            });
            return [..emb, ..mats];
        }
        set
        {
            var idx = 0;
            foreach (var p in _embedding.Parameters)
                p.Data = value[idx++];
            foreach (var m in ParameterMatrices)
                for (var i = 0; i < m.Data.Length; i++)
                    m.Data[i] = value[idx++];
        }
    }
}

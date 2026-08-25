using System.Linq;
using TinyBrain;

namespace slm;

public class SlmModel
{
    private readonly EmbeddingTable _embedding;
    private readonly Brain _hiddenBrain;
    private readonly Brain _outputBrain;

    public SlmModel(int vocabSize, int contextSize, int embedDim, int hiddenSize)
    {
        _embedding = new EmbeddingTable(vocabSize, embedDim);
        _hiddenBrain = new Brain("slm_h", contextSize * embedDim, [hiddenSize], ActivationType.Tanh);
        _outputBrain = new Brain("slm_o", hiddenSize, [vocabSize], ActivationType.None);
    }

    public Operand Forward(int[] contextIndices)
    {
        var input = _embedding.LookupFlat(contextIndices);
        var hidden = _hiddenBrain.Forward(input);
        return _outputBrain.Forward(hidden);
    }

    public void ZeroGradients() => _embedding.ZeroGradients();

    public Operand[] ParameterMatrices
        => [_embedding.ParameterMatrix, .._hiddenBrain.ParameterMatrices, .._outputBrain.ParameterMatrices];

    public double[] FlatParameters
    {
        get => ParameterMatrices.SelectMany(m => m.Data).ToArray();
        set
        {
            var idx = 0;
            foreach (var m in ParameterMatrices)
                for (var i = 0; i < m.Data.Length; i++)
                    m.Data[i] = value[idx++];
        }
    }
}

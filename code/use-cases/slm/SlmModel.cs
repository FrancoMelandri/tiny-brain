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

    public Operand[] Forward(int[] contextIndices)
    {
        var flatEmbedding = contextIndices
            .SelectMany(idx => _embedding.Lookup(idx))
            .ToArray();
        var hidden = _hiddenBrain.Forward(flatEmbedding);
        return _outputBrain.Forward(hidden);
    }

    public Operand[] Parameters
        => [.._embedding.Parameters, .._hiddenBrain.Parameters, .._outputBrain.Parameters];
}

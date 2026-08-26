using System.Linq;
using TinyBrain;

namespace slm;

public class SlmModel
{
    private readonly EmbeddingTable _embedding;
    private readonly AttentionHead _attention;
    private readonly Brain _brain;

    private readonly int _contextSize;

    public SlmModel(int vocabSize, int contextSize, int embedDim, int dHead, int hiddenSize)
    {
        _contextSize = contextSize;
        _embedding = new EmbeddingTable(vocabSize, embedDim);
        _attention = new AttentionHead(embedDim, dHead, contextSize);
        _brain = new Brain("slm", embedDim, [hiddenSize, vocabSize],
                           [ActivationType.Tanh, ActivationType.None]);
    }

    public Operand Forward(int[] contextIndices)
    {
        var x = _embedding.LookupSequence(contextIndices);  // [T, embedDim]
        var attended = _attention.Forward(x);               // [T, embedDim]
        var last = attended.SliceRow(_contextSize - 1);     // [1, embedDim]
        return _brain.Forward(last);                        // [1, vocabSize]
    }

    public void ZeroGradients()
    {
        _embedding.ZeroGradients();
        _attention.ZeroGradients();
    }

    public Operand[] ParameterMatrices
        => [_embedding.ParameterMatrix, .._attention.ParameterMatrices, .._brain.ParameterMatrices];

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

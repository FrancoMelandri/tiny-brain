using System.Linq;
using TinyBrain;

namespace slm;

public class SlmModel
{
    private readonly EmbeddingTable _embedding;
    private readonly Brain _brain;

    public SlmModel(int vocabSize, int contextSize, int embedDim, int hiddenSize)
    {
        _embedding = new EmbeddingTable(vocabSize, embedDim);
        _brain = new Brain("slm", contextSize * embedDim, [hiddenSize, vocabSize],
                           [ActivationType.Tanh, ActivationType.None]);
    }

    public Operand Forward(int[] contextIndices)
        => _brain.Forward(_embedding.LookupFlat(contextIndices));

    public void ZeroGradients() => _embedding.ZeroGradients();

    public Operand[] ParameterMatrices
        => [_embedding.ParameterMatrix, .._brain.ParameterMatrices];

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

using System.Collections.Generic;
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

    public (string name, Operand tensor)[] NamedParameterMatrices
    {
        get
        {
            var named = new List<(string, Operand)>
            {
                ("token_embd.weight", _embedding.ParameterMatrix),
                ("attn.q_proj.weight", _attention.ParameterMatrices[0]),
                ("attn.k_proj.weight", _attention.ParameterMatrices[1]),
                ("attn.v_proj.weight", _attention.ParameterMatrices[2]),
                ("attn.o_proj.weight", _attention.ParameterMatrices[3]),
            };
            for (var i = 0; i < _brain.Layers.Length; i++)
            {
                named.Add(($"slm.layer{i}.weight", _brain.Layers[i].Weights));
                named.Add(($"slm.layer{i}.bias",   _brain.Layers[i].Bias));
            }
            return named.ToArray();
        }
    }

    public float[] FlatParameters
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

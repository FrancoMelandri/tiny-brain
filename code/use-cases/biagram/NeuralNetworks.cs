using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TinyBrain;
using TinyFp.Extensions;

namespace biagram;

public class NeuralNetworks
{
    private readonly Brain _brain;

    int CtoI(char c) => c - '`';
    char ItoC(int i) => (char)(i + '`');

    public NeuralNetworks(string[] words)
    {
        _brain = new Brain("biagram", 27, [27], ActivationType.None);
    }

    public void LoadGguf(string path)
    {
        if (!File.Exists(path)) return;
        var tensors = GgufSerializer.Read(path);
        FlatParameters = tensors.SelectMany(t => t.data).ToArray();
    }

    public void SaveGguf(string path)
        => GgufSerializer.Write(path, "biagram", NamedParameterMatrices);

    public float[] FlatParameters
    {
        get => _brain.ParameterMatrices.SelectMany(m => m.Data).ToArray();
        set
        {
            var idx = 0;
            foreach (var m in _brain.ParameterMatrices)
                for (var i = 0; i < m.Data.Length; i++)
                    m.Data[i] = value[idx++];
        }
    }

    public Operand[] ParameterMatrices => _brain.ParameterMatrices;

    public (string name, Operand tensor)[] NamedParameterMatrices
        => _brain.Layers.SelectMany((l, i) => new[]
        {
            ($"biagram.layer{i}.weight", l.Weights),
            ($"biagram.layer{i}.bias",   l.Bias),
        }).ToArray();

    public Operand Forward(Operand input) => _brain.Forward(input);

    public void Generate(int generations)
    {
        for (var toGenerate = 0; toGenerate < generations; toGenerate++)
        {
            var ix = 0;
            var steps = 0;
            var generated = new List<char>();
            while (true)
            {
                var xenc = SamplingUtils.OneHotMatrix([ix], 27);
                var probs = _brain.Forward(xenc).Softmax();
                var p = new float[27];
                for (var j = 0; j < 27; j++) p[j] = probs.Data[j];
                ix = SamplingUtils.Multinomial(p);
                if (ix == 0 || ++steps >= 100) break;
                generated.Add(ItoC(ix));
            }
            Console.WriteLine(generated.Fold(string.Empty, (a, c) => a + c));
        }
    }
}

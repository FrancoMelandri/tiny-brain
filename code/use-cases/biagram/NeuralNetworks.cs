using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using TinyBrain;
using TinyFp.Extensions;

namespace biagram;

public class NeuralNetworks
{
    private readonly MatrixBrain _brain;

    int CtoI(char c) => c - '`';
    char ItoC(int i) => (char)(i + '`');

    public NeuralNetworks(string[] words)
    {
        _brain = new MatrixBrain("biagram", 27, [27], ActivationType.None);
    }

    public void Initialize()
    {
        if (!File.Exists("parameters.txt")) return;
        var flat = File.ReadAllLines("parameters.txt")
            .Select(l => double.Parse(l, CultureInfo.InvariantCulture)).ToArray();
        FlatParameters = flat;
    }

    public double[] FlatParameters
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

    public MatrixOperand[] ParameterMatrices => _brain.ParameterMatrices;

    public MatrixOperand Forward(MatrixOperand input) => _brain.Forward(input);

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
                var p = new double[27];
                for (var j = 0; j < 27; j++) p[j] = probs.Data[j];
                ix = SamplingUtils.Multinomial(p);
                if (ix == 0 || ++steps >= 100) break;
                generated.Add(ItoC(ix));
            }
            Console.WriteLine(generated.Fold(string.Empty, (a, c) => a + c));
        }
    }

    public void SaveParameters()
        => File.WriteAllText("parameters.txt",
            FlatParameters
                .Aggregate(new System.Text.StringBuilder(),
                    (sb, v) => sb.AppendLine(v.ToString(CultureInfo.InvariantCulture)))
                .ToString());
}

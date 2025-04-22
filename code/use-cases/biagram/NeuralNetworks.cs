using System;
using System.Collections.Generic;
using System.Data;
using System.Globalization;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;
using TinyBrain;
using TinyFp.Extensions;

namespace biagram;

public class NeuralNetworks
{
    private IEnumerable<(char Char1, char Char2)> _biagrams;
    private int[] _xs = Array.Empty<int>();
    private int[] _ys = Array.Empty<int>();
    private readonly Brain _brain;

    public NeuralNetworks(string[] words)
    {
        _brain = new Brain("biagram", 27, [27], ActivationType.None);
        _biagrams = EvaluateBiagrams(words);
    }
    
    //
    // convert character to index and vice versa
    int CtoI(char c) => c - '`';
    char ItoC(int i) => (char)(i + '`');
    
    IEnumerable<(char Char1, char Char2)> EvaluateBiagrams(string[] words) =>
        words
            .Select(_ => $"`{_}`")
            .Select(
                w =>
                    w.SkipLast(1)
                        .Select((c, index) => (Char1: c, Char2: w[index + 1]))
            )
            .Fold(new List<(char, char)>(),
                (a, i) => a.Tee(_ => _.AddRange(i.Select(_ => (_.Char1, _.Char2)))));

    public void Initialize()
    {
        if (!System.IO.File.Exists("parameters.txt"))
            return;
        var lines = System.IO.File.ReadAllLines("parameters.txt");
        for (var index = 0; index < lines.Length; index++)
            _brain.Parameters[index].Data = double.Parse(lines[index], CultureInfo.InvariantCulture);
    }

    public Operand[] Parameters => _brain.Parameters;
    
    //
    // generate characters using multinomial
    public void Generate(int generations)
    {
        for (var toGenerate = 0; toGenerate < generations; toGenerate++)
        {
            var ix = 0;
            var steps = 0;
            var generated = new List<char>();
            while (true)
            {
                //
                // get the probability row
                var xenc = SamplingUtils.OneHot([ix], 27);
                var logits = Forward(xenc);
                var softMax = Softmax(logits);
                var p = softMax[0].Select(_ => _.Data).ToArray();;
                //
                // get next character using multinomial based on probability vector 
                ix = SamplingUtils.Multinomial(p);

                //
                // in case of next character = 0 we reach the end of the generated word
                if (ix == 0 || ++steps >= 100)
                    break;
                generated.Add(ItoC(ix));
            }
            var generateString = generated.Fold(string.Empty, (a, c) => a.Tee(_ => _ + c));
            Console.WriteLine(generateString);
        }    
    }

    public Operand[][] Forward(Operand[][] operands)
        => operands
            .Fold((Index: 0, Logits: new Operand[operands.Length][]),
            (a, i) => 
                (a.Index + 1, 
                    a.Logits.Tee(_ => _[a.Index] = _brain.Forward(i))))
            .Map(fold => fold.Logits);

    public static Operand[][] Softmax(Operand[][] logits)
        => logits.Fold((Index: 0, Probs: new Operand[logits.Length][]),
                (a, i) =>
                    (a.Index + 1,
                        a.Probs.Tee(_ => _[a.Index] = i.Select(_ => _.Exp()).ToArray()
                            .Map(counts => (Counts: counts, Sum: counts.Sum(_ => _.Data)))
                            .Map(tuple => tuple.Counts.Select(_ => _.Tee(_ => _ /= tuple.Sum)).ToArray()))))
            .Map(_ => _.Probs);

    public void SaveParameters()
        => System.IO.File.WriteAllText(
            "parameters.txt",
            _brain
            .Parameters
            .Fold(new StringBuilder(),
                (a, i) => a.AppendLine(i.Data.ToString(CultureInfo.InvariantCulture)))
                    .ToString());
}
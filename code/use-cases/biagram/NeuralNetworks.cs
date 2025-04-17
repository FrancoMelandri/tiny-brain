using System;
using System.Collections.Generic;
using System.Linq;
using TinyFp.Extensions;

namespace biagram;

public class NeuralNetworks
{
    private IEnumerable<(char Char1, char Char2)> _biagrams;
    private int[] _xs = Array.Empty<int>();
    private int[] _ys = Array.Empty<int>();
    
    public NeuralNetworks(string[] words)
    {
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
    }
}
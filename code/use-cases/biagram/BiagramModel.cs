using System;
using System.Collections.Generic;
using System.Linq;
using TinyFp.Extensions;

namespace biagram;

public class BiagramModel
{
    private IEnumerable<(char Char1, char Char2)> _biagrams;
    private double[,] _probabilities;
    
    public BiagramModel(string[] words)
    {
        _biagrams = EvaluateBiagrams(words);
    }
    
    //
    // convert character to index and vice versa
    int CtoI(char c) => c - '`';
    char ItoC(int i) => (char)(i + '`');
    
    public void InitializeBiagramModel()
    {
        // define the biagrams model using the probability
        // for each biagram we have a probability of the next character
        //
        // initialize a matrix containing the sum of the second character following the previous character
        var N = _biagrams.Fold(new int[27, 27],
            (a, i) => a.Tee(_ => a[CtoI(i.Item1), CtoI(i.Item2)] += 1));

        //
        // compute the probability of a character to follow another character as the
        // count / sum()
        _probabilities = new double[27, 27];
        for (var i = 0; i < 27; ++i)
        {
            var sum = Enumerable.Range(0, 27).Select(j => N[i, j]).Sum();
            for (var j = 0; j < 27; ++j)
                _probabilities[i, j] = (double)N[i, j]/sum;
        }
    }
    
    //
    // generate characters using multinomial
    public void Generate(int generations)
    {
        for (var toGenerate = 0; toGenerate < generations; toGenerate++)
        {
            var ix = 0;
            var steps = 0;
            var generated = new List<char>();
            var p = new double[27];
            while (true)
            {
                //
                // get the probability row
                for (int j = 0; j < 27; j++)
                    p[j] = _probabilities[ix,j];

                //
                // get next character using multinomial based on probability vector 
                ix = SamplingUtils.Multinomial(p);

                //
                // in case of next character = 0 we reach the end of the generated word
                if (ix == 0 || ++steps >= 10)
                    break;
                generated.Add(ItoC(ix));
            }
            var generateString = generated.Fold(string.Empty, (a, c) => a.Tee(_ => _ + c));
            Console.WriteLine(generateString);
        }    
    }

    //
    // define a function to estimate the the model prediction
    // we can consider the likelihood to estimate the parameter
    // likelihood = Product(pi)
    // we can consider the log
    // log(likelihood) = sum ( log(pi) )
    //
    // due the fact the loss function should be close to zero when is good, 
    // we can define our loss funciton as
    // loss = -log(likelihood)/N
    // where N is to normalize
    public double EvaluateLoss() 
        => _biagrams
            .Fold((0.0, 0),
                (a, i) => 
                (
                    a.Item1 + Math.Log(_probabilities[CtoI(i.Item1), CtoI(i.Item2)]), 
                    a.Item2 + 1
                ))
            .Map(_ => -_.Item1 / _.Item2);

    //
    // character sequence
    //
    // create a list of tuple containing two character the previous and the next
    // adding two special characters at the beginning and the end of the string load form file
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
    
}
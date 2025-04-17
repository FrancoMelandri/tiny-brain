

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using TinyFp.Extensions;

//
// giving one character we want to predict the next character in the sequence
//

//
// dataset information
var words = System.IO.File.ReadAllLines("names.txt");
Console.WriteLine($"Found {words.Length} words");
words.Min(_ => _.Length).Tee(_ => Console.WriteLine($"Min: {_}"));
words.Max(_ => _.Length).Tee(_ => Console.WriteLine($"Max: {_}"));

//
// character sequence
//
// create a list of tuple containing two character the previous and the next
// adding two special characters at the beginning and the end of the string load form file
IEnumerable<(char Char1, char Char2)> ComputeBiagrams(string[] words) =>
    words
    .Select(_ => $"`{_}`")
    .Select(
        w =>
            w.SkipLast(1)
            .Select((c, index) => (Char1: c, Char2: w[index + 1]))
     )
    .Fold(new List<(char, char)>(),
         (a, i) => a.Tee(_ => _.AddRange(i.Select(_ => (_.Char1, _.Char2)))));


var biagrams = ComputeBiagrams(words);

//
// count all occurence for each tuple (c1, c2)
var dictionary = biagrams.Fold(new Dictionary<(char, char), int>(),
    (a, i) =>
        a.ToOption(_ => !_.ContainsKey(i))
            .Match(_ => a.Tee(__ => a[i] += 1),
                () => a.Tee(__ => a[i] = 1)));

//
// convert character to index and vice versa
Func<char, int> CtoI = c => (int)c - (int)'`';
Func<int, char> ItoC = i => (char)(i + (int)'`');

//
// initialize a matrix containing the sum of the second character following the previous character
var N = biagrams.Fold(new int[27, 27],
    (a, i) => a.Tee(_ => a[CtoI(i.Item1), CtoI(i.Item2)] += 1));

//
// compute the probability of a character to follow another character as the
// count / sum()
var P = new double[27, 27];
for (var i = 0; i < 27; ++i)
{
    var sum = Enumerable.Range(0, 27).Select(j => N[i, j]).Sum();
    for (var j = 0; j < 27; ++j)
        P[i,j] = (double)N[i, j]/sum;
}

//
// generate characters using multinomial
Console.WriteLine("Generate:");
for (var toGenerate = 0; toGenerate < 10; toGenerate++)
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
            p[j] = P[ix,j];

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

double ComputeLoss(string[] words, double[,] P, Func<char, int> CtoI) 
    => ComputeBiagrams(words)
    .Fold((0.0, 0),
        (a, i) => 
            (
                a.Item1 + Math.Log(P[CtoI(i.Item1), CtoI(i.Item2)]), 
                a.Item2 + 1
                ))
    .Map(_ => -_.Item1 / _.Item2);;

var loss = ComputeLoss(words, P, CtoI);
Console.WriteLine($"Loss: {loss}");




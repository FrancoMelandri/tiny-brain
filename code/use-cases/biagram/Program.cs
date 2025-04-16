

using System;
using System.Collections.Generic;
using System.Linq;
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
// create a list of tuple containing two character the previsou and the next
// adding two special characters at the beginning and the end of the string load form file
var chars = words
    // .Take(3)
    .Select(_ => $"`{_}`")
    .Select(
        w =>
            w.SkipLast(1)
            .Select((c, index) => (Char1: c, Char2: w[index + 1]))
     )
    .Fold(new List<(char, char)>(),
         (a, i) => a.Tee(_ => _.AddRange(i.Select(_ => (_.Char1, _.Char2)))));


//
// count all occurence for each tuple (c1, c2)
var dictionary = chars.Fold(new Dictionary<(char, char), int>(),
    (a, i) =>
        a.ToOption(_ => !_.ContainsKey(i))
            .Match(_ => a.Tee(__ => a[i] += 1),
                () => a.Tee(__ => a[i] = 1)));

Func<char, int> CtoI = c => (int)c - (int)'`';
Func<int, char> ItoC = i => (char)(i + (int)'`');

//
// initialize a matrix containing the sum 
var N = chars.Fold(new int[27, 27],
    (a, i) => a.Tee(_ => a[CtoI(i.Item1), CtoI(i.Item2)] += 1));


Console.WriteLine($"Found {dictionary.Count} words");
    



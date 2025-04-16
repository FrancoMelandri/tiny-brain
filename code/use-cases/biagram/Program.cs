

using System;
using System.Collections.Generic;
using System.ComponentModel.Design.Serialization;
using System.Linq;
using TinyFp.Extensions;

//
// giving one character we want to predict the next character in the sequence
//

var words = System.IO.File.ReadAllLines("names.txt");

// dataset information
Console.WriteLine($"Found {words.Length} words");
words.Min(_ => _.Length).Tee(_ => Console.WriteLine($"Min: {_}"));
words.Max(_ => _.Length).Tee(_ => Console.WriteLine($"Max: {_}"));

// character sequence
var chars = words
    .Take(2)
    .Select(_ => $".{_}.")
    .Select(
        w =>
            w.SkipLast(1)
            .Select((c, index) => (Char1: c, Char2: w[index + 1]))
     )
    .Fold(new List<(char, char)>(),
         (a, i) => a.Tee(_ => _.AddRange(i.Select(_ => (_.Char1, _.Char2)))));
Console.WriteLine($"Found {chars.Count} words");
    



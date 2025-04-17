

using System;
using System.Collections.Generic;
using System.Linq;
using biagram;
using TinyFp.Extensions;

//
// giving one character we want to predict the next character in the sequence
//

//
// dataset information
var wordsDataset = System.IO.File.ReadAllLines("names.txt");
Console.WriteLine($"Found {wordsDataset.Length} words");
wordsDataset.Min(_ => _.Length).Tee(_ => Console.WriteLine($"Min: {_}"));
wordsDataset.Max(_ => _.Length).Tee(_ => Console.WriteLine($"Max: {_}"));

//
// evaluate all the biagrams for the list of words
var biagramsModel = new BiagramModel(wordsDataset);
biagramsModel.InitializeBiagramModel();

//
// generate characters using multinomial
Console.WriteLine("Generate:");
biagramsModel.Generate(10);

//
// evaluate the loss function
var loss = biagramsModel.EvaluateLoss();
Console.WriteLine($"Loss: {loss}");

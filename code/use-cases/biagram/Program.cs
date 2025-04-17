using System;
using biagram;

//
// giving one character we want to predict the next character in the sequence
//

//
// dataset information
var wordsDataset = System.IO.File.ReadAllLines("names.txt");
Console.WriteLine($"Found {wordsDataset.Length} words");

Console.WriteLine("----");
Console.WriteLine("BiagramsModel");

//
// evaluate all the biagrams for the list of words
var biagramsModel = new BiagramModel(wordsDataset);
biagramsModel.Initialize();

//
// generate characters using multinomial
Console.WriteLine("Generate:");
biagramsModel.Generate(10);

//
// evaluate the loss function
var loss = biagramsModel.EvaluateLoss();
Console.WriteLine($"Loss: {loss}");

Console.WriteLine("");
Console.WriteLine("----");
Console.WriteLine("NeuralNetworks");

var neuralNetwork = new NeuralNetworks(wordsDataset);
neuralNetwork.Initialize();

var trainingSet = biagramsModel.CreateTraining();
Console.WriteLine("");

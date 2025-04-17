using System;
using System.Linq;
using biagram;
using TinyBrain;

//
// giving one character we want to predict the next character in the sequence
//

//
// dataset information
var wordsDataset = System.IO.File.ReadAllLines("names.txt").Take(1).ToArray();
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
var xs = SamplingUtils.OneHot(trainingSet.xs, 27);
var y = SamplingUtils.OneHot(trainingSet.ys, 27);

var layer = new Layer("biagram", 27, 27, ActivationType.None);
var act = layer.Forward(xs[0]); // logits

act = act.Select(_ => _.Exp()).ToArray(); // count  
var sum = act.Sum(_ => _.Data); 
act = act.Select(_ => _ / sum).ToArray(); // probabilities

Console.WriteLine("");

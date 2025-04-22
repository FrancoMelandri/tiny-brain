using System;
using System.Linq;
using biagram;
using TinyBrain;

//
// giving one character we want to predict the next character in the sequence
//

const int MaxWords = 100;
//
// dataset information
var wordsDataset = System.IO.File.ReadAllLines("names.txt").Take(MaxWords).ToArray();
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

//
// Learning
var trainingSet = biagramsModel.CreateTraining();

var xs = SamplingUtils.OneHot(trainingSet.xs, 27);
var ys = SamplingUtils.OneHot(trainingSet.ys, 27);
Console.WriteLine($"Training set dimension: {xs.Length}");

for (var loop = 0; loop < 20; loop += 1)
{
    var logits = neuralNetwork.Forward(xs);
    var softMax = NeuralNetworks.Softmax(logits);

    //
    // loss
    var logSum = Operand.Of(0);
    for (var i = 0; i < xs.Length; i++)
    {
        var y = trainingSet.ys[i];
        logSum -= softMax[i][y].Log();
    }
    var lossNeural = logSum / trainingSet.xs.Length;
    lossNeural.Label = "lossNeural";
    
    Console.WriteLine($"Step {loop}: loss={lossNeural.Data}");

    lossNeural.Backpropagation();

    //
    // update the weights
    foreach (var x in neuralNetwork.Parameters)
        x.Data += -0.1 * x.Gradient;
}

neuralNetwork.Generate(5);
neuralNetwork.SaveParameters();

Console.WriteLine("");

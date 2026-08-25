using System;
using System.Linq;
using biagram;
using TinyBrain;

//
// giving one character we want to predict the next character in the sequence
//

const int MaxWords = 100;

var wordsDataset = System.IO.File.ReadAllLines(System.IO.Path.Combine(AppContext.BaseDirectory, "names.txt")).Take(MaxWords).ToArray();
Console.WriteLine($"Found {wordsDataset.Length} words");

Console.WriteLine("----");
Console.WriteLine("BiagramsModel");

var biagramsModel = new BiagramModel(wordsDataset);
biagramsModel.Initialize();

Console.WriteLine("Generate:");
biagramsModel.Generate(10);

var loss = biagramsModel.EvaluateLoss();
Console.WriteLine($"Loss: {loss}");

Console.WriteLine("");
Console.WriteLine("----");
Console.WriteLine("NeuralNetworks");

var neuralNetwork = new NeuralNetworks(wordsDataset);
neuralNetwork.Initialize();

var trainingSet = biagramsModel.CreateTraining();
var inputMatrix = SamplingUtils.OneHotMatrix(trainingSet.xs, 27);  // [N, 27]
Console.WriteLine($"Training set dimension: {trainingSet.xs.Length}");

for (var loop = 0; loop < 50; loop++)
{
    var logits = neuralNetwork.Forward(inputMatrix);   // [N, 27]
    var probs  = logits.Softmax();                     // [N, 27]
    var lossNN = probs.NLL(trainingSet.ys);            // [1, 1]  mean NLL

    Console.WriteLine($"Step {loop}: loss={lossNN.Data[0]:F4}");

    lossNN.Backpropagation();

    foreach (var m in neuralNetwork.ParameterMatrices)
        m.ApplyGradients(0.1, 1.0);
}

neuralNetwork.Generate(5);
neuralNetwork.SaveParameters();

Console.WriteLine("");

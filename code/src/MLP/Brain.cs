using System;
using System.Linq;

namespace TinyBrain;

public class Brain
{
    public string Id { get; }
    public Layer[] Layers { get; }

    public Brain(string id, int inputSize, int[] layerSizes, ActivationType activationType = ActivationType.Tanh)
        : this(id, inputSize, layerSizes, Enumerable.Repeat(activationType, layerSizes.Length).ToArray()) { }

    public Brain(string id, int inputSize, int[] layerSizes, ActivationType[] activationTypes)
    {
        if (activationTypes.Length != layerSizes.Length)
            throw new ArgumentException("activationTypes length must match layerSizes length");

        Id = id;
        int[] sizes = [inputSize, ..layerSizes];
        Layers = new Layer[sizes.Length - 1];
        for (var i = 0; i < Layers.Length; i++)
            Layers[i] = new Layer(sizes[i], sizes[i + 1], activationTypes[i]);
    }

    public Operand[] ParameterMatrices => Layers.SelectMany(l => l.ParameterMatrices).ToArray();

    public Operand Forward(Operand input)
    {
        foreach (var layer in Layers)
            layer.ZeroGradients();

        var current = input;
        foreach (var layer in Layers)
            current = layer.Forward(current);
        return current;
    }
}

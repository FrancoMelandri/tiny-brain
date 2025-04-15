using System.Linq;

namespace TinyBrain;

public class Layer(string id, int numberOfInputs, int numberOfNeurons)
{
    public string Id { get; } = id;

    public Neuron[] Neurons { get; } = new Neuron[numberOfNeurons]
        .Select((_, index) => new Neuron($"{id}-{index}", numberOfInputs))
        .ToArray();

    public Operand[] Parameters => Neurons
            .SelectMany(_ => _.Parameters)
            .ToArray();

    public Operand[] Forward(Operand[] inputs)
        => Neurons.Select(neuron => neuron.Forward(inputs)).ToArray();
}
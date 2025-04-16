using System.Linq;
using TinyFp;
using TinyFp.Extensions;

namespace TinyBrain;

public class Layer(string id, int numberOfInputs,
    int numberOfNeurons,
    ActivationType activationType)
{
    public string Id { get; } = id;

    public Neuron[] Neurons { get; } = new Neuron[numberOfNeurons]
        .Select((_, index) => new Neuron($"{id}-{index}", numberOfInputs, activationType))
        .ToArray();

    public Operand[] Parameters => Neurons
            .SelectMany(_ => _.Parameters)
            .ToArray();

    private Unit ZeroGradient()
        => Parameters.ForEach(_ => _.Gradient = 0);

    public Operand[] Forward(Operand[] inputs)
        => ZeroGradient()
            .Map(_ => Neurons.Select(neuron => neuron.Forward(inputs)).ToArray());
}
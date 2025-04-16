using System;
using System.Linq;
using TinyFp;
using TinyFp.Extensions;
using static TinyBrain.Constants;

namespace TinyBrain;

public class Neuron(string id, int numberOfInputs) :
    INeuron
{
    public string Id { get; } = id;
    private static readonly Random _random = new();

    private Operand[] _weights = new Operand[numberOfInputs]
        .Select((_, index) =>
            Operand.Of(_random.NextDouble() * 2 - 1, $"{WEIGHT_PREFIX}{ID_SEPARATOR}{id}{ID_SEPARATOR}{index}"))
        .ToArray();

    private Operand _bias = Operand.Of(_random.NextDouble() * 2 - 1, $"{BIAS_PREFIX}{ID_SEPARATOR}{id}");
    
    public Operand[] Weights => _weights;
    public Operand Bias => _bias;

    public Operand[] Parameters => [.._weights, _bias];

    private Operand CellBody(Operand[] inputs)
        => (inputs
            .Select((input, index) => input * Weights[index])
            .Fold(Operand.Of(ZERO), (a, i) => a + i))
            .Map(_ => _ + Bias);

    private static Operand Activation(Operand body)
        => body.Activation();

    public Unit ZeroGradient()
        => Parameters.ForEach(_ => _.Gradient = 0);

    public virtual Operand Forward(Operand[] inputs)
        => ZeroGradient()
            .Map(_ => CellBody(inputs))
            .Map(Activation);
}
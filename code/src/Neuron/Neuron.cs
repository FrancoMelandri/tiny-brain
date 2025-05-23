﻿using System;
using System.Linq;
using TinyFp;
using TinyFp.Extensions;
using static TinyBrain.Constants;

namespace TinyBrain;

public class Neuron(string id, int numberOfInputs, ActivationType activationType) :
    INeuron
{
    public string Id { get; } = id;
    public ActivationType ActivationType { get; } = activationType;

    private static readonly Random _random = new();

    public Operand[] Weights { get; } = new Operand[numberOfInputs]
        .Select((_, index) =>
            Operand.Of(_random.NextDouble() * 2 - 1, $"{WEIGHT_PREFIX}{ID_SEPARATOR}{id}{ID_SEPARATOR}{index}"))
        .ToArray();

    public Operand Bias { get; } = Operand.Of(_random.NextDouble() * 2 - 1, $"{BIAS_PREFIX}{ID_SEPARATOR}{id}");

    public Operand[] Parameters => [..Weights, Bias];

    private Operand CellBody(Operand[] inputs)
        => (inputs
            .Select((input, index) => input * Weights[index])
            .Fold(Operand.Of(ZERO), (a, i) => a + i))
            .Map(_ => _ + Bias);

    private Unit ZeroGradient()
        => Parameters.ForEach(_ => _.Gradient = 0);

    public Operand Forward(Operand[] inputs)
        => ZeroGradient()
            .Map(_ => CellBody(inputs))
            .Map(this.Activation);
}
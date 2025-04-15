﻿using System;
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

    protected Operand InternalForward(Operand[] inputs, Func<string, NeuronSteps, object, Unit> observe)
        => (inputs
            .Select((input, index) => input * Weights[index])
            .Tee(_ => observe(Id, NeuronSteps.WEIGHTS, _))
            .Fold(Operand.Of(ZERO), (a, i) => a + i))
            .Tee(_ => observe(Id, NeuronSteps.SUM, new[] {_, Bias}))
            .Map(_ => _ + Bias)
            .Tee(_ => observe(Id, NeuronSteps.BODY, _))
            .Activation()
            .Tee(_ => observe(Id, NeuronSteps.ACTIVATION, _));

    public Unit ZeroGradient()
        => Parameters.ForEach(_ => _.Gradient = 0);

    public virtual Operand Forward(Operand[] inputs)
        => ZeroGradient()
            .Map(_ => InternalForward(inputs, (_, _, _) => Unit.Default));
}
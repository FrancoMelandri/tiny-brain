﻿using System.Linq;
using TinyFp;
using TinyFp.Extensions;
using static TinyBrain.Constants;

namespace TinyBrain;

public class MLP
{
    public string Id { get;  }
    public Layer[] Layers { get; }

    public MLP(string id,
        int numberOfInputs,
        int[] numberOfNeurons,
        ActivationType activationType = ActivationType.Tanh)
    {
        Id = id;
        int[] totals = [numberOfInputs];
        totals = [..totals, ..numberOfNeurons];
        Layers = new Layer[totals.Length]
            .SkipLast(ONE)
            .Select((input, index) => new Layer($"{index}", totals[index], totals[index + 1], activationType))
            .ToArray();
    }

    public Operand[] Parameters => Layers
        .SelectMany(_ => _.Parameters)
        .ToArray();

    private Unit ZeroGradient()
        => Parameters.ForEach(_ => _.Gradient = 0);

    public Operand[] Forward(Operand[] inputs)
        => ZeroGradient()
            .Map(_ => Layers
                            .Fold(inputs,
                                (a, i) => i.Forward(a)));
}
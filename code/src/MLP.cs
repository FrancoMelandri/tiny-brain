using System.Linq;
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

    public Operand[] Forward(Operand[] inputs)
        => Layers
            .Fold(inputs,
                (a, i) => i.Forward(a));
}
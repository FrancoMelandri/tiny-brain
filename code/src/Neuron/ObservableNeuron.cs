using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using TinyFp;
using TinyFp.Extensions;

namespace TinyBrain;

public class ObservableNeuron(
    string id,
    int numberOfInputs,
    Func<string, string, Unit> observe)
    : Neuron(id, numberOfInputs)
{
    private const string Separator = ", ";
    
    private readonly Dictionary<NeuronSteps, Func<object, string>> _internalObserves = new()
    {
        { NeuronSteps.WEIGHTS, OnWeights },
        { NeuronSteps.SUM, OnSum },
        { NeuronSteps.BODY, OnBody },
        { NeuronSteps.ACTIVATION, OnActivation }
    };

    private Func<string, NeuronSteps, object, Unit> _internalObserve
        => (id, step, output) 
            => _internalObserves[step](output)
                .Map(log =>  $"{step} {log}") 
                .Map(log =>  observe(id, log)); 

    
    private static string OnWeights(object operands)
        => operands.AsOperands()
            .Fold(new StringBuilder(),
                (a, i) => a.Append(i.Data.ToString(CultureInfo.InvariantCulture))
                    .Append(Separator))
            .ToString();

    private static string OnSum(object body)
        => body.AsOperands()
            .Fold(new StringBuilder(),
                (a, i) => a.Append(i.Data.ToString(CultureInfo.InvariantCulture))
                    .Append(Separator))
            .ToString();

    private static string OnBody(object body)
        => body.AsOperand()
            .Map(_ => _.Data.ToString(CultureInfo.InvariantCulture));

    private static string OnActivation(object activation)
        => activation.AsOperand()
            .Map(_ => _.Data.ToString(CultureInfo.InvariantCulture));

    public override Operand Forward(Operand[] inputs)
        => InternalForward(inputs, _internalObserve);
}
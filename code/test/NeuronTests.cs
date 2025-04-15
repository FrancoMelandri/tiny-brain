using System;
using System.Linq;
using Newtonsoft.Json.Serialization;
using Shouldly;
using TinyFp;
using TinyFp.Extensions;

namespace TinyBrain.Test;

public class NeuronTests
{
    [Test]
    public void Neuron_Weights_and_Bias()
    {
        var neuron = new Neuron("test", 3);
        
        neuron.Weights.Length.ShouldBe(3);
        neuron.Weights.Min(_ => _.Data).ShouldBeGreaterThanOrEqualTo(-1);
        neuron.Weights.Max(_ => _.Data).ShouldBeLessThanOrEqualTo(1);
        
        neuron.Bias.Data.ShouldBeGreaterThanOrEqualTo(-1);
        neuron.Bias.Data.ShouldBeLessThanOrEqualTo(1);
        
        neuron.Parameters.Length.ShouldBe(4);
        neuron.Parameters[0].ShouldBe(neuron.Weights[0]);
    }

    [Test]
    public void Forward()
    {
        Func<string, string, Unit> observe = (id, log) => Unit.Default.Tee(__ => Console.WriteLine($"{id}: {log}")); 
        var neuron = new Neuron("test", 3);
        Operand[] operands = [Operand.Of(1), Operand.Of(1), Operand.Of(1)];
        
        var target = Operand.Of(1);
        var o = Operand.Of(-1);
        var loss = Operand.Of(0);
        var step = 0;
        while(true)
        {
            step++;
            
            o = neuron.Forward(operands);

            loss = (o - target).Pow(2);
            
            loss.Backpropagation();
            
            if (loss.Data < 0.0000001 || step > 10000)
                break;

            foreach (var x in neuron.Parameters)
            {
                x.Data += -0.01 * x.Gradient;
            }
        }
        Console.WriteLine($"Step {step}: target={target.Data} o={o.Data} loss={loss.Data}");
    }
}
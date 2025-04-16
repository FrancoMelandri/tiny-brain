using System;
using System.Linq;
using Shouldly;
using TinyFp.Extensions;

namespace TinyBrain.Test;

public class LayerTests
{
    [Test]
    public void Layer_Ok()
    {
        var layer = new Layer("test", 4, 3, ActivationType.Tanh);
        var outs = layer.Forward([Operand.Of(1), Operand.Of(2), Operand.Of(3), Operand.Of(4)]);
        
        layer.Neurons.Length.ShouldBe(3);
        layer.Neurons[0].Weights.Length.ShouldBe(4);
        outs.ShouldNotBeEmpty();
        outs.ForEach(o =>
        {
            o.Data.ShouldBeLessThanOrEqualTo(1);
            o.Data.ShouldBeGreaterThanOrEqualTo(-1);
        });
        
        layer.Parameters.Length.ShouldBe(15);
    }
    
    [Test]
    public void Forward()
    {
        var layer = new Layer("test", 3, 3, ActivationType.Tanh);
        Operand[] operands = [Operand.Of(1), Operand.Of(2), Operand.Of(3)];

        var target = new[]
        {
            Operand.Of(1),Operand.Of(0),Operand.Of(1)
        };
        var o = new[]
        {
            Operand.Of(0),Operand.Of(0),Operand.Of(0)
        };
        var step = 0;
        var loss = Operand.Of(0);
        while(true)
        {
            step++;
            
            o = layer.Forward(operands);
            
            var current = o.Select((item, index) 
                => (item - target[index]).Pow(2));
            loss = current.Fold(Operand.Of(0),
                    (a, i) => a + i);
            
            if (loss.Data < 0.00001)
                break;
            
            loss.Backpropagation();

            foreach (var x in layer.Parameters)
                x.Data += -0.01 * x.Gradient;
        }
        Console.WriteLine($"{step} -> o = {o[0].Data} {o[1].Data} {o[2].Data} loss = {loss.Data}");
    }    
}
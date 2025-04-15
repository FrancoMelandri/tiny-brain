using System;
using Shouldly;

namespace TinyBrain.Test;

public class MLPTrainingTests
{
    [Test]
    public void Training()
    {
        // input instances
        Operand[][] xs = 
        [
            [Operand.Of(2), Operand.Of(3), Operand.Of(-1)],
            [Operand.Of(3), Operand.Of(-1), Operand.Of(0.5)],
            [Operand.Of(0.5), Operand.Of(1), Operand.Of(1)],
            [Operand.Of(1), Operand.Of(1), Operand.Of(-1)],
        ];
        // target
        Operand[] ys =
        [
            Operand.Of(1), Operand.Of(-1), Operand.Of(1), Operand.Of(-1)
        ];
        
        var mpl = new MLP("test", 3, [4, 4, 1]);
        mpl.Parameters.Length.ShouldBe(41);
        
        var step = 0;
        var totalLoss = Operand.Of(0);
        var ypred = new Operand[xs.Length];
        var loss = new Operand[xs.Length];
        while(true)
        {
            step++;

            totalLoss = Operand.Of(0);
            for (int i = 0; i < xs.Length; i++)
            {
                ypred[i] = mpl.Forward(xs[i])[0];
                loss[i] = (ypred[i] - ys[i]).Pow(2);
                totalLoss += loss[i];
            }

            if (totalLoss.Data < 0.00001 || step > 10000)
                break;

            totalLoss.Backpropagation();
            foreach (var x in mpl.Parameters)
                x.Data += -0.1 * x.Gradient;
        }
        
        Console.WriteLine($"totalLoss {step} = {totalLoss.Data}");
        Console.WriteLine($"ypred[0] {ypred[0].Data} ypred[1] {ypred[1].Data} ypred[2] {ypred[2].Data} ypred[3] {ypred[3].Data}");
        Console.WriteLine($"loss[0] {loss[0].Data} loss[1] {loss[1].Data} loss[2] {loss[2].Data} loss[3] {loss[3].Data}");
    }
}
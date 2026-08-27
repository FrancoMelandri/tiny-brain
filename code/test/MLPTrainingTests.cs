using System;
using Shouldly;

namespace TinyBrain.Test;

public class MLPTrainingTests
{
    [Test]
    public void Training()
    {
        // 4 samples, 3 inputs, binary target (2 classes)
        var xs = new float[,]
        {
            { 2,  3, -1 },
            { 3, -1,  0.5f },
            { 0.5f, 1, 1 },
            { 1,  1, -1 }
        };
        var ys = new[] { 0, 1, 0, 1 };  // class indices

        var brain = new Brain("test", 3, [4, 4, 2], ActivationType.Tanh);
        var input = Operand.Of(xs);  // [4, 3]

        float firstLoss = 0, lastLoss = 0;

        for (var step = 0; step < 50; step++)
        {
            var logits = brain.Forward(input);     // [4, 2]
            var probs  = logits.Softmax();         // [4, 2]
            var loss   = probs.NLL(ys);            // [1, 1]

            if (step == 0)  firstLoss = loss.Data[0];
            if (step == 49) lastLoss  = loss.Data[0];

            loss.Backpropagation();

            foreach (var m in brain.ParameterMatrices)
                m.ApplyGradients(0.5f, 1.0f);
        }

        Console.WriteLine($"first_loss={firstLoss:F4}  last_loss={lastLoss:F4}");
        lastLoss.ShouldBeLessThan(firstLoss * 0.9f);  // loss must drop by at least 10%
    }
}

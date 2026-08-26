using System;

namespace TinyBrain;

public class AttentionHead
{
    private readonly Operand _wq;   // [dModel, dHead]
    private readonly Operand _wk;   // [dModel, dHead]
    private readonly Operand _wv;   // [dModel, dHead]
    private readonly Operand _wo;   // [dHead, dModel]

    private readonly bool[,] _causalMask;  // [T, T], upper triangle = true
    private readonly double _scale;        // 1 / sqrt(dHead)

    public AttentionHead(int dModel, int dHead, int contextSize)
    {
        var s = 1.0 / Math.Sqrt(dModel);
        _wq = Operand.OfRandom(dModel, dHead, s);
        _wk = Operand.OfRandom(dModel, dHead, s);
        _wv = Operand.OfRandom(dModel, dHead, s);
        _wo = Operand.OfRandom(dHead, dModel, s);

        _scale = 1.0 / Math.Sqrt(dHead);

        _causalMask = new bool[contextSize, contextSize];
        for (var i = 0; i < contextSize; i++)
            for (var j = 0; j < contextSize; j++)
                _causalMask[i, j] = j > i;
    }

    // x: [T, dModel]  ->  [T, dModel]
    public Operand Forward(Operand x)
    {
        var q = x.MatMul(_wq);                           // [T, dHead]
        var k = x.MatMul(_wk);                           // [T, dHead]
        var v = x.MatMul(_wv);                           // [T, dHead]

        var scores = q.MatMul(k.Transpose())             // [T, T]
                      .Scale(_scale)
                      .MaskFill(_causalMask, -1e9)
                      .Softmax();                        // [T, T]

        var ctx = scores.MatMul(v);                      // [T, dHead]
        var projected = ctx.MatMul(_wo);                 // [T, dModel]
        return projected.Add(x);                         // [T, dModel] residual
    }

    public Operand[] ParameterMatrices => [_wq, _wk, _wv, _wo];

    public void ZeroGradients()
    {
        foreach (var p in ParameterMatrices)
            p.ZeroGradient();
    }

    public double GradientNormSquared()
    {
        var sum = 0.0;
        foreach (var p in ParameterMatrices)
            sum += p.GradientNormSquared();
        return sum;
    }
}

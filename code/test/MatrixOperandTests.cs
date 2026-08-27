using System;
using Shouldly;

namespace TinyBrain.Test;

public class MatrixOperandTests
{
    [Test]
    public void MatMul_Forward_Correct()
    {
        var a = Operand.Of(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var w = Operand.Of(new float[,] { { 7, 8 }, { 9, 10 }, { 11, 12 } });
        var c = a.MatMul(w);

        c.Rows.ShouldBe(2);
        c.Cols.ShouldBe(2);
        c.Data[0 * 2 + 0].ShouldBe(58.0f,  tolerance: 1e-3f);   // 1*7+2*9+3*11
        c.Data[0 * 2 + 1].ShouldBe(64.0f,  tolerance: 1e-3f);   // 1*8+2*10+3*12
        c.Data[1 * 2 + 0].ShouldBe(139.0f, tolerance: 1e-3f);   // 4*7+5*9+6*11
        c.Data[1 * 2 + 1].ShouldBe(154.0f, tolerance: 1e-3f);   // 4*8+5*10+6*12
    }

    [Test]
    public void MatMul_Backward_Correct()
    {
        var a = Operand.Of(new float[,] { { 1.0f, 2.0f } });
        var w = Operand.Of(new float[,] { { 3.0f, 4.0f }, { 5.0f, 6.0f } });
        var loss = a.MatMul(w).Sum();
        loss.Backpropagation();

        // dOut = [[1,1]]; dA = dOut x W^T = [[7, 11]]
        a.Gradient[0].ShouldBe(7.0f, tolerance: 1e-4f);
        a.Gradient[1].ShouldBe(11.0f, tolerance: 1e-4f);

        // dW = A^T x dOut = [[1,1],[2,2]]
        w.Gradient[0 * 2 + 0].ShouldBe(1.0f, tolerance: 1e-4f);
        w.Gradient[0 * 2 + 1].ShouldBe(1.0f, tolerance: 1e-4f);
        w.Gradient[1 * 2 + 0].ShouldBe(2.0f, tolerance: 1e-4f);
        w.Gradient[1 * 2 + 1].ShouldBe(2.0f, tolerance: 1e-4f);
    }

    [Test]
    public void AddBias_Backward_Correct()
    {
        var a = Operand.Of(new float[,] { { 1.0f, 2.0f }, { 3.0f, 4.0f } });
        var b = Operand.Of(new float[,] { { 10.0f, 20.0f } });
        var c = a.AddBias(b);

        c.Data[0 * 2 + 0].ShouldBe(11.0f, tolerance: 1e-4f);
        c.Data[1 * 2 + 1].ShouldBe(24.0f, tolerance: 1e-4f);

        c.Sum().Backpropagation();

        a.Gradient[0 * 2 + 0].ShouldBe(1.0f, tolerance: 1e-4f);
        a.Gradient[1 * 2 + 1].ShouldBe(1.0f, tolerance: 1e-4f);
        b.Gradient[0].ShouldBe(2.0f, tolerance: 1e-4f);
        b.Gradient[1].ShouldBe(2.0f, tolerance: 1e-4f);
    }

    [Test]
    public void Tanh_Forward_Backward_Correct()
    {
        var a = Operand.Of(new float[,] { { 0.0f, 1.0f } });
        var t = a.Tanh();

        t.Data[0].ShouldBe(0.0f, tolerance: 1e-4f);
        t.Data[1].ShouldBe(MathF.Tanh(1.0f), tolerance: 1e-4f);

        t.Sum().Backpropagation();

        a.Gradient[0].ShouldBe(1.0f, tolerance: 1e-4f);
        a.Gradient[1].ShouldBe(1 - MathF.Tanh(1.0f) * MathF.Tanh(1.0f), tolerance: 1e-4f);
    }

    [Test]
    public void Softmax_Rows_Sum_To_One()
    {
        var a = Operand.Of(new float[,] { { 1.0f, 2.0f, 3.0f }, { 0.5f, -1.0f, 2.0f } });
        var s = a.Softmax();

        var row0 = s.Data[0] + s.Data[1] + s.Data[2];
        var row1 = s.Data[3] + s.Data[4] + s.Data[5];
        row0.ShouldBe(1.0f, tolerance: 1e-4f);
        row1.ShouldBe(1.0f, tolerance: 1e-4f);
        for (var j = 0; j < 6; j++)
            s.Data[j].ShouldBeGreaterThan(0);
    }

    [Test]
    public void NLL_Loss_And_Gradient()
    {
        var probs = Operand.Of(new float[,] { { 0.1f, 0.7f, 0.2f } });
        var loss = probs.NLL(1);

        loss.Data[0].ShouldBe(-MathF.Log(0.7f + 1e-6f), tolerance: 1e-4f);
        loss.Backpropagation();

        probs.Gradient[1].ShouldBe(-1.0f / (0.7f + 1e-6f), tolerance: 1e-4f);
        probs.Gradient[0].ShouldBe(0.0f, tolerance: 1e-6f);
        probs.Gradient[2].ShouldBe(0.0f, tolerance: 1e-6f);
    }

    [Test]
    public void MatMul_AddBias_Softmax_NLL_Chain()
    {
        var input = Operand.Of(new float[,] { { 1.0f, 1.0f } });
        var w     = Operand.Of(new float[,] { { 0.1f, 0.2f, 0.3f }, { 0.4f, 0.5f, 0.6f } });
        var b     = Operand.Of(new float[,] { { 0.0f, 0.0f, 0.0f } });

        var loss = input.MatMul(w).AddBias(b).Softmax().NLL(0);
        loss.Backpropagation();

        var wGradNonZero = false;
        for (var i = 0; i < w.Gradient.Length; i++)
            if (MathF.Abs(w.Gradient[i]) > 1e-6f) wGradNonZero = true;
        wGradNonZero.ShouldBeTrue();

        loss.Data[0].ShouldBeGreaterThan(0);
    }

    [Test]
    public void Transpose_Forward_Correct()
    {
        var a = Operand.Of(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } }); // [2,3]
        var t = a.Transpose();                                             // [3,2]

        t.Rows.ShouldBe(3);
        t.Cols.ShouldBe(2);
        t.Data[0 * 2 + 0].ShouldBe(1.0f, tolerance: 1e-4f); // [0,0]
        t.Data[0 * 2 + 1].ShouldBe(4.0f, tolerance: 1e-4f); // [0,1]
        t.Data[1 * 2 + 0].ShouldBe(2.0f, tolerance: 1e-4f); // [1,0]
        t.Data[1 * 2 + 1].ShouldBe(5.0f, tolerance: 1e-4f); // [1,1]
        t.Data[2 * 2 + 0].ShouldBe(3.0f, tolerance: 1e-4f); // [2,0]
        t.Data[2 * 2 + 1].ShouldBe(6.0f, tolerance: 1e-4f); // [2,1]
    }

    [Test]
    public void Transpose_Backward_Correct()
    {
        var a = Operand.Of(new float[,] { { 1.0f, 2.0f }, { 3.0f, 4.0f } }); // [2,2]
        a.Transpose().Sum().Backpropagation();

        for (var i = 0; i < 4; i++)
            a.Gradient[i].ShouldBe(1.0f, tolerance: 1e-4f);
    }

    [Test]
    public void Scale_Forward_Backward_Correct()
    {
        var a = Operand.Of(new float[,] { { 1.0f, 2.0f }, { 3.0f, 4.0f } });
        var s = a.Scale(0.5f);

        s.Data[0].ShouldBe(0.5f, tolerance: 1e-4f);
        s.Data[3].ShouldBe(2.0f, tolerance: 1e-4f);

        s.Sum().Backpropagation();

        for (var i = 0; i < 4; i++)
            a.Gradient[i].ShouldBe(0.5f, tolerance: 1e-4f);
    }

    [Test]
    public void Add_Forward_Backward_Correct()
    {
        var a = Operand.Of(new float[,] { { 1.0f, 2.0f }, { 3.0f, 4.0f } });
        var b = Operand.Of(new float[,] { { 10.0f, 20.0f }, { 30.0f, 40.0f } });
        var c = a.Add(b);

        c.Data[0].ShouldBe(11.0f, tolerance: 1e-4f);
        c.Data[3].ShouldBe(44.0f, tolerance: 1e-4f);

        c.Sum().Backpropagation();

        for (var i = 0; i < 4; i++)
        {
            a.Gradient[i].ShouldBe(1.0f, tolerance: 1e-4f);
            b.Gradient[i].ShouldBe(1.0f, tolerance: 1e-4f);
        }
    }

    [Test]
    public void MaskFill_Forward_Backward_Correct()
    {
        var mask = new bool[,] { { false, true }, { false, false } };
        var a = Operand.Of(new float[,] { { 1.0f, 2.0f }, { 3.0f, 4.0f } });
        var m = a.MaskFill(mask, -1e9f);

        m.Data[0 * 2 + 0].ShouldBe(1.0f,   tolerance: 1e-4f); // not masked
        m.Data[0 * 2 + 1].ShouldBe(-1e9f,  tolerance: 1e3f);  // masked
        m.Data[1 * 2 + 0].ShouldBe(3.0f,   tolerance: 1e-4f); // not masked
        m.Data[1 * 2 + 1].ShouldBe(4.0f,   tolerance: 1e-4f); // not masked

        m.Sum().Backpropagation();

        a.Gradient[0 * 2 + 0].ShouldBe(1.0f, tolerance: 1e-4f); // passes through
        a.Gradient[0 * 2 + 1].ShouldBe(0.0f, tolerance: 1e-6f); // masked — zero grad
        a.Gradient[1 * 2 + 0].ShouldBe(1.0f, tolerance: 1e-4f);
        a.Gradient[1 * 2 + 1].ShouldBe(1.0f, tolerance: 1e-4f);
    }

    [Test]
    public void SliceRow_Forward_Backward_Correct()
    {
        var a = Operand.Of(new float[,] { { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } }); // [2,3]
        var s = a.SliceRow(1);                                                                // row 1 -> [1,3]

        s.Rows.ShouldBe(1);
        s.Cols.ShouldBe(3);
        s.Data[0].ShouldBe(4.0f, tolerance: 1e-4f);
        s.Data[1].ShouldBe(5.0f, tolerance: 1e-4f);
        s.Data[2].ShouldBe(6.0f, tolerance: 1e-4f);

        s.Sum().Backpropagation();

        // gradient flows only to row 1
        a.Gradient[0].ShouldBe(0.0f, tolerance: 1e-6f);
        a.Gradient[1].ShouldBe(0.0f, tolerance: 1e-6f);
        a.Gradient[2].ShouldBe(0.0f, tolerance: 1e-6f);
        a.Gradient[3].ShouldBe(1.0f, tolerance: 1e-4f);
        a.Gradient[4].ShouldBe(1.0f, tolerance: 1e-4f);
        a.Gradient[5].ShouldBe(1.0f, tolerance: 1e-4f);
    }
}

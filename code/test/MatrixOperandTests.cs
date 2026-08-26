using System;
using Shouldly;

namespace TinyBrain.Test;

public class MatrixOperandTests
{
    [Test]
    public void MatMul_Forward_Correct()
    {
        var a = Operand.Of(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var w = Operand.Of(new double[,] { { 7, 8 }, { 9, 10 }, { 11, 12 } });
        var c = a.MatMul(w);

        c.Rows.ShouldBe(2);
        c.Cols.ShouldBe(2);
        c.Data[0 * 2 + 0].ShouldBe(58.0);   // 1*7+2*9+3*11
        c.Data[0 * 2 + 1].ShouldBe(64.0);   // 1*8+2*10+3*12
        c.Data[1 * 2 + 0].ShouldBe(139.0);  // 4*7+5*9+6*11
        c.Data[1 * 2 + 1].ShouldBe(154.0);  // 4*8+5*10+6*12
    }

    [Test]
    public void MatMul_Backward_Correct()
    {
        var a = Operand.Of(new double[,] { { 1.0, 2.0 } });
        var w = Operand.Of(new double[,] { { 3.0, 4.0 }, { 5.0, 6.0 } });
        var loss = a.MatMul(w).Sum();
        loss.Backpropagation();

        // dOut = [[1,1]]; dA = dOut x W^T = [[7, 11]]
        a.Gradient[0].ShouldBe(7.0, tolerance: 1e-10);
        a.Gradient[1].ShouldBe(11.0, tolerance: 1e-10);
        // dW = A^T x dOut = [[1,1],[2,2]]
        w.Gradient[0 * 2 + 0].ShouldBe(1.0, tolerance: 1e-10);
        w.Gradient[0 * 2 + 1].ShouldBe(1.0, tolerance: 1e-10);
        w.Gradient[1 * 2 + 0].ShouldBe(2.0, tolerance: 1e-10);
        w.Gradient[1 * 2 + 1].ShouldBe(2.0, tolerance: 1e-10);
    }

    [Test]
    public void AddBias_Backward_Correct()
    {
        var a = Operand.Of(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        var b = Operand.Of(new double[,] { { 10.0, 20.0 } });
        var c = a.AddBias(b);

        c.Data[0 * 2 + 0].ShouldBe(11.0);
        c.Data[1 * 2 + 1].ShouldBe(24.0);

        c.Sum().Backpropagation();

        a.Gradient[0 * 2 + 0].ShouldBe(1.0, tolerance: 1e-10);
        a.Gradient[1 * 2 + 1].ShouldBe(1.0, tolerance: 1e-10);
        b.Gradient[0].ShouldBe(2.0, tolerance: 1e-10);
        b.Gradient[1].ShouldBe(2.0, tolerance: 1e-10);
    }

    [Test]
    public void Tanh_Forward_Backward_Correct()
    {
        var a = Operand.Of(new double[,] { { 0.0, 1.0 } });
        var t = a.Tanh();

        t.Data[0].ShouldBe(0.0, tolerance: 1e-10);
        t.Data[1].ShouldBe(Math.Tanh(1.0), tolerance: 1e-10);

        t.Sum().Backpropagation();

        a.Gradient[0].ShouldBe(1.0, tolerance: 1e-10);
        a.Gradient[1].ShouldBe(1 - Math.Tanh(1.0) * Math.Tanh(1.0), tolerance: 1e-10);
    }

    [Test]
    public void Softmax_Rows_Sum_To_One()
    {
        var a = Operand.Of(new double[,] { { 1.0, 2.0, 3.0 }, { 0.5, -1.0, 2.0 } });
        var s = a.Softmax();

        var row0 = s.Data[0] + s.Data[1] + s.Data[2];
        var row1 = s.Data[3] + s.Data[4] + s.Data[5];
        row0.ShouldBe(1.0, tolerance: 1e-10);
        row1.ShouldBe(1.0, tolerance: 1e-10);
        for (var j = 0; j < 6; j++)
            s.Data[j].ShouldBeGreaterThan(0);
    }

    [Test]
    public void NLL_Loss_And_Gradient()
    {
        var probs = Operand.Of(new double[,] { { 0.1, 0.7, 0.2 } });
        var loss = probs.NLL(1);

        loss.Data[0].ShouldBe(-Math.Log(0.7 + 1e-10), tolerance: 1e-8);
        loss.Backpropagation();

        probs.Gradient[1].ShouldBe(-1.0 / (0.7 + 1e-10), tolerance: 1e-8);
        probs.Gradient[0].ShouldBe(0.0);
        probs.Gradient[2].ShouldBe(0.0);
    }

    [Test]
    public void MatMul_AddBias_Softmax_NLL_Chain()
    {
        var input = Operand.Of(new double[,] { { 1.0, 1.0 } });
        var w     = Operand.Of(new double[,] { { 0.1, 0.2, 0.3 }, { 0.4, 0.5, 0.6 } });
        var b     = Operand.Of(new double[,] { { 0.0, 0.0, 0.0 } });

        var loss = input.MatMul(w).AddBias(b).Softmax().NLL(0);
        loss.Backpropagation();

        var wGradNonZero = false;
        for (var i = 0; i < w.Gradient.Length; i++)
            if (Math.Abs(w.Gradient[i]) > 1e-10) wGradNonZero = true;
        wGradNonZero.ShouldBeTrue();

        loss.Data[0].ShouldBeGreaterThan(0);
    }

    [Test]
    public void Transpose_Forward_Correct()
    {
        var a = Operand.Of(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } }); // [2,3]
        var t = a.Transpose();                                             // [3,2]

        t.Rows.ShouldBe(3);
        t.Cols.ShouldBe(2);
        t.Data[0 * 2 + 0].ShouldBe(1.0); // [0,0]
        t.Data[0 * 2 + 1].ShouldBe(4.0); // [0,1]
        t.Data[1 * 2 + 0].ShouldBe(2.0); // [1,0]
        t.Data[1 * 2 + 1].ShouldBe(5.0); // [1,1]
        t.Data[2 * 2 + 0].ShouldBe(3.0); // [2,0]
        t.Data[2 * 2 + 1].ShouldBe(6.0); // [2,1]
    }

    [Test]
    public void Transpose_Backward_Correct()
    {
        // Transpose is its own inverse: grad flows back correctly
        var a = Operand.Of(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } }); // [2,2]
        a.Transpose().Sum().Backpropagation();

        for (var i = 0; i < 4; i++)
            a.Gradient[i].ShouldBe(1.0, tolerance: 1e-10);
    }

    [Test]
    public void Scale_Forward_Backward_Correct()
    {
        var a = Operand.Of(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        var s = a.Scale(0.5);

        s.Data[0].ShouldBe(0.5, tolerance: 1e-10);
        s.Data[3].ShouldBe(2.0, tolerance: 1e-10);

        s.Sum().Backpropagation();

        for (var i = 0; i < 4; i++)
            a.Gradient[i].ShouldBe(0.5, tolerance: 1e-10);
    }

    [Test]
    public void Add_Forward_Backward_Correct()
    {
        var a = Operand.Of(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        var b = Operand.Of(new double[,] { { 10.0, 20.0 }, { 30.0, 40.0 } });
        var c = a.Add(b);

        c.Data[0].ShouldBe(11.0, tolerance: 1e-10);
        c.Data[3].ShouldBe(44.0, tolerance: 1e-10);

        c.Sum().Backpropagation();

        for (var i = 0; i < 4; i++)
        {
            a.Gradient[i].ShouldBe(1.0, tolerance: 1e-10);
            b.Gradient[i].ShouldBe(1.0, tolerance: 1e-10);
        }
    }

    [Test]
    public void MaskFill_Forward_Backward_Correct()
    {
        // upper-triangle causal mask for 2x2
        var mask = new bool[,] { { false, true }, { false, false } };
        var a = Operand.Of(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        var m = a.MaskFill(mask, -1e9);

        m.Data[0 * 2 + 0].ShouldBe(1.0,   tolerance: 1e-10); // not masked
        m.Data[0 * 2 + 1].ShouldBe(-1e9,  tolerance: 1e-3);  // masked
        m.Data[1 * 2 + 0].ShouldBe(3.0,   tolerance: 1e-10); // not masked
        m.Data[1 * 2 + 1].ShouldBe(4.0,   tolerance: 1e-10); // not masked

        m.Sum().Backpropagation();

        a.Gradient[0 * 2 + 0].ShouldBe(1.0, tolerance: 1e-10); // passes through
        a.Gradient[0 * 2 + 1].ShouldBe(0.0, tolerance: 1e-10); // masked — zero grad
        a.Gradient[1 * 2 + 0].ShouldBe(1.0, tolerance: 1e-10);
        a.Gradient[1 * 2 + 1].ShouldBe(1.0, tolerance: 1e-10);
    }

    [Test]
    public void SliceRow_Forward_Backward_Correct()
    {
        var a = Operand.Of(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } }); // [2,3]
        var s = a.SliceRow(1);                                                          // row 1 -> [1,3]

        s.Rows.ShouldBe(1);
        s.Cols.ShouldBe(3);
        s.Data[0].ShouldBe(4.0, tolerance: 1e-10);
        s.Data[1].ShouldBe(5.0, tolerance: 1e-10);
        s.Data[2].ShouldBe(6.0, tolerance: 1e-10);

        s.Sum().Backpropagation();

        // gradient flows only to row 1
        a.Gradient[0].ShouldBe(0.0, tolerance: 1e-10);
        a.Gradient[1].ShouldBe(0.0, tolerance: 1e-10);
        a.Gradient[2].ShouldBe(0.0, tolerance: 1e-10);
        a.Gradient[3].ShouldBe(1.0, tolerance: 1e-10);
        a.Gradient[4].ShouldBe(1.0, tolerance: 1e-10);
        a.Gradient[5].ShouldBe(1.0, tolerance: 1e-10);
    }
}

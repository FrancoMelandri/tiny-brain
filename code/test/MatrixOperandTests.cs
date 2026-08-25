using System;
using Shouldly;

namespace TinyBrain.Test;

public class MatrixOperandTests
{
    [Test]
    public void MatMul_Forward_Correct()
    {
        var a = MatrixOperand.Of(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var w = MatrixOperand.Of(new double[,] { { 7, 8 }, { 9, 10 }, { 11, 12 } });
        var c = a.MatMul(w);

        c.Rows.ShouldBe(2);
        c.Cols.ShouldBe(2);
        c.Data[0, 0].ShouldBe(58.0);   // 1*7+2*9+3*11
        c.Data[0, 1].ShouldBe(64.0);   // 1*8+2*10+3*12
        c.Data[1, 0].ShouldBe(139.0);  // 4*7+5*9+6*11
        c.Data[1, 1].ShouldBe(154.0);  // 4*8+5*10+6*12
    }

    [Test]
    public void MatMul_Backward_Correct()
    {
        // [1,2] x [2,2] = [1,2], sum → loss
        var a = MatrixOperand.Of(new double[,] { { 1.0, 2.0 } });
        var w = MatrixOperand.Of(new double[,] { { 3.0, 4.0 }, { 5.0, 6.0 } });
        var loss = a.MatMul(w).Sum();
        loss.Backpropagation();

        // dOut = [[1,1]]
        // dA = dOut x W^T = [[1,1]] x [[3,5],[4,6]] = [[7, 11]]
        a.Gradient[0, 0].ShouldBe(7.0, tolerance: 1e-10);
        a.Gradient[0, 1].ShouldBe(11.0, tolerance: 1e-10);
        // dW = A^T x dOut = [[1],[2]] x [[1,1]] = [[1,1],[2,2]]
        w.Gradient[0, 0].ShouldBe(1.0, tolerance: 1e-10);
        w.Gradient[0, 1].ShouldBe(1.0, tolerance: 1e-10);
        w.Gradient[1, 0].ShouldBe(2.0, tolerance: 1e-10);
        w.Gradient[1, 1].ShouldBe(2.0, tolerance: 1e-10);
    }

    [Test]
    public void AddBias_Backward_Correct()
    {
        var a = MatrixOperand.Of(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        var b = MatrixOperand.Of(new double[,] { { 10.0, 20.0 } });
        var c = a.AddBias(b);

        c.Data[0, 0].ShouldBe(11.0);
        c.Data[1, 1].ShouldBe(24.0);

        var loss = c.Sum();
        loss.Backpropagation();

        // dA = ones (dOut is all 1s from Sum)
        a.Gradient[0, 0].ShouldBe(1.0, tolerance: 1e-10);
        a.Gradient[1, 1].ShouldBe(1.0, tolerance: 1e-10);
        // db = sum over rows = [2, 2]
        b.Gradient[0, 0].ShouldBe(2.0, tolerance: 1e-10);
        b.Gradient[0, 1].ShouldBe(2.0, tolerance: 1e-10);
    }

    [Test]
    public void Tanh_Forward_Backward_Correct()
    {
        var a = MatrixOperand.Of(new double[,] { { 0.0, 1.0 } });
        var t = a.Tanh();

        t.Data[0, 0].ShouldBe(0.0, tolerance: 1e-10);
        t.Data[0, 1].ShouldBe(Math.Tanh(1.0), tolerance: 1e-10);

        t.Sum().Backpropagation();

        // d tanh(x)/dx = 1 - tanh(x)^2; dOut from Sum = 1
        a.Gradient[0, 0].ShouldBe(1.0, tolerance: 1e-10);  // 1 - tanh(0)^2 = 1
        a.Gradient[0, 1].ShouldBe(1 - Math.Tanh(1.0) * Math.Tanh(1.0), tolerance: 1e-10);
    }

    [Test]
    public void Softmax_Rows_Sum_To_One()
    {
        var a = MatrixOperand.Of(new double[,] { { 1.0, 2.0, 3.0 }, { 0.5, -1.0, 2.0 } });
        var s = a.Softmax();

        var row0 = s.Data[0, 0] + s.Data[0, 1] + s.Data[0, 2];
        var row1 = s.Data[1, 0] + s.Data[1, 1] + s.Data[1, 2];
        row0.ShouldBe(1.0, tolerance: 1e-10);
        row1.ShouldBe(1.0, tolerance: 1e-10);
        for (var j = 0; j < 3; j++)
        {
            s.Data[0, j].ShouldBeGreaterThan(0);
            s.Data[1, j].ShouldBeGreaterThan(0);
        }
    }

    [Test]
    public void NLL_Loss_And_Gradient()
    {
        var probs = MatrixOperand.Of(new double[,] { { 0.1, 0.7, 0.2 } });
        var loss = probs.NLL(1);

        loss.Data[0, 0].ShouldBe(-Math.Log(0.7 + 1e-10), tolerance: 1e-8);

        loss.Backpropagation();

        probs.Gradient[0, 1].ShouldBe(-1.0 / (0.7 + 1e-10), tolerance: 1e-8);
        probs.Gradient[0, 0].ShouldBe(0.0);
        probs.Gradient[0, 2].ShouldBe(0.0);
    }

    [Test]
    public void MatMul_AddBias_Softmax_NLL_Chain()
    {
        var input = MatrixOperand.Of(new double[,] { { 1.0, 1.0 } });
        var w     = MatrixOperand.Of(new double[,] { { 0.1, 0.2, 0.3 }, { 0.4, 0.5, 0.6 } });
        var b     = MatrixOperand.Of(new double[,] { { 0.0, 0.0, 0.0 } });

        var loss = input.MatMul(w).AddBias(b).Softmax().NLL(0);
        loss.Backpropagation();

        var wGradNonZero = false;
        for (var i = 0; i < 2; i++)
            for (var j = 0; j < 3; j++)
                if (Math.Abs(w.Gradient[i, j]) > 1e-10) wGradNonZero = true;
        wGradNonZero.ShouldBeTrue();

        // Loss should be positive (NLL is always >= 0)
        loss.Data[0, 0].ShouldBeGreaterThan(0);
    }
}

using System;
using System.Collections.Generic;
using System.Numerics.Tensors;

namespace TinyBrain;

public class MatrixOperand
{
    public double[] Data { get; }
    public double[] Gradient { get; }

    private readonly int _rows;
    private readonly int _cols;

    public int Rows => _rows;
    public int Cols => _cols;

    private readonly (MatrixOperand Left, MatrixOperand Right) _previous;
    private Action _backward;

    private MatrixOperand(double[] data, int rows, int cols)
    {
        Data = data;
        Gradient = new double[rows * cols];
        _rows = rows;
        _cols = cols;
        _previous = (null, null);
        _backward = () => { };
    }

    private MatrixOperand(double[] data, int rows, int cols,
        (MatrixOperand Left, MatrixOperand Right) previous)
        : this(data, rows, cols)
    {
        _previous = previous;
    }

    public static MatrixOperand Of(double[,] src)
    {
        var rows = src.GetLength(0);
        var cols = src.GetLength(1);
        var flat = new double[rows * cols];
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                flat[i * cols + j] = src[i, j];
        return new MatrixOperand(flat, rows, cols);
    }

    public static MatrixOperand OfZero(int rows, int cols)
        => new(new double[rows * cols], rows, cols);

    public static MatrixOperand OfRandom(int rows, int cols, double scale = 0.1)
    {
        var rng = new Random();
        var data = new double[rows * cols];
        for (var i = 0; i < rows * cols; i++)
            data[i] = (rng.NextDouble() * 2 - 1) * scale;
        return new MatrixOperand(data, rows, cols);
    }

    // Allows leaf nodes to register a custom backward (e.g. embedding bridge)
    public void SetBackward(Action<double[]> backward)
        => _backward = () => backward(Gradient);

    // [m,k] x [k,n] -> [m,n]
    public MatrixOperand MatMul(MatrixOperand w)
    {
        var m = _rows;
        var k = _cols;
        var n = w._cols;
        var outFlat = new double[m * n];

        // Forward: for each (i,p), accumulate A[i,p] * W_row_p into Out_row_i
        for (var i = 0; i < m; i++)
            for (var p = 0; p < k; p++)
                TensorPrimitives.MultiplyAdd(
                    new ReadOnlySpan<double>(w.Data, p * n, n),
                    Data[i * k + p],
                    new ReadOnlySpan<double>(outFlat, i * n, n),
                    new Span<double>(outFlat, i * n, n));

        var result = new MatrixOperand(outFlat, m, n, (this, w));
        var aData = Data;
        var wData = w.Data;
        var aGrad = Gradient;
        var wGrad = w.Gradient;
        var dOut = result.Gradient;

        result._backward = () =>
        {
            // dA[i,p] = Dot(dOut_row_i, W_row_p)
            for (var i = 0; i < m; i++)
                for (var p = 0; p < k; p++)
                    aGrad[i * k + p] += TensorPrimitives.Dot(
                        new ReadOnlySpan<double>(dOut, i * n, n),
                        new ReadOnlySpan<double>(wData, p * n, n));

            // dW[p,j] += A[i,p] * dOut[i,j]  (MultiplyAdd over rows)
            for (var i = 0; i < m; i++)
                for (var p = 0; p < k; p++)
                    TensorPrimitives.MultiplyAdd(
                        new ReadOnlySpan<double>(dOut, i * n, n),
                        aData[i * k + p],
                        new ReadOnlySpan<double>(wGrad, p * n, n),
                        new Span<double>(wGrad, p * n, n));
        };
        return result;
    }

    // [m,n] + [1,n] broadcast -> [m,n]
    public MatrixOperand AddBias(MatrixOperand b)
    {
        var m = _rows;
        var n = _cols;
        var outFlat = new double[m * n];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
                outFlat[i * n + j] = Data[i * n + j] + b.Data[j];

        var result = new MatrixOperand(outFlat, m, n, (this, b));
        var aGrad = Gradient;
        var bGrad = b.Gradient;
        var dOut = result.Gradient;

        result._backward = () =>
        {
            for (var i = 0; i < m; i++)
                for (var j = 0; j < n; j++)
                {
                    aGrad[i * n + j] += dOut[i * n + j];
                    bGrad[j] += dOut[i * n + j];
                }
        };
        return result;
    }

    // elementwise tanh
    public MatrixOperand Tanh()
    {
        var len = _rows * _cols;
        var outFlat = new double[len];
        for (var i = 0; i < len; i++)
            outFlat[i] = Math.Tanh(Data[i]);

        var result = new MatrixOperand(outFlat, _rows, _cols, (this, null));
        var inGrad = Gradient;
        var dOut = result.Gradient;

        result._backward = () =>
        {
            for (var i = 0; i < len; i++)
                inGrad[i] += (1 - outFlat[i] * outFlat[i]) * dOut[i];
        };
        return result;
    }

    // per-row stable softmax
    public MatrixOperand Softmax()
    {
        var m = _rows;
        var n = _cols;
        var outFlat = new double[m * n];
        for (var i = 0; i < m; i++)
        {
            var rowStart = i * n;
            var max = double.NegativeInfinity;
            for (var j = 0; j < n; j++)
                if (Data[rowStart + j] > max) max = Data[rowStart + j];
            var sum = 0.0;
            for (var j = 0; j < n; j++)
            {
                outFlat[rowStart + j] = Math.Exp(Data[rowStart + j] - max);
                sum += outFlat[rowStart + j];
            }
            for (var j = 0; j < n; j++)
                outFlat[rowStart + j] /= sum;
        }

        var result = new MatrixOperand(outFlat, m, n, (this, null));
        var inGrad = Gradient;
        var dOut = result.Gradient;

        result._backward = () =>
        {
            for (var i = 0; i < m; i++)
            {
                var rowStart = i * n;
                var dot = TensorPrimitives.Dot(
                    new ReadOnlySpan<double>(dOut, rowStart, n),
                    new ReadOnlySpan<double>(outFlat, rowStart, n));
                for (var j = 0; j < n; j++)
                    inGrad[rowStart + j] += outFlat[rowStart + j] * (dOut[rowStart + j] - dot);
            }
        };
        return result;
    }

    // -log(Data[0, targetCol])  ->  [1,1]
    public MatrixOperand NLL(int targetCol)
    {
        var p = Data[targetCol];
        var loss = -Math.Log(p + 1e-10);
        var result = new MatrixOperand(new double[] { loss }, 1, 1, (this, null));
        var inGrad = Gradient;
        var dOut = result.Gradient;

        result._backward = () =>
        {
            inGrad[targetCol] += dOut[0] * (-1.0 / (p + 1e-10));
        };
        return result;
    }

    // Sum all elements -> [1,1]
    public MatrixOperand Sum()
    {
        var total = 0.0;
        var len = _rows * _cols;
        for (var i = 0; i < len; i++) total += Data[i];

        var result = new MatrixOperand(new double[] { total }, 1, 1, (this, null));
        var inGrad = Gradient;
        var dOut = result.Gradient;

        result._backward = () =>
        {
            for (var i = 0; i < len; i++)
                inGrad[i] += dOut[0];
        };
        return result;
    }

    public void Backpropagation()
    {
        Gradient[0] = 1.0;
        var ordered = BuildTopological();
        for (var i = ordered.Count - 1; i >= 0; i--)
            ordered[i]._backward();
    }

    public void ZeroGradient()
        => Array.Clear(Gradient, 0, Gradient.Length);

    public double GradientNormSquared()
        => TensorPrimitives.Dot(Gradient, Gradient);

    public void ApplyGradients(double lr, double clipScale)
    {
        var scale = lr * clipScale;
        for (var i = 0; i < Data.Length; i++)
            Data[i] -= scale * Gradient[i];
    }

    private List<MatrixOperand> BuildTopological()
    {
        var visited = new HashSet<MatrixOperand>();
        var ordered = new List<MatrixOperand>();
        var stack = new Stack<MatrixOperand>();
        stack.Push(this);

        while (stack.Count > 0)
        {
            var current = stack.Peek();
            var pushed = false;

            if (TryPushUnvisited(stack, visited, current._previous.Left, ref pushed)) continue;
            if (TryPushUnvisited(stack, visited, current._previous.Right, ref pushed)) continue;

            if (!pushed)
            {
                stack.Pop();
                if (visited.Add(current))
                    ordered.Add(current);
            }
        }
        return ordered;
    }

    private static bool TryPushUnvisited(Stack<MatrixOperand> stack, HashSet<MatrixOperand> visited,
        MatrixOperand node, ref bool pushed)
    {
        if (node != null && !visited.Contains(node))
        {
            stack.Push(node);
            pushed = true;
            return true;
        }
        return false;
    }
}

using System;
using System.Collections.Generic;

namespace TinyBrain;

public class MatrixOperand
{
    public double[,] Data { get; }
    public double[,] Gradient { get; }

    public int Rows => Data.GetLength(0);
    public int Cols => Data.GetLength(1);

    private readonly (MatrixOperand? Left, MatrixOperand? Right) _previous;
    private Action _backward;

    private MatrixOperand(double[,] data)
    {
        Data = data;
        Gradient = new double[data.GetLength(0), data.GetLength(1)];
        _previous = (null, null);
        _backward = () => { };
    }

    private MatrixOperand(double[,] data, (MatrixOperand? Left, MatrixOperand? Right) previous)
        : this(data)
    {
        _previous = previous;
    }

    public static MatrixOperand Of(double[,] data) => new(data);

    // Allows leaf nodes to register a custom backward (e.g. embedding bridge)
    public void SetBackward(Action<double[,]> backward)
        => _backward = () => backward(Gradient);

    public static MatrixOperand OfZero(int rows, int cols) => new(new double[rows, cols]);

    public static MatrixOperand OfRandom(int rows, int cols, double scale = 0.1)
    {
        var rng = new Random();
        var data = new double[rows, cols];
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                data[i, j] = (rng.NextDouble() * 2 - 1) * scale;
        return new MatrixOperand(data);
    }

    // [m,k] x [k,n] -> [m,n]
    public MatrixOperand MatMul(MatrixOperand w)
    {
        var m = Rows;
        var k = Cols;
        var n = w.Cols;
        var outData = new double[m, n];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
                for (var p = 0; p < k; p++)
                    outData[i, j] += Data[i, p] * w.Data[p, j];

        var result = new MatrixOperand(outData, (this, w));
        result._backward = () =>
        {
            // dA = dOut x W^T
            for (var i = 0; i < m; i++)
                for (var p = 0; p < k; p++)
                    for (var j = 0; j < n; j++)
                        Gradient[i, p] += result.Gradient[i, j] * w.Data[p, j];
            // dW = A^T x dOut
            for (var p = 0; p < k; p++)
                for (var j = 0; j < n; j++)
                    for (var i = 0; i < m; i++)
                        w.Gradient[p, j] += Data[i, p] * result.Gradient[i, j];
        };
        return result;
    }

    // [m,n] + [1,n] (broadcast) -> [m,n]
    public MatrixOperand AddBias(MatrixOperand b)
    {
        var m = Rows;
        var n = Cols;
        var outData = new double[m, n];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
                outData[i, j] = Data[i, j] + b.Data[0, j];

        var result = new MatrixOperand(outData, (this, b));
        result._backward = () =>
        {
            for (var i = 0; i < m; i++)
                for (var j = 0; j < n; j++)
                {
                    Gradient[i, j] += result.Gradient[i, j];
                    b.Gradient[0, j] += result.Gradient[i, j];
                }
        };
        return result;
    }

    // elementwise tanh
    public MatrixOperand Tanh()
    {
        var m = Rows;
        var n = Cols;
        var outData = new double[m, n];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
                outData[i, j] = Math.Tanh(Data[i, j]);

        var result = new MatrixOperand(outData, (this, null));
        result._backward = () =>
        {
            for (var i = 0; i < m; i++)
                for (var j = 0; j < n; j++)
                    Gradient[i, j] += (1 - result.Data[i, j] * result.Data[i, j]) * result.Gradient[i, j];
        };
        return result;
    }

    // per-row stable softmax
    public MatrixOperand Softmax()
    {
        var m = Rows;
        var n = Cols;
        var outData = new double[m, n];
        for (var i = 0; i < m; i++)
        {
            var max = double.NegativeInfinity;
            for (var j = 0; j < n; j++)
                if (Data[i, j] > max) max = Data[i, j];
            var sum = 0.0;
            for (var j = 0; j < n; j++)
            {
                outData[i, j] = Math.Exp(Data[i, j] - max);
                sum += outData[i, j];
            }
            for (var j = 0; j < n; j++)
                outData[i, j] /= sum;
        }

        var result = new MatrixOperand(outData, (this, null));
        result._backward = () =>
        {
            // Jacobian-vector product: dIn[i] = p[i] * (dOut[i] - dot(dOut, p))
            for (var i = 0; i < m; i++)
            {
                var dot = 0.0;
                for (var j = 0; j < n; j++)
                    dot += result.Gradient[i, j] * result.Data[i, j];
                for (var j = 0; j < n; j++)
                    Gradient[i, j] += result.Data[i, j] * (result.Gradient[i, j] - dot);
            }
        };
        return result;
    }

    // Negative log-likelihood on row 0: -log(Data[0, targetCol])
    // Returns MatrixOperand [1,1]
    public MatrixOperand NLL(int targetCol)
    {
        var loss = -Math.Log(Data[0, targetCol] + 1e-10);
        var outData = new double[1, 1] { { loss } };

        var result = new MatrixOperand(outData, (this, null));
        result._backward = () =>
        {
            Gradient[0, targetCol] += result.Gradient[0, 0] * (-1.0 / (Data[0, targetCol] + 1e-10));
        };
        return result;
    }

    // Sums all elements into a [1,1] scalar — useful as a loss aggregator and in tests
    public MatrixOperand Sum()
    {
        var total = 0.0;
        var m = Rows;
        var n = Cols;
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
                total += Data[i, j];

        var result = new MatrixOperand(new double[1, 1] { { total } }, (this, null));
        result._backward = () =>
        {
            for (var i = 0; i < m; i++)
                for (var j = 0; j < n; j++)
                    Gradient[i, j] += result.Gradient[0, 0];
        };
        return result;
    }

    public void Backpropagation()
    {
        Gradient[0, 0] = 1.0;
        var ordered = BuildTopological();
        for (var i = ordered.Count - 1; i >= 0; i--)
            ordered[i]._backward();
    }

    public void ZeroGradient()
    {
        var m = Rows;
        var n = Cols;
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
                Gradient[i, j] = 0;
    }

    public double GradientNormSquared()
    {
        var sum = 0.0;
        var m = Rows;
        var n = Cols;
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
                sum += Gradient[i, j] * Gradient[i, j];
        return sum;
    }

    public void ApplyGradients(double lr, double clipScale)
    {
        var m = Rows;
        var n = Cols;
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
                Data[i, j] -= lr * Gradient[i, j] * clipScale;
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
            var pushedChild = false;

            if (_TryPushUnvisited(stack, visited, current._previous.Left, ref pushedChild)) continue;
            if (_TryPushUnvisited(stack, visited, current._previous.Right, ref pushedChild)) continue;

            if (!pushedChild)
            {
                stack.Pop();
                if (visited.Add(current))
                    ordered.Add(current);
            }
        }
        return ordered;
    }

    private static bool _TryPushUnvisited(Stack<MatrixOperand> stack, HashSet<MatrixOperand> visited,
        MatrixOperand? node, ref bool pushed)
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

using System;
using System.Collections.Generic;
using System.Numerics.Tensors;

namespace TinyBrain;

public class Operand
{
    public float[] Data { get; }
    public float[] Gradient { get; }

    private readonly int _rows;
    private readonly int _cols;

    public int Rows => _rows;
    public int Cols => _cols;

    private readonly (Operand Left, Operand Right) _previous;
    private Action _backward;

    private Operand(float[] data, int rows, int cols)
    {
        Data = data;
        Gradient = new float[rows * cols];
        _rows = rows;
        _cols = cols;
        _previous = (null, null);
        _backward = () => { };
    }

    private Operand(float[] data, int rows, int cols,
        (Operand Left, Operand Right) previous)
        : this(data, rows, cols)
    {
        _previous = previous;
    }

    public static Operand Of(float[,] src)
    {
        var rows = src.GetLength(0);
        var cols = src.GetLength(1);
        var flat = new float[rows * cols];
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                flat[i * cols + j] = src[i, j];
        return new Operand(flat, rows, cols);
    }

    public static Operand OfZero(int rows, int cols)
        => new(new float[rows * cols], rows, cols);

    public static Operand OfRandom(int rows, int cols, float scale = 0.1f)
    {
        var rng = new Random();
        var data = new float[rows * cols];
        for (var i = 0; i < rows * cols; i++)
            data[i] = ((float)rng.NextDouble() * 2 - 1) * scale;
        return new Operand(data, rows, cols);
    }

    // Allows leaf nodes to register a custom backward (e.g. embedding bridge)
    public void SetBackward(Action<float[]> backward)
        => _backward = () => backward(Gradient);

    // [m,k] x [k,n] -> [m,n]
    public Operand MatMul(Operand w)
    {
        var m = _rows;
        var k = _cols;
        var n = w._cols;
        var outFlat = new float[m * n];

        // Forward: for each (i,p), accumulate A[i,p] * W_row_p into Out_row_i
        for (var i = 0; i < m; i++)
            for (var p = 0; p < k; p++)
                TensorPrimitives.MultiplyAdd(
                    new ReadOnlySpan<float>(w.Data, p * n, n),
                    Data[i * k + p],
                    new ReadOnlySpan<float>(outFlat, i * n, n),
                    new Span<float>(outFlat, i * n, n));

        var result = new Operand(outFlat, m, n, (this, w));
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
                        new ReadOnlySpan<float>(dOut, i * n, n),
                        new ReadOnlySpan<float>(wData, p * n, n));

            // dW[p,j] += A[i,p] * dOut[i,j]  (MultiplyAdd over rows)
            for (var i = 0; i < m; i++)
                for (var p = 0; p < k; p++)
                    TensorPrimitives.MultiplyAdd(
                        new ReadOnlySpan<float>(dOut, i * n, n),
                        aData[i * k + p],
                        new ReadOnlySpan<float>(wGrad, p * n, n),
                        new Span<float>(wGrad, p * n, n));
        };
        return result;
    }

    // [m,n] + [1,n] broadcast -> [m,n]
    public Operand AddBias(Operand b)
    {
        var m = _rows;
        var n = _cols;
        var outFlat = new float[m * n];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
                outFlat[i * n + j] = Data[i * n + j] + b.Data[j];

        var result = new Operand(outFlat, m, n, (this, b));
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
    public Operand Tanh()
    {
        var len = _rows * _cols;
        var outFlat = new float[len];
        for (var i = 0; i < len; i++)
            outFlat[i] = MathF.Tanh(Data[i]);

        var result = new Operand(outFlat, _rows, _cols, (this, null));
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
    public Operand Softmax()
    {
        var m = _rows;
        var n = _cols;
        var outFlat = new float[m * n];
        for (var i = 0; i < m; i++)
        {
            var rowStart = i * n;
            var max = float.NegativeInfinity;
            for (var j = 0; j < n; j++)
                if (Data[rowStart + j] > max) max = Data[rowStart + j];
            var sum = 0.0f;
            for (var j = 0; j < n; j++)
            {
                outFlat[rowStart + j] = MathF.Exp(Data[rowStart + j] - max);
                sum += outFlat[rowStart + j];
            }
            for (var j = 0; j < n; j++)
                outFlat[rowStart + j] /= sum;
        }

        var result = new Operand(outFlat, m, n, (this, null));
        var inGrad = Gradient;
        var dOut = result.Gradient;

        result._backward = () =>
        {
            for (var i = 0; i < m; i++)
            {
                var rowStart = i * n;
                var dot = TensorPrimitives.Dot(
                    new ReadOnlySpan<float>(dOut, rowStart, n),
                    new ReadOnlySpan<float>(outFlat, rowStart, n));
                for (var j = 0; j < n; j++)
                    inGrad[rowStart + j] += outFlat[rowStart + j] * (dOut[rowStart + j] - dot);
            }
        };
        return result;
    }

    // Mean NLL over all rows: loss = -mean_i( log(Data[i, targets[i]]) )  ->  [1,1]
    public Operand NLL(int[] targets)
    {
        var m = _rows;
        var n = _cols;
        var total = 0.0f;
        for (var i = 0; i < m; i++)
            total += -MathF.Log(Data[i * n + targets[i]] + 1e-6f);
        var loss = total / m;

        var result = new Operand(new float[] { loss }, 1, 1, (this, null));
        var inGrad = Gradient;
        var dOut = result.Gradient;

        result._backward = () =>
        {
            for (var i = 0; i < m; i++)
                inGrad[i * n + targets[i]] += dOut[0] * (-1.0f / (m * (Data[i * n + targets[i]] + 1e-6f)));
        };
        return result;
    }

    // -log(Data[0, targetCol])  ->  [1,1]
    public Operand NLL(int targetCol)
    {
        var p = Data[targetCol];
        var loss = -MathF.Log(p + 1e-6f);
        var result = new Operand(new float[] { loss }, 1, 1, (this, null));
        var inGrad = Gradient;
        var dOut = result.Gradient;

        result._backward = () =>
        {
            inGrad[targetCol] += dOut[0] * (-1.0f / (p + 1e-6f));
        };
        return result;
    }

    // [m,n] -> [n,m]
    public Operand Transpose()
    {
        var m = _rows;
        var n = _cols;
        var outFlat = new float[m * n];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
                outFlat[j * m + i] = Data[i * n + j];

        var result = new Operand(outFlat, n, m, (this, null));
        var inGrad = Gradient;
        var dOut = result.Gradient;

        result._backward = () =>
        {
            for (var i = 0; i < m; i++)
                for (var j = 0; j < n; j++)
                    inGrad[i * n + j] += dOut[j * m + i];
        };
        return result;
    }

    // elementwise multiply by fixed scalar
    public Operand Scale(float s)
    {
        var len = _rows * _cols;
        var outFlat = new float[len];
        for (var i = 0; i < len; i++)
            outFlat[i] = s * Data[i];

        var result = new Operand(outFlat, _rows, _cols, (this, null));
        var inGrad = Gradient;
        var dOut = result.Gradient;

        result._backward = () =>
        {
            for (var i = 0; i < len; i++)
                inGrad[i] += s * dOut[i];
        };
        return result;
    }

    // [m,n] + [m,n] -> [m,n] (full-shape elementwise add, for residuals)
    public Operand Add(Operand other)
    {
        var len = _rows * _cols;
        var outFlat = new float[len];
        for (var i = 0; i < len; i++)
            outFlat[i] = Data[i] + other.Data[i];

        var result = new Operand(outFlat, _rows, _cols, (this, other));
        var aGrad = Gradient;
        var bGrad = other.Gradient;
        var dOut = result.Gradient;

        result._backward = () =>
        {
            for (var i = 0; i < len; i++)
            {
                aGrad[i] += dOut[i];
                bGrad[i] += dOut[i];
            }
        };
        return result;
    }

    // where mask[i,j] is true set output to fill, else copy; masked positions get zero gradient
    public Operand MaskFill(bool[,] mask, float fill)
    {
        var m = _rows;
        var n = _cols;
        var outFlat = new float[m * n];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
                outFlat[i * n + j] = mask[i, j] ? fill : Data[i * n + j];

        var result = new Operand(outFlat, m, n, (this, null));
        var inGrad = Gradient;
        var dOut = result.Gradient;

        result._backward = () =>
        {
            for (var i = 0; i < m; i++)
                for (var j = 0; j < n; j++)
                    if (!mask[i, j])
                        inGrad[i * n + j] += dOut[i * n + j];
        };
        return result;
    }

    // extract row -> [1, cols]
    public Operand SliceRow(int row)
    {
        var n = _cols;
        var outFlat = new float[n];
        Array.Copy(Data, row * n, outFlat, 0, n);

        var result = new Operand(outFlat, 1, n, (this, null));
        var inGrad = Gradient;
        var dOut = result.Gradient;
        var offset = row * n;

        result._backward = () =>
        {
            for (var j = 0; j < n; j++)
                inGrad[offset + j] += dOut[j];
        };
        return result;
    }

    // Sum all elements -> [1,1]
    public Operand Sum()
    {
        var total = 0.0f;
        var len = _rows * _cols;
        for (var i = 0; i < len; i++) total += Data[i];

        var result = new Operand(new float[] { total }, 1, 1, (this, null));
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
        Gradient[0] = 1.0f;
        var ordered = BuildTopological();
        for (var i = ordered.Count - 1; i >= 0; i--)
            ordered[i]._backward();
    }

    public void ZeroGradient()
        => Array.Clear(Gradient, 0, Gradient.Length);

    public float GradientNormSquared()
        => TensorPrimitives.Dot(Gradient, Gradient);

    public void ApplyGradients(float lr, float clipScale)
    {
        var scale = lr * clipScale;
        for (var i = 0; i < Data.Length; i++)
            Data[i] -= scale * Gradient[i];
    }

    private List<Operand> BuildTopological()
    {
        var visited = new HashSet<Operand>();
        var ordered = new List<Operand>();
        var stack = new Stack<Operand>();
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

    private static bool TryPushUnvisited(Stack<Operand> stack, HashSet<Operand> visited,
        Operand node, ref bool pushed)
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

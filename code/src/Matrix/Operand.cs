#nullable enable
using System;
using System.Collections.Generic;
using System.Numerics.Tensors;

namespace TinyBrain;

public class Operand
{
    private static volatile IMatrixBackend _backend = new CpuMatrixBackend();

    public static void SetBackend(IMatrixBackend backend)
        => _backend = backend ?? throw new ArgumentNullException(nameof(backend));


    public float[] Data { get; }
    public float[] Gradient { get; }

    private readonly int _rows;
    private readonly int _cols;

    public int Rows => _rows;
    public int Cols => _cols;

    private readonly (Operand Left, Operand Right) _previous;
    private readonly Operand[]? _previousN;
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

    private Operand(float[] data, int rows, int cols, Operand[] previousN)
        : this(data, rows, cols)
    {
        _previousN = previousN;
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

    // Static helpers for external code that does CPU-side reads/writes on float[] arrays
    public static void SynchronizeDeviceArray(float[] data) => _backend.Synchronize(data);
    public static void InvalidateDeviceArray(float[] data) => _backend.InvalidateDevice(data);

    // Allows leaf nodes to register a custom backward (e.g. embedding bridge)
    // Wraps with sync/invalidate so CPU-side callbacks see current GPU data and
    // any writes they make are correctly marked for re-upload.
    public void SetBackward(Action<float[]> backward)
        => _backward = () =>
        {
            _backend.Synchronize(Gradient);
            backward(Gradient);
            _backend.InvalidateDevice(Gradient);
        };

    // [m,k] x [k,n] -> [m,n]
    public Operand MatMul(Operand w)
    {
        var m = _rows;
        var k = _cols;
        var n = w._cols;
        var outFlat = new float[m * n];
        _backend.MatMul(Data, w.Data, outFlat, m, k, n);

        var result = new Operand(outFlat, m, n, (this, w));
        var aData = Data;
        var wData = w.Data;
        var aGrad = Gradient;
        var wGrad = w.Gradient;
        var dOut = result.Gradient;

        result._backward = () =>
        {
            _backend.MatMulBackwardLeft(dOut, wData, aGrad, m, k, n);
            _backend.MatMulBackwardRight(aData, dOut, wGrad, m, k, n);
        };
        return result;
    }

    // [m,n] + [1,n] broadcast -> [m,n]
    public Operand AddBias(Operand b)
    {
        var m = _rows;
        var n = _cols;
        var outFlat = new float[m * n];
        _backend.AddBias(Data, b.Data, outFlat, m, n);

        var result = new Operand(outFlat, m, n, (this, b));
        var aGrad = Gradient;
        var bGrad = b.Gradient;
        var dOut = result.Gradient;

        result._backward = () => _backend.AddBiasBackward(dOut, aGrad, bGrad, m, n);
        return result;
    }

    // elementwise tanh
    public Operand Tanh()
    {
        var len = _rows * _cols;
        var outFlat = new float[len];
        _backend.Tanh(Data, outFlat, len);

        var result = new Operand(outFlat, _rows, _cols, (this, null));
        var inGrad = Gradient;
        var dOut = result.Gradient;

        result._backward = () => _backend.TanhBackward(outFlat, dOut, inGrad, len);
        return result;
    }

    // per-row stable softmax
    public Operand Softmax()
    {
        var m = _rows;
        var n = _cols;
        var outFlat = new float[m * n];
        _backend.Softmax(Data, outFlat, m, n);

        var result = new Operand(outFlat, m, n, (this, null));
        var inGrad = Gradient;
        var dOut = result.Gradient;

        result._backward = () => _backend.SoftmaxBackward(outFlat, dOut, inGrad, m, n);
        return result;
    }

    // Mean NLL over all rows: loss = -mean_i( log(Data[i, targets[i]]) )  ->  [1,1]
    public Operand NLL(int[] targets)
    {
        var m = _rows;
        var n = _cols;
        _backend.Synchronize(Data);
        var total = 0.0f;
        for (var i = 0; i < m; i++)
            total += -MathF.Log(Data[i * n + targets[i]] + 1e-6f);
        var loss = total / m;

        var result = new Operand(new float[] { loss }, 1, 1, (this, null));
        var inGrad = Gradient;
        var dOut = result.Gradient;

        result._backward = () =>
        {
            _backend.Synchronize(dOut);
            for (var i = 0; i < m; i++)
                inGrad[i * n + targets[i]] += dOut[0] * (-1.0f / (m * (Data[i * n + targets[i]] + 1e-6f)));
            _backend.InvalidateDevice(inGrad);
        };
        return result;
    }

    // -log(Data[0, targetCol])  ->  [1,1]
    public Operand NLL(int targetCol)
    {
        _backend.Synchronize(Data);
        var p = Data[targetCol];
        var loss = -MathF.Log(p + 1e-6f);
        var result = new Operand(new float[] { loss }, 1, 1, (this, null));
        var inGrad = Gradient;
        var dOut = result.Gradient;

        result._backward = () =>
        {
            _backend.Synchronize(dOut);
            inGrad[targetCol] += dOut[0] * (-1.0f / (p + 1e-6f));
            _backend.InvalidateDevice(inGrad);
        };
        return result;
    }

    // [m,n] -> [n,m]
    public Operand Transpose()
    {
        var m = _rows;
        var n = _cols;
        var outFlat = new float[m * n];
        _backend.Transpose(Data, outFlat, m, n);

        var result = new Operand(outFlat, n, m, (this, null));
        var inGrad = Gradient;
        var dOut = result.Gradient;

        result._backward = () => _backend.TransposeBackward(dOut, inGrad, m, n);
        return result;
    }

    // elementwise multiply by fixed scalar
    public Operand Scale(float s)
    {
        var len = _rows * _cols;
        var outFlat = new float[len];
        _backend.Scale(Data, s, outFlat, len);

        var result = new Operand(outFlat, _rows, _cols, (this, null));
        var inGrad = Gradient;
        var dOut = result.Gradient;

        result._backward = () => _backend.ScaleBackward(s, dOut, inGrad, len);
        return result;
    }

    // [m,n] + [m,n] -> [m,n] (full-shape elementwise add, for residuals)
    public Operand Add(Operand other)
    {
        var len = _rows * _cols;
        var outFlat = new float[len];
        _backend.Add(Data, other.Data, outFlat, len);

        var result = new Operand(outFlat, _rows, _cols, (this, other));
        var aGrad = Gradient;
        var bGrad = other.Gradient;
        var dOut = result.Gradient;

        result._backward = () => _backend.AddBackward(dOut, aGrad, bGrad, len);
        return result;
    }

    // where mask[i,j] is true set output to fill, else copy; masked positions get zero gradient
    public Operand MaskFill(bool[,] mask, float fill)
    {
        var m = _rows;
        var n = _cols;
        _backend.Synchronize(Data);
        var outFlat = new float[m * n];
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
                outFlat[i * n + j] = mask[i, j] ? fill : Data[i * n + j];

        var result = new Operand(outFlat, m, n, (this, null));
        var inGrad = Gradient;
        var dOut = result.Gradient;

        result._backward = () =>
        {
            _backend.Synchronize(dOut);
            for (var i = 0; i < m; i++)
                for (var j = 0; j < n; j++)
                    if (!mask[i, j])
                        inGrad[i * n + j] += dOut[i * n + j];
            _backend.InvalidateDevice(inGrad);
        };
        return result;
    }

    // extract row -> [1, cols]
    public Operand SliceRow(int row)
    {
        var n = _cols;
        _backend.Synchronize(Data);
        var outFlat = new float[n];
        Array.Copy(Data, row * n, outFlat, 0, n);

        var result = new Operand(outFlat, 1, n, (this, null));
        var inGrad = Gradient;
        var dOut = result.Gradient;
        var offset = row * n;

        result._backward = () =>
        {
            _backend.Synchronize(dOut);
            for (var j = 0; j < n; j++)
                inGrad[offset + j] += dOut[j];
            _backend.InvalidateDevice(inGrad);
        };
        return result;
    }

    // Sum all elements -> [1,1]
    public Operand Sum()
    {
        var total = 0.0f;
        var len = _rows * _cols;
        _backend.Synchronize(Data);
        for (var i = 0; i < len; i++) total += Data[i];

        var result = new Operand(new float[] { total }, 1, 1, (this, null));
        var inGrad = Gradient;
        var dOut = result.Gradient;

        result._backward = () =>
        {
            _backend.Synchronize(dOut);
            for (var i = 0; i < len; i++)
                inGrad[i] += dOut[0];
            _backend.InvalidateDevice(inGrad);
        };
        return result;
    }

    public void Backpropagation()
    {
        Gradient[0] = 1.0f;
        _backend.InvalidateDevice(Gradient);
        var ordered = BuildTopological();
        for (var i = ordered.Count - 1; i >= 0; i--)
            ordered[i]._backward();
    }

    public void ZeroGradient()
    {
        Array.Clear(Gradient, 0, Gradient.Length);
        _backend.InvalidateDevice(Gradient);
    }

    public float GradientNormSquared()
    {
        _backend.Synchronize(Gradient);
        return TensorPrimitives.Dot(Gradient, Gradient);
    }

    public void ApplyGradients(float lr, float clipScale)
    {
        _backend.Synchronize(Data);
        _backend.Synchronize(Gradient);
        var scale = lr * clipScale;
        for (var i = 0; i < Data.Length; i++)
            Data[i] -= scale * Gradient[i];
        _backend.InvalidateDevice(Data);
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

            if (!pushed && current._previousN != null)
                foreach (var p in current._previousN)
                    if (TryPushUnvisited(stack, visited, p, ref pushed)) break;

            if (!pushed)
            {
                stack.Pop();
                if (visited.Add(current))
                    ordered.Add(current);
            }
        }
        return ordered;
    }

    // Stack B [1, n] operands -> [B, n]
    public static Operand Stack(Operand[] rows)
    {
        var b = rows.Length;
        var n = rows[0].Cols;
        var outFlat = new float[b * n];
        for (var i = 0; i < b; i++)
            Array.Copy(rows[i].Data, 0, outFlat, i * n, n);

        var result = new Operand(outFlat, b, n, rows);
        var rowGrads = Array.ConvertAll(rows, r => r.Gradient);
        var dOut = result.Gradient;

        result._backward = () =>
        {
            _backend.Synchronize(dOut);
            for (var i = 0; i < b; i++)
            {
                for (var j = 0; j < n; j++)
                    rowGrads[i][j] += dOut[i * n + j];
                _backend.InvalidateDevice(rowGrads[i]);
            }
        };
        return result;
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

using System;
using System.Numerics.Tensors;

namespace TinyBrain;

public sealed class CpuMatrixBackend : IMatrixBackend
{
    public void MatMul(float[] a, float[] w, float[] output, int m, int k, int n)
    {
        Array.Clear(output, 0, m * n);
        for (var i = 0; i < m; i++)
            for (var p = 0; p < k; p++)
                TensorPrimitives.MultiplyAdd(
                    new ReadOnlySpan<float>(w, p * n, n),
                    a[i * k + p],
                    new ReadOnlySpan<float>(output, i * n, n),
                    new Span<float>(output, i * n, n));
    }

    public void AddBias(float[] a, float[] bias, float[] output, int m, int n)
    {
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
                output[i * n + j] = a[i * n + j] + bias[j];
    }

    public void Tanh(float[] input, float[] output, int len)
    {
        for (var i = 0; i < len; i++)
            output[i] = MathF.Tanh(input[i]);
    }

    public void Softmax(float[] input, float[] output, int m, int n)
    {
        for (var i = 0; i < m; i++)
        {
            var rowStart = i * n;
            var max = float.NegativeInfinity;
            for (var j = 0; j < n; j++)
                if (input[rowStart + j] > max) max = input[rowStart + j];
            var sum = 0.0f;
            for (var j = 0; j < n; j++)
            {
                output[rowStart + j] = MathF.Exp(input[rowStart + j] - max);
                sum += output[rowStart + j];
            }
            for (var j = 0; j < n; j++)
                output[rowStart + j] /= sum;
        }
    }

    public void Scale(float[] input, float s, float[] output, int len)
    {
        for (var i = 0; i < len; i++)
            output[i] = s * input[i];
    }

    public void Add(float[] a, float[] b, float[] output, int len)
    {
        for (var i = 0; i < len; i++)
            output[i] = a[i] + b[i];
    }

    public void Transpose(float[] input, float[] output, int m, int n)
    {
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
                output[j * m + i] = input[i * n + j];
    }

    public void MatMulBackwardLeft(float[] dOut, float[] w, float[] dA, int m, int k, int n)
    {
        for (var i = 0; i < m; i++)
            for (var p = 0; p < k; p++)
                dA[i * k + p] += TensorPrimitives.Dot(
                    new ReadOnlySpan<float>(dOut, i * n, n),
                    new ReadOnlySpan<float>(w, p * n, n));
    }

    public void MatMulBackwardRight(float[] a, float[] dOut, float[] dW, int m, int k, int n)
    {
        for (var i = 0; i < m; i++)
            for (var p = 0; p < k; p++)
                TensorPrimitives.MultiplyAdd(
                    new ReadOnlySpan<float>(dOut, i * n, n),
                    a[i * k + p],
                    new ReadOnlySpan<float>(dW, p * n, n),
                    new Span<float>(dW, p * n, n));
    }

    public void AddBiasBackward(float[] dOut, float[] dA, float[] dBias, int m, int n)
    {
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
            {
                var val = dOut[i * n + j];
                dA[i * n + j] += val;
                dBias[j] += val;
            }
    }

    public void TanhBackward(float[] tanhOut, float[] dOut, float[] dIn, int len)
    {
        for (var i = 0; i < len; i++)
            dIn[i] += (1.0f - tanhOut[i] * tanhOut[i]) * dOut[i];
    }

    public void SoftmaxBackward(float[] softOut, float[] dOut, float[] dIn, int m, int n)
    {
        for (var i = 0; i < m; i++)
        {
            var rowStart = i * n;
            var dot = TensorPrimitives.Dot(
                new ReadOnlySpan<float>(dOut, rowStart, n),
                new ReadOnlySpan<float>(softOut, rowStart, n));
            for (var j = 0; j < n; j++)
                dIn[rowStart + j] += softOut[rowStart + j] * (dOut[rowStart + j] - dot);
        }
    }

    public void ScaleBackward(float s, float[] dOut, float[] dIn, int len)
    {
        for (var i = 0; i < len; i++)
            dIn[i] += s * dOut[i];
    }

    public void AddBackward(float[] dOut, float[] dA, float[] dB, int len)
    {
        for (var i = 0; i < len; i++)
        {
            dA[i] += dOut[i];
            dB[i] += dOut[i];
        }
    }

    public void TransposeBackward(float[] dOut, float[] dIn, int m, int n)
    {
        // dOut is [n,m] (the transposed shape); dIn is [m,n]
        for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
                dIn[i * n + j] += dOut[j * m + i];
    }

    public void Dispose() { }
}

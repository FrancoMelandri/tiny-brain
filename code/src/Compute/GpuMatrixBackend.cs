#nullable enable
using System;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace TinyBrain;

public sealed class GpuMatrixBackend : IMatrixBackend
{
    private readonly Context _context;
    private readonly CudaAccelerator _accelerator;

    private readonly Action<Index2D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, int, int> _matMulKernel;

    private readonly Action<Index2D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, int, int> _matMulBackwardLeftKernel;

    private readonly Action<Index2D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, int, int> _matMulBackwardRightKernel;

    private readonly Action<Index2D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, int> _addBiasKernel;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, int> _addBiasBackwardBiasKernel;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int> _addBiasBackwardInputKernel;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int> _tanhKernel;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int> _tanhBackwardKernel;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, int> _softmaxKernel;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, int> _softmaxBackwardKernel;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        float,
        ArrayView1D<float, Stride1D.Dense>,
        int> _scaleKernel;

    private readonly Action<Index1D,
        float,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int> _scaleBackwardKernel;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int> _addKernel;

    private readonly Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int> _addBackwardKernel;

    private readonly Action<Index2D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, int> _transposeKernel;

    private readonly Action<Index2D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, int> _transposeBackwardKernel;

    private GpuMatrixBackend(Context context, CudaAccelerator accelerator)
    {
        _context = context;
        _accelerator = accelerator;

        _matMulKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index2D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int, int>(MatMulKernel);

        _matMulBackwardLeftKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index2D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int, int>(MatMulBackwardLeftKernel);

        _matMulBackwardRightKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index2D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int, int>(MatMulBackwardRightKernel);

        _addBiasKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index2D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int>(AddBiasKernel);

        _addBiasBackwardBiasKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int>(AddBiasBackwardBiasKernel);

        _addBiasBackwardInputKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int>(AddBiasBackwardInputKernel);

        _tanhKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int>(TanhKernel);

        _tanhBackwardKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int>(TanhBackwardKernel);

        _softmaxKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int>(SoftmaxKernel);

        _softmaxBackwardKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int>(SoftmaxBackwardKernel);

        _scaleKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            float,
            ArrayView1D<float, Stride1D.Dense>,
            int>(ScaleKernel);

        _scaleBackwardKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            float,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int>(ScaleBackwardKernel);

        _addKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int>(AddKernel);

        _addBackwardKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int>(AddBackwardKernel);

        _transposeKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index2D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int>(TransposeKernel);

        _transposeBackwardKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index2D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            int, int>(TransposeBackwardKernel);
    }

    public static GpuMatrixBackend? TryCreate(bool verbose = false)
    {
        try
        {
            var context = Context.Create(b => b.Default().EnableAlgorithms());
            var acc = context.CreateCudaAccelerator(0);
            return new GpuMatrixBackend(context, acc);
        }
        catch (Exception ex)
        {
            if (verbose)
            {
                Console.Error.WriteLine($"[GpuMatrixBackend] Init failed: {ex.GetType().Name}: {ex.Message}");
                var inner = ex.InnerException;
                while (inner != null)
                {
                    Console.Error.WriteLine($"  Caused by: {inner.GetType().Name}: {inner.Message}");
                    inner = inner.InnerException;
                }
            }
            return null;
        }
    }

    public void Dispose()
    {
        _accelerator.Dispose();
        _context.Dispose();
    }

    // -------------------------------------------------------------------------
    // Forward passes
    // -------------------------------------------------------------------------

    public void MatMul(float[] a, float[] w, float[] output, int m, int k, int n)
    {
        using var devA = _accelerator.Allocate1D<float>(a.Length);
        using var devW = _accelerator.Allocate1D<float>(w.Length);
        using var devOut = _accelerator.Allocate1D<float>(output.Length);
        devA.CopyFromCPU(a);
        devW.CopyFromCPU(w);
        _matMulKernel(new Index2D(m, n), devA.View, devW.View, devOut.View, m, k, n);
        _accelerator.Synchronize();
        devOut.CopyToCPU(output);
    }

    public void AddBias(float[] a, float[] bias, float[] output, int m, int n)
    {
        using var devA = _accelerator.Allocate1D<float>(a.Length);
        using var devBias = _accelerator.Allocate1D<float>(bias.Length);
        using var devOut = _accelerator.Allocate1D<float>(output.Length);
        devA.CopyFromCPU(a);
        devBias.CopyFromCPU(bias);
        _addBiasKernel(new Index2D(m, n), devA.View, devBias.View, devOut.View, m, n);
        _accelerator.Synchronize();
        devOut.CopyToCPU(output);
    }

    public void Tanh(float[] input, float[] output, int len)
    {
        using var devIn = _accelerator.Allocate1D<float>(len);
        using var devOut = _accelerator.Allocate1D<float>(len);
        devIn.CopyFromCPU(input);
        _tanhKernel(new Index1D(len), devIn.View, devOut.View, len);
        _accelerator.Synchronize();
        devOut.CopyToCPU(output);
    }

    public void Softmax(float[] input, float[] output, int m, int n)
    {
        using var devIn = _accelerator.Allocate1D<float>(m * n);
        using var devOut = _accelerator.Allocate1D<float>(m * n);
        devIn.CopyFromCPU(input);
        _softmaxKernel(new Index1D(m), devIn.View, devOut.View, m, n);
        _accelerator.Synchronize();
        devOut.CopyToCPU(output);
    }

    public void Scale(float[] input, float s, float[] output, int len)
    {
        using var devIn = _accelerator.Allocate1D<float>(len);
        using var devOut = _accelerator.Allocate1D<float>(len);
        devIn.CopyFromCPU(input);
        _scaleKernel(new Index1D(len), devIn.View, s, devOut.View, len);
        _accelerator.Synchronize();
        devOut.CopyToCPU(output);
    }

    public void Add(float[] a, float[] b, float[] output, int len)
    {
        using var devA = _accelerator.Allocate1D<float>(len);
        using var devB = _accelerator.Allocate1D<float>(len);
        using var devOut = _accelerator.Allocate1D<float>(len);
        devA.CopyFromCPU(a);
        devB.CopyFromCPU(b);
        _addKernel(new Index1D(len), devA.View, devB.View, devOut.View, len);
        _accelerator.Synchronize();
        devOut.CopyToCPU(output);
    }

    public void Transpose(float[] input, float[] output, int m, int n)
    {
        using var devIn = _accelerator.Allocate1D<float>(m * n);
        using var devOut = _accelerator.Allocate1D<float>(m * n);
        devIn.CopyFromCPU(input);
        _transposeKernel(new Index2D(m, n), devIn.View, devOut.View, m, n);
        _accelerator.Synchronize();
        devOut.CopyToCPU(output);
    }

    // -------------------------------------------------------------------------
    // Backward passes — upload existing gradient state, accumulate, download back
    // -------------------------------------------------------------------------

    public void MatMulBackwardLeft(float[] dOut, float[] w, float[] dA, int m, int k, int n)
    {
        using var devDOut = _accelerator.Allocate1D<float>(dOut.Length);
        using var devW = _accelerator.Allocate1D<float>(w.Length);
        using var devDA = _accelerator.Allocate1D<float>(dA.Length);
        devDOut.CopyFromCPU(dOut);
        devW.CopyFromCPU(w);
        devDA.CopyFromCPU(dA);
        _matMulBackwardLeftKernel(new Index2D(m, k), devDOut.View, devW.View, devDA.View, m, k, n);
        _accelerator.Synchronize();
        devDA.CopyToCPU(dA);
    }

    public void MatMulBackwardRight(float[] a, float[] dOut, float[] dW, int m, int k, int n)
    {
        using var devA = _accelerator.Allocate1D<float>(a.Length);
        using var devDOut = _accelerator.Allocate1D<float>(dOut.Length);
        using var devDW = _accelerator.Allocate1D<float>(dW.Length);
        devA.CopyFromCPU(a);
        devDOut.CopyFromCPU(dOut);
        devDW.CopyFromCPU(dW);
        _matMulBackwardRightKernel(new Index2D(k, n), devA.View, devDOut.View, devDW.View, m, k, n);
        _accelerator.Synchronize();
        devDW.CopyToCPU(dW);
    }

    public void AddBiasBackward(float[] dOut, float[] dA, float[] dBias, int m, int n)
    {
        using var devDOut = _accelerator.Allocate1D<float>(dOut.Length);
        using var devDA = _accelerator.Allocate1D<float>(dA.Length);
        using var devDBias = _accelerator.Allocate1D<float>(dBias.Length);
        devDOut.CopyFromCPU(dOut);
        devDA.CopyFromCPU(dA);
        devDBias.CopyFromCPU(dBias);
        _addBiasBackwardInputKernel(new Index1D(m * n), devDOut.View, devDA.View, m * n);
        _addBiasBackwardBiasKernel(new Index1D(n), devDOut.View, devDBias.View, m, n);
        _accelerator.Synchronize();
        devDA.CopyToCPU(dA);
        devDBias.CopyToCPU(dBias);
    }

    public void TanhBackward(float[] tanhOut, float[] dOut, float[] dIn, int len)
    {
        using var devTanh = _accelerator.Allocate1D<float>(len);
        using var devDOut = _accelerator.Allocate1D<float>(len);
        using var devDIn = _accelerator.Allocate1D<float>(len);
        devTanh.CopyFromCPU(tanhOut);
        devDOut.CopyFromCPU(dOut);
        devDIn.CopyFromCPU(dIn);
        _tanhBackwardKernel(new Index1D(len), devTanh.View, devDOut.View, devDIn.View, len);
        _accelerator.Synchronize();
        devDIn.CopyToCPU(dIn);
    }

    public void SoftmaxBackward(float[] softOut, float[] dOut, float[] dIn, int m, int n)
    {
        using var devSoftOut = _accelerator.Allocate1D<float>(m * n);
        using var devDOut = _accelerator.Allocate1D<float>(m * n);
        using var devDIn = _accelerator.Allocate1D<float>(m * n);
        devSoftOut.CopyFromCPU(softOut);
        devDOut.CopyFromCPU(dOut);
        devDIn.CopyFromCPU(dIn);
        _softmaxBackwardKernel(new Index1D(m), devSoftOut.View, devDOut.View, devDIn.View, m, n);
        _accelerator.Synchronize();
        devDIn.CopyToCPU(dIn);
    }

    public void ScaleBackward(float s, float[] dOut, float[] dIn, int len)
    {
        using var devDOut = _accelerator.Allocate1D<float>(len);
        using var devDIn = _accelerator.Allocate1D<float>(len);
        devDOut.CopyFromCPU(dOut);
        devDIn.CopyFromCPU(dIn);
        _scaleBackwardKernel(new Index1D(len), s, devDOut.View, devDIn.View, len);
        _accelerator.Synchronize();
        devDIn.CopyToCPU(dIn);
    }

    public void AddBackward(float[] dOut, float[] dA, float[] dB, int len)
    {
        using var devDOut = _accelerator.Allocate1D<float>(len);
        using var devDA = _accelerator.Allocate1D<float>(len);
        using var devDB = _accelerator.Allocate1D<float>(len);
        devDOut.CopyFromCPU(dOut);
        devDA.CopyFromCPU(dA);
        devDB.CopyFromCPU(dB);
        _addBackwardKernel(new Index1D(len), devDOut.View, devDA.View, devDB.View, len);
        _accelerator.Synchronize();
        devDA.CopyToCPU(dA);
        devDB.CopyToCPU(dB);
    }

    public void TransposeBackward(float[] dOut, float[] dIn, int m, int n)
    {
        using var devDOut = _accelerator.Allocate1D<float>(m * n);
        using var devDIn = _accelerator.Allocate1D<float>(m * n);
        devDOut.CopyFromCPU(dOut);
        devDIn.CopyFromCPU(dIn);
        _transposeBackwardKernel(new Index2D(m, n), devDOut.View, devDIn.View, m, n);
        _accelerator.Synchronize();
        devDIn.CopyToCPU(dIn);
    }

    // -------------------------------------------------------------------------
    // Static GPU kernels — must be static for ILGPU
    // Use XMath.* for GPU-safe math functions (MathF.* is CPU-only)
    // -------------------------------------------------------------------------

    static void MatMulKernel(
        Index2D idx,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> output,
        int m, int k, int n)
    {
        var row = idx.X;
        var col = idx.Y;
        if (row >= m || col >= n) return;
        var sum = 0.0f;
        for (var p = 0; p < k; p++)
            sum += a[row * k + p] * w[p * n + col];
        output[row * n + col] = sum;
    }

    static void MatMulBackwardLeftKernel(
        Index2D idx,
        ArrayView1D<float, Stride1D.Dense> dOut,
        ArrayView1D<float, Stride1D.Dense> w,
        ArrayView1D<float, Stride1D.Dense> dA,
        int m, int k, int n)
    {
        var i = idx.X;
        var p = idx.Y;
        if (i >= m || p >= k) return;
        var sum = 0.0f;
        for (var j = 0; j < n; j++)
            sum += dOut[i * n + j] * w[p * n + j];  // W^T: w[p,j] = w[p*n+j]
        dA[i * k + p] += sum;
    }

    static void MatMulBackwardRightKernel(
        Index2D idx,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> dOut,
        ArrayView1D<float, Stride1D.Dense> dW,
        int m, int k, int n)
    {
        var p = idx.X;
        var j = idx.Y;
        if (p >= k || j >= n) return;
        var sum = 0.0f;
        for (var i = 0; i < m; i++)
            sum += a[i * k + p] * dOut[i * n + j];  // A^T: a[i,p] = a[i*k+p]
        dW[p * n + j] += sum;
    }

    static void AddBiasKernel(
        Index2D idx,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> bias,
        ArrayView1D<float, Stride1D.Dense> output,
        int m, int n)
    {
        var i = idx.X;
        var j = idx.Y;
        if (i >= m || j >= n) return;
        output[i * n + j] = a[i * n + j] + bias[j];
    }

    static void AddBiasBackwardInputKernel(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> dOut,
        ArrayView1D<float, Stride1D.Dense> dA,
        int len)
    {
        if (idx >= len) return;
        dA[idx] += dOut[idx];
    }

    static void AddBiasBackwardBiasKernel(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> dOut,
        ArrayView1D<float, Stride1D.Dense> dBias,
        int m, int n)
    {
        var j = (int)idx;
        if (j >= n) return;
        var sum = 0.0f;
        for (var i = 0; i < m; i++)
            sum += dOut[i * n + j];
        dBias[j] += sum;
    }

    static void TanhKernel(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int len)
    {
        if (idx >= len) return;
        output[idx] = XMath.Tanh(input[idx]);
    }

    static void TanhBackwardKernel(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> tanhOut,
        ArrayView1D<float, Stride1D.Dense> dOut,
        ArrayView1D<float, Stride1D.Dense> dIn,
        int len)
    {
        if (idx >= len) return;
        var t = tanhOut[idx];
        dIn[idx] += (1.0f - t * t) * dOut[idx];
    }

    static void SoftmaxKernel(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int m, int n)
    {
        var i = (int)idx;
        if (i >= m) return;
        var rowStart = i * n;
        var max = float.NegativeInfinity;
        for (var j = 0; j < n; j++)
            if (input[rowStart + j] > max) max = input[rowStart + j];
        var sum = 0.0f;
        for (var j = 0; j < n; j++)
        {
            output[rowStart + j] = XMath.Exp(input[rowStart + j] - max);
            sum += output[rowStart + j];
        }
        for (var j = 0; j < n; j++)
            output[rowStart + j] /= sum;
    }

    static void SoftmaxBackwardKernel(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> softOut,
        ArrayView1D<float, Stride1D.Dense> dOut,
        ArrayView1D<float, Stride1D.Dense> dIn,
        int m, int n)
    {
        var i = (int)idx;
        if (i >= m) return;
        var rowStart = i * n;
        var dot = 0.0f;
        for (var j = 0; j < n; j++)
            dot += dOut[rowStart + j] * softOut[rowStart + j];
        for (var j = 0; j < n; j++)
            dIn[rowStart + j] += softOut[rowStart + j] * (dOut[rowStart + j] - dot);
    }

    static void ScaleKernel(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        float s,
        ArrayView1D<float, Stride1D.Dense> output,
        int len)
    {
        if (idx >= len) return;
        output[idx] = s * input[idx];
    }

    static void ScaleBackwardKernel(
        Index1D idx,
        float s,
        ArrayView1D<float, Stride1D.Dense> dOut,
        ArrayView1D<float, Stride1D.Dense> dIn,
        int len)
    {
        if (idx >= len) return;
        dIn[idx] += s * dOut[idx];
    }

    static void AddKernel(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b,
        ArrayView1D<float, Stride1D.Dense> output,
        int len)
    {
        if (idx >= len) return;
        output[idx] = a[idx] + b[idx];
    }

    static void AddBackwardKernel(
        Index1D idx,
        ArrayView1D<float, Stride1D.Dense> dOut,
        ArrayView1D<float, Stride1D.Dense> dA,
        ArrayView1D<float, Stride1D.Dense> dB,
        int len)
    {
        if (idx >= len) return;
        dA[idx] += dOut[idx];
        dB[idx] += dOut[idx];
    }

    static void TransposeKernel(
        Index2D idx,
        ArrayView1D<float, Stride1D.Dense> input,
        ArrayView1D<float, Stride1D.Dense> output,
        int m, int n)
    {
        var i = idx.X;
        var j = idx.Y;
        if (i >= m || j >= n) return;
        output[j * m + i] = input[i * n + j];
    }

    // dOut is [n,m] (transposed shape); writes back to dIn which is [m,n]
    static void TransposeBackwardKernel(
        Index2D idx,
        ArrayView1D<float, Stride1D.Dense> dOut,
        ArrayView1D<float, Stride1D.Dense> dIn,
        int m, int n)
    {
        var i = idx.X;
        var j = idx.Y;
        if (i >= m || j >= n) return;
        dIn[i * n + j] += dOut[j * m + i];
    }
}

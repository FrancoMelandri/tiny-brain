#nullable enable
using System;
using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace TinyBrain;

public sealed class GpuMatrixBackend : IMatrixBackend
{
    // -------------------------------------------------------------------------
    // Device buffer tracking — keeps float[] arrays mirrored on GPU
    // -------------------------------------------------------------------------

    private sealed class DeviceBuffer
    {
        public MemoryBuffer1D<float, Stride1D.Dense>? Memory;
        public bool HostNewer = true; // true = CPU has data not yet on GPU
    }

    private readonly ConditionalWeakTable<float[], DeviceBuffer> _cache = new();

    // Upload if stale, return GPU view — used for kernel inputs and grad accumulation
    private ArrayView1D<float, Stride1D.Dense> GetReadView(float[] host)
    {
        var buf = _cache.GetOrCreateValue(host);
        if (buf.Memory == null)
            buf.Memory = _accelerator.Allocate1D<float>(host.Length);
        if (buf.HostNewer)
        {
            buf.Memory.CopyFromCPU(host);
            buf.HostNewer = false;
        }
        return buf.Memory.View;
    }

    // Allocate if needed, do NOT upload — used for kernel outputs (kernel overwrites everything)
    private ArrayView1D<float, Stride1D.Dense> GetWriteView(float[] host)
    {
        var buf = _cache.GetOrCreateValue(host);
        if (buf.Memory == null)
            buf.Memory = _accelerator.Allocate1D<float>(host.Length);
        buf.HostNewer = false; // GPU will own after kernel runs
        return buf.Memory.View;
    }

    public void Synchronize(float[] host)
    {
        if (_cache.TryGetValue(host, out var buf) && buf?.Memory != null && !buf.HostNewer)
        {
            _accelerator.Synchronize();
            buf.Memory.CopyToCPU(host);
            // HostNewer stays false — GPU buffer is still valid
        }
    }

    public void InvalidateDevice(float[] host)
    {
        if (_cache.TryGetValue(host, out var buf) && buf != null)
            buf.HostNewer = true;
    }

    // -------------------------------------------------------------------------
    // ILGPU infrastructure
    // -------------------------------------------------------------------------

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
    // Forward passes — data stays on GPU, no download
    // -------------------------------------------------------------------------

    public void MatMul(float[] a, float[] w, float[] output, int m, int k, int n)
        => _matMulKernel(new Index2D(m, n), GetReadView(a), GetReadView(w), GetWriteView(output), m, k, n);

    public void AddBias(float[] a, float[] bias, float[] output, int m, int n)
        => _addBiasKernel(new Index2D(m, n), GetReadView(a), GetReadView(bias), GetWriteView(output), m, n);

    public void Tanh(float[] input, float[] output, int len)
        => _tanhKernel(new Index1D(len), GetReadView(input), GetWriteView(output), len);

    public void Softmax(float[] input, float[] output, int m, int n)
        => _softmaxKernel(new Index1D(m), GetReadView(input), GetWriteView(output), m, n);

    public void Scale(float[] input, float s, float[] output, int len)
        => _scaleKernel(new Index1D(len), GetReadView(input), s, GetWriteView(output), len);

    public void Add(float[] a, float[] b, float[] output, int len)
        => _addKernel(new Index1D(len), GetReadView(a), GetReadView(b), GetWriteView(output), len);

    public void Transpose(float[] input, float[] output, int m, int n)
        => _transposeKernel(new Index2D(m, n), GetReadView(input), GetWriteView(output), m, n);

    // -------------------------------------------------------------------------
    // Backward passes — accumulate on GPU, no download
    // GetReadView is used for gradient arrays too: uploads current value (e.g. zeros
    // after ZeroGradient), then kernel accumulates; GPU becomes authoritative.
    // -------------------------------------------------------------------------

    public void MatMulBackwardLeft(float[] dOut, float[] w, float[] dA, int m, int k, int n)
        => _matMulBackwardLeftKernel(new Index2D(m, k), GetReadView(dOut), GetReadView(w), GetReadView(dA), m, k, n);

    public void MatMulBackwardRight(float[] a, float[] dOut, float[] dW, int m, int k, int n)
        => _matMulBackwardRightKernel(new Index2D(k, n), GetReadView(a), GetReadView(dOut), GetReadView(dW), m, k, n);

    public void AddBiasBackward(float[] dOut, float[] dA, float[] dBias, int m, int n)
    {
        var devDOut = GetReadView(dOut);
        _addBiasBackwardInputKernel(new Index1D(m * n), devDOut, GetReadView(dA), m * n);
        _addBiasBackwardBiasKernel(new Index1D(n), devDOut, GetReadView(dBias), m, n);
    }

    public void TanhBackward(float[] tanhOut, float[] dOut, float[] dIn, int len)
        => _tanhBackwardKernel(new Index1D(len), GetReadView(tanhOut), GetReadView(dOut), GetReadView(dIn), len);

    public void SoftmaxBackward(float[] softOut, float[] dOut, float[] dIn, int m, int n)
        => _softmaxBackwardKernel(new Index1D(m), GetReadView(softOut), GetReadView(dOut), GetReadView(dIn), m, n);

    public void ScaleBackward(float s, float[] dOut, float[] dIn, int len)
        => _scaleBackwardKernel(new Index1D(len), s, GetReadView(dOut), GetReadView(dIn), len);

    public void AddBackward(float[] dOut, float[] dA, float[] dB, int len)
        => _addBackwardKernel(new Index1D(len), GetReadView(dOut), GetReadView(dA), GetReadView(dB), len);

    public void TransposeBackward(float[] dOut, float[] dIn, int m, int n)
        => _transposeBackwardKernel(new Index2D(m, n), GetReadView(dOut), GetReadView(dIn), m, n);

    // -------------------------------------------------------------------------
    // Static GPU kernels — must be static for ILGPU
    // Use XMath.* for GPU-safe math (MathF.* is CPU-only)
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
            sum += dOut[i * n + j] * w[p * n + j];
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
            sum += a[i * k + p] * dOut[i * n + j];
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

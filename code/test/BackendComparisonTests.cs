using System;
using NUnit.Framework;
using Shouldly;
using TinyBrain;

namespace TinyBrainTest;

[TestFixture]
public class BackendComparisonTests
{
    private static readonly GpuMatrixBackend? Gpu = GpuMatrixBackend.TryCreate();
    private static readonly CpuMatrixBackend Cpu = new();
    private const float Tolerance = 1e-4f;

    [OneTimeTearDown]
    public void TearDown() => Gpu?.Dispose();

    private static void SkipIfNoGpu()
    {
        if (Gpu is null)
            Assert.Ignore("CUDA not available — skipping GPU comparison test");
    }

    [Test]
    public void MatMul_CpuGpuAgreement()
    {
        SkipIfNoGpu();
        // [4,3] x [3,2] -> [4,2]
        float[] a = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        float[] w = { 1, 2, 3, 4, 5, 6 };
        var cpuOut = new float[8];
        var gpuOut = new float[8];
        Cpu.MatMul(a, w, cpuOut, 4, 3, 2);
        Gpu!.MatMul(a, w, gpuOut, 4, 3, 2);
        AssertClose(cpuOut, gpuOut);
    }

    [Test]
    public void MatMulBackwardLeft_CpuGpuAgreement()
    {
        SkipIfNoGpu();
        // dOut[4,2], w[3,2] -> dA[4,3]
        float[] dOut = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f };
        float[] w = { 1, 2, 3, 4, 5, 6 };
        var cpuDA = new float[12];
        var gpuDA = new float[12];
        Cpu.MatMulBackwardLeft(dOut, w, cpuDA, 4, 3, 2);
        Gpu!.MatMulBackwardLeft(dOut, w, gpuDA, 4, 3, 2);
        AssertClose(cpuDA, gpuDA);
    }

    [Test]
    public void MatMulBackwardRight_CpuGpuAgreement()
    {
        SkipIfNoGpu();
        // a[4,3], dOut[4,2] -> dW[3,2]
        float[] a = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        float[] dOut = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f };
        var cpuDW = new float[6];
        var gpuDW = new float[6];
        Cpu.MatMulBackwardRight(a, dOut, cpuDW, 4, 3, 2);
        Gpu!.MatMulBackwardRight(a, dOut, gpuDW, 4, 3, 2);
        AssertClose(cpuDW, gpuDW);
    }

    [Test]
    public void AddBias_CpuGpuAgreement()
    {
        SkipIfNoGpu();
        float[] a = { 1, 2, 3, 4, 5, 6 };
        float[] bias = { 10, 20, 30 };
        var cpuOut = new float[6];
        var gpuOut = new float[6];
        Cpu.AddBias(a, bias, cpuOut, 2, 3);
        Gpu!.AddBias(a, bias, gpuOut, 2, 3);
        AssertClose(cpuOut, gpuOut);
    }

    [Test]
    public void AddBiasBackward_CpuGpuAgreement()
    {
        SkipIfNoGpu();
        float[] dOut = { 1, 2, 3, 4, 5, 6 };
        var cpuDA = new float[6];
        var cpuDBias = new float[3];
        var gpuDA = new float[6];
        var gpuDBias = new float[3];
        Cpu.AddBiasBackward(dOut, cpuDA, cpuDBias, 2, 3);
        Gpu!.AddBiasBackward(dOut, gpuDA, gpuDBias, 2, 3);
        AssertClose(cpuDA, gpuDA);
        AssertClose(cpuDBias, gpuDBias);
    }

    [Test]
    public void Tanh_CpuGpuAgreement()
    {
        SkipIfNoGpu();
        float[] input = { -2f, -1f, 0f, 1f, 2f };
        var cpuOut = new float[5];
        var gpuOut = new float[5];
        Cpu.Tanh(input, cpuOut, 5);
        Gpu!.Tanh(input, gpuOut, 5);
        AssertClose(cpuOut, gpuOut);
    }

    [Test]
    public void TanhBackward_CpuGpuAgreement()
    {
        SkipIfNoGpu();
        float[] tanhOut = { -0.9f, -0.5f, 0f, 0.5f, 0.9f };
        float[] dOut = { 1f, 1f, 1f, 1f, 1f };
        var cpuDIn = new float[5];
        var gpuDIn = new float[5];
        Cpu.TanhBackward(tanhOut, dOut, cpuDIn, 5);
        Gpu!.TanhBackward(tanhOut, dOut, gpuDIn, 5);
        AssertClose(cpuDIn, gpuDIn);
    }

    [Test]
    public void Softmax_CpuGpuAgreement()
    {
        SkipIfNoGpu();
        float[] input = { 1f, 2f, 3f, 1f, 1f, 1f };
        var cpuOut = new float[6];
        var gpuOut = new float[6];
        Cpu.Softmax(input, cpuOut, 2, 3);
        Gpu!.Softmax(input, gpuOut, 2, 3);
        AssertClose(cpuOut, gpuOut);
    }

    [Test]
    public void SoftmaxBackward_CpuGpuAgreement()
    {
        SkipIfNoGpu();
        float[] softOut = { 0.1f, 0.7f, 0.2f, 0.3f, 0.4f, 0.3f };
        float[] dOut = { 1f, 0f, 0f, 0f, 1f, 0f };
        var cpuDIn = new float[6];
        var gpuDIn = new float[6];
        Cpu.SoftmaxBackward(softOut, dOut, cpuDIn, 2, 3);
        Gpu!.SoftmaxBackward(softOut, dOut, gpuDIn, 2, 3);
        AssertClose(cpuDIn, gpuDIn);
    }

    [Test]
    public void Scale_CpuGpuAgreement()
    {
        SkipIfNoGpu();
        float[] input = { 1f, 2f, 3f, 4f };
        var cpuOut = new float[4];
        var gpuOut = new float[4];
        Cpu.Scale(input, 0.5f, cpuOut, 4);
        Gpu!.Scale(input, 0.5f, gpuOut, 4);
        AssertClose(cpuOut, gpuOut);
    }

    [Test]
    public void ScaleBackward_CpuGpuAgreement()
    {
        SkipIfNoGpu();
        float[] dOut = { 1f, 2f, 3f, 4f };
        var cpuDIn = new float[4];
        var gpuDIn = new float[4];
        Cpu.ScaleBackward(0.5f, dOut, cpuDIn, 4);
        Gpu!.ScaleBackward(0.5f, dOut, gpuDIn, 4);
        AssertClose(cpuDIn, gpuDIn);
    }

    [Test]
    public void Add_CpuGpuAgreement()
    {
        SkipIfNoGpu();
        float[] a = { 1f, 2f, 3f };
        float[] b = { 4f, 5f, 6f };
        var cpuOut = new float[3];
        var gpuOut = new float[3];
        Cpu.Add(a, b, cpuOut, 3);
        Gpu!.Add(a, b, gpuOut, 3);
        AssertClose(cpuOut, gpuOut);
    }

    [Test]
    public void AddBackward_CpuGpuAgreement()
    {
        SkipIfNoGpu();
        float[] dOut = { 1f, 2f, 3f };
        var cpuDA = new float[3];
        var cpuDB = new float[3];
        var gpuDA = new float[3];
        var gpuDB = new float[3];
        Cpu.AddBackward(dOut, cpuDA, cpuDB, 3);
        Gpu!.AddBackward(dOut, gpuDA, gpuDB, 3);
        AssertClose(cpuDA, gpuDA);
        AssertClose(cpuDB, gpuDB);
    }

    [Test]
    public void Transpose_CpuGpuAgreement()
    {
        SkipIfNoGpu();
        float[] input = { 1, 2, 3, 4, 5, 6 };  // [2,3]
        var cpuOut = new float[6];
        var gpuOut = new float[6];
        Cpu.Transpose(input, cpuOut, 2, 3);
        Gpu!.Transpose(input, gpuOut, 2, 3);
        AssertClose(cpuOut, gpuOut);
    }

    [Test]
    public void TransposeBackward_CpuGpuAgreement()
    {
        SkipIfNoGpu();
        float[] dOut = { 1, 2, 3, 4, 5, 6 };  // [3,2] (transposed of [2,3])
        var cpuDIn = new float[6];
        var gpuDIn = new float[6];
        Cpu.TransposeBackward(dOut, cpuDIn, 2, 3);
        Gpu!.TransposeBackward(dOut, gpuDIn, 2, 3);
        AssertClose(cpuDIn, gpuDIn);
    }

    [Test]
    public void FullForwardBackward_GpuMatchesCpu()
    {
        SkipIfNoGpu();

        // Build a 2-layer chain: [2,3] -MatMul-> [2,2] -AddBias-> [2,2] -Tanh-> [2,2]
        var weights = Operand.Of(new float[,] { { 0.1f, 0.2f }, { 0.3f, 0.4f }, { 0.5f, 0.6f } });
        var bias = Operand.OfZero(1, 2);
        var input = Operand.Of(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });

        // CPU run
        Operand.SetBackend(Cpu);
        var cpuWeights = Operand.Of(new float[,] { { 0.1f, 0.2f }, { 0.3f, 0.4f }, { 0.5f, 0.6f } });
        var cpuBias = Operand.OfZero(1, 2);
        var cpuInput = Operand.Of(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        var cpuOut = cpuInput.MatMul(cpuWeights).AddBias(cpuBias).Tanh().Sum();
        cpuOut.Backpropagation();

        // GPU run
        Operand.SetBackend(Gpu!);
        var gpuWeights = Operand.Of(new float[,] { { 0.1f, 0.2f }, { 0.3f, 0.4f }, { 0.5f, 0.6f } });
        var gpuBias = Operand.OfZero(1, 2);
        var gpuInput = Operand.Of(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        var gpuOut = gpuInput.MatMul(gpuWeights).AddBias(gpuBias).Tanh().Sum();
        gpuOut.Backpropagation();

        // Restore CPU as default so other tests are unaffected
        Operand.SetBackend(Cpu);

        AssertClose(cpuOut.Data, gpuOut.Data);
        AssertClose(cpuWeights.Gradient, gpuWeights.Gradient);
        AssertClose(cpuBias.Gradient, gpuBias.Gradient);
        AssertClose(cpuInput.Gradient, gpuInput.Gradient);
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        actual.Length.ShouldBe(expected.Length);
        for (var i = 0; i < expected.Length; i++)
            actual[i].ShouldBe(expected[i], Tolerance,
                $"index {i}: expected {expected[i]} but got {actual[i]}");
    }
}

using System;

namespace TinyBrain;

public interface IMatrixBackend : IDisposable
{
    // Forwards — write output[] fresh, do not accumulate

    void MatMul(float[] a, float[] w, float[] output, int m, int k, int n);
    // a[m,k] x w[k,n] -> output[m,n]

    void AddBias(float[] a, float[] bias, float[] output, int m, int n);
    // a[m,n] + bias[1,n] broadcast -> output[m,n]

    void Tanh(float[] input, float[] output, int len);
    // output[i] = tanh(input[i])

    void Softmax(float[] input, float[] output, int m, int n);
    // per-row numerically stable softmax

    void Scale(float[] input, float s, float[] output, int len);
    // output[i] = s * input[i]

    void Add(float[] a, float[] b, float[] output, int len);
    // output[i] = a[i] + b[i]

    void Transpose(float[] input, float[] output, int m, int n);
    // input[m,n] -> output[n,m]

    // Backwards — accumulate into gradient buffers (+=)

    void MatMulBackwardLeft(float[] dOut, float[] w, float[] dA, int m, int k, int n);
    // dA[m,k] += dOut[m,n] x W[k,n]^T

    void MatMulBackwardRight(float[] a, float[] dOut, float[] dW, int m, int k, int n);
    // dW[k,n] += A[m,k]^T x dOut[m,n]

    void AddBiasBackward(float[] dOut, float[] dA, float[] dBias, int m, int n);
    // dA[i*n+j] += dOut[i*n+j];  dBias[j] += sum_i dOut[i*n+j]

    void TanhBackward(float[] tanhOut, float[] dOut, float[] dIn, int len);
    // dIn[i] += (1 - tanhOut[i]^2) * dOut[i]

    void SoftmaxBackward(float[] softOut, float[] dOut, float[] dIn, int m, int n);
    // per-row: dot = dot(dOut_row, softOut_row); dIn[j] += softOut[j] * (dOut[j] - dot)

    void ScaleBackward(float s, float[] dOut, float[] dIn, int len);
    // dIn[i] += s * dOut[i]

    void AddBackward(float[] dOut, float[] dA, float[] dB, int len);
    // dA[i] += dOut[i];  dB[i] += dOut[i]

    void TransposeBackward(float[] dOut, float[] dIn, int m, int n);
    // original was [m,n]->[n,m]; dOut is [n,m]; dIn[i,j] += dOut[j,i]

    // Sync primitives — no-op on CPU; manage host/device coherency on GPU
    void Synchronize(float[] host);      // GPU -> CPU if GPU is authoritative
    void InvalidateDevice(float[] host); // mark HostNewer=true after CPU writes
}

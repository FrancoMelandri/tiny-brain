#nullable enable
using TinyBrain;
using System;

Console.WriteLine("Attempting GpuMatrixBackend.TryCreate(verbose: true)...\n");
var backend = GpuMatrixBackend.TryCreate(verbose: true);

if (backend == null)
{
    Console.WriteLine("\nResult: FAILED — returned null");
}
else
{
    Console.WriteLine($"\nResult: SUCCESS — {backend.GetType().Name} created");

    // Run a tiny MatMul to confirm compute works
    float[] a   = { 1, 0, 0, 1 };  // [2,2] identity
    float[] w   = { 2, 3, 4, 5 };  // [2,2]
    float[] out_ = new float[4];
    backend.MatMul(a, w, out_, 2, 2, 2);
    Console.WriteLine($"MatMul test: [{out_[0]}, {out_[1]}, {out_[2]}, {out_[3]}]  (expected [2,3,4,5])");

    backend.Dispose();
}

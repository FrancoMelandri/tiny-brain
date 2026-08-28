# GPU Acceleration & Improved Sampling

## Overview

Two major features were added to tiny-brain in this session:

1. **GPU acceleration** — a backend abstraction that lets all matrix operations run on an NVIDIA
   GPU via ILGPU/CUDA, with the CPU path preserved as the default fallback.
2. **Temperature + Top-k sampling** — a proper generation strategy for the SLM use-case that
   replaces the broken `Multinomial` helper.

---

## 1. GPU Acceleration

### Motivation

The SLM training loop is dominated by `MatMul` calls (forward and backward through `Brain` and
`AttentionHead`). On CPU, `TensorPrimitives.MultiplyAdd` with AVX-512 is fast for small matrices,
but it does not scale. Moving compute to a GPU (RTX A2000 8 GB) unlocks parallelism that matters
once the batch dimension is large enough.

### Architecture: `IMatrixBackend`

A single interface seam was introduced in `code/src/Compute/IMatrixBackend.cs`.
All hot-path forward and backward operations are declared here:

| Group | Methods |
|---|---|
| Forward | `MatMul`, `AddBias`, `Tanh`, `Softmax`, `Scale`, `Add`, `Transpose` |
| Backward | `MatMulBackwardLeft`, `MatMulBackwardRight`, `AddBiasBackward`, `TanhBackward`, `SoftmaxBackward`, `ScaleBackward`, `AddBackward`, `TransposeBackward` |
| Coherency | `Synchronize`, `InvalidateDevice` |

`Operand` holds a `static volatile IMatrixBackend _backend` (default = `CpuMatrixBackend`).
Switching backends at startup is a single call: `Operand.SetBackend(backend)`.

Operations not included (not on the hot path): `NLL`, `MaskFill`, `SliceRow`, `Sum`,
`ApplyGradients`, `GradientNormSquared`.

### `CpuMatrixBackend`

A direct extraction of the existing `TensorPrimitives`-based loops from `Operand`.
Stateless sealed class. No behaviour change from before.

### `GpuMatrixBackend` (ILGPU 1.5.3)

`code/src/Compute/GpuMatrixBackend.cs`

**Lifecycle:**

```csharp
// Auto-detect — returns null if no CUDA device is found
var backend = GpuMatrixBackend.TryCreate(verbose: true)
              ?? (IMatrixBackend)new CpuMatrixBackend();
Operand.SetBackend(backend);
```

The `--backend cpu|gpu` CLI flag overrides auto-detection.

**Kernels** — all are `static` methods compiled to PTX by ILGPU at startup:

| Kernel | Index | Hot path |
|---|---|---|
| `MatMulKernel` | `Index2D(m, n)` | ✓ most expensive |
| `MatMulBackwardLeftKernel` | `Index2D(m, k)` | ✓ |
| `MatMulBackwardRightKernel` | `Index2D(k, n)` | ✓ |
| `TanhKernel / TanhBackwardKernel` | `Index1D` | ✓ |
| `SoftmaxKernel / SoftmaxBackwardKernel` | `Index1D(m rows)` | ✓ |
| `ScaleKernel / ScaleBackwardKernel` | `Index1D` | used in attention |
| `AddKernel / AddBackwardKernel` | `Index1D` | residual connections |
| `TransposeKernel / TransposeBackwardKernel` | `Index2D(m, n)` | attention QKᵀ |
| `AddBiasKernel` | `Index2D(m, n)` | every layer |
| `AddBiasBackwardInputKernel / BiasKernel` | `Index1D` | bias gradient |

`XMath.Tanh` and `XMath.Exp` from `ILGPU.Algorithms` are used inside kernels (GPU-safe math).
The context must be created with `Context.Create(b => b.Default().EnableAlgorithms())`.

**Important:** `GgufSerializer.Write` (used from `code/use-cases/slm/Program.cs`) also received
new nullable KV parameters to store vocabulary and story offset in the GGUF file — see
`code/src/Serialization/GgufSerializer.cs`.

### V1 → V2: Device-Resident Buffers

**V1 (per-operation round-trip):**

```
Each operation: allocate → CopyFromCPU → kernel → Synchronize → CopyToCPU → Dispose
```

Every MatMul was doing ~3 PCIe transfers. For the matrix sizes in the SLM
(`[1×3200]×[3200×128]`, `[1×128]×[128×1000]`), the bus overhead exceeded compute time and the
GPU was slower than CPU.

**V2 (device-resident buffers):**

`GpuMatrixBackend` maintains a `ConditionalWeakTable<float[], DeviceBuffer>` that maps each CPU
`float[]` to a GPU `MemoryBuffer1D<float>` mirror. Two states:

| `HostNewer` | Meaning |
|---|---|
| `true` | CPU array was written (e.g. after `ZeroGradient`, `ApplyGradients`, first use) — upload before next kernel |
| `false` | GPU is authoritative — use device buffer directly, no upload needed |

Two primitives control coherency:

- **`Synchronize(float[] host)`** — downloads GPU→CPU if `HostNewer=false` (called before any
  CPU-side read of a tensor that may be GPU-resident: `NLL`, `GradientNormSquared`,
  `ApplyGradients`, validation `SoftmaxRow`, generation `SoftmaxRow`)
- **`InvalidateDevice(float[] host)`** — sets `HostNewer=true` (called after any CPU write:
  `ZeroGradient`, `ApplyGradients`, `Backpropagation` seed)

With V2, a full training step has approximately 3 PCIe round-trips (loss sync, gradient norm,
parameter update) instead of ~50 MB per step in V1. Parameters upload once after each
`ApplyGradients`; all forward + backward kernels run entirely on GPU.

### Mini-Batch Training

Because the SLM trained sample-by-sample (`m=1`), GPU occupancy was 5–40%.
`[1×3200]×[3200×128]` launches 128 threads against 2560 CUDA cores.

Mini-batch stacks the attention outputs for `B` samples into a `[B, embedDim]` matrix before
the Brain layers:

```
B samples → embed+attention (sequential) → Operand.Stack → [B, embedDim] → Brain → [B, vocabSize]
```

`Operand.Stack` required extending the autograd graph from binary (`_previous = (Left, Right)`) to
N-ary (`_previousN : Operand[]`). `BuildTopological` was updated to traverse `_previousN`.

With B=32:

| Layer | Old shape (B=1) | New shape (B=32) | Threads |
|---|---|---|---|
| Hidden | `[1×64]×[64×128]` | `[32×64]×[64×128]` | 4 096 |
| Output | `[1×128]×[128×1000]` | `[32×128]×[128×1000]` | 32 000 |

CLI: `--batch-size N` (default 32).

### GPU sync bug in validation and generation

`SoftmaxRow(logits.Data)` reads the CPU `float[]` directly. With V2, `logits.Data` is stale
after a GPU forward pass. Fix: call `Operand.SynchronizeDeviceArray(logits.Data)` before any
`SoftmaxRow` that is not going through `NLL`:

```csharp
// validation
var logitsVal = model.Forward(p.Context);
Operand.SynchronizeDeviceArray(logitsVal.Data);
return -MathF.Log(SoftmaxRow(logitsVal.Data)[p.Target] + 1e-6f);

// generation
var logits = model.Forward(genContext);
Operand.SynchronizeDeviceArray(logits.Data);
var next = Sample(logits.Data, temperature, topK);
```

---

## 2. Temperature + Top-k Sampling

### Motivation

The original `Multinomial` helper had two bugs:
- Created `new Random()` on every token → same seed for consecutive calls → repeated tokens
- No control over distribution sharpness or vocabulary restriction

### Algorithm

`Sample(float[] logits, float temperature, int topK, Random rng)` in `Program.cs`:

```
k = 1  →  return argmax(logits)              greedy, deterministic

k > 1  →  scaled[i] = logits[i] / temperature
           probs = softmax(scaled)
           keep exactly top-k indices by descending prob (index-sorted, handles ties)
           zero out the rest, renormalise
           inverse-CDF sample from filtered distribution
```

**Temperature** controls sharpness applied to raw logits *before* softmax:
- `T < 1.0` (e.g. 0.7) — sharpens, model is more confident, less varied output
- `T = 1.0` — no change to the distribution
- `T > 1.0` (e.g. 1.2) — flattens, more surprising / creative output

**Top-k** restricts sampling to the k most probable tokens.
Setting `k = 1` gives pure greedy decoding (no randomness, no softmax needed).

### CLI flags

```
--top-k N          default: 10
--temperature T    default: 1.0
```

Examples:

```bash
# greedy
dotnet run -- --top-k 1

# focused + cool
dotnet run -- --top-k 5 --temperature 0.7

# creative + warm
dotnet run -- --top-k 50 --temperature 1.2
```

---

## Files changed

| File | Change |
|---|---|
| `code/src/Compute/IMatrixBackend.cs` | New — 16-method interface + Synchronize/InvalidateDevice |
| `code/src/Compute/CpuMatrixBackend.cs` | New — TensorPrimitives CPU implementation |
| `code/src/Compute/GpuMatrixBackend.cs` | New — ILGPU CUDA implementation with device-resident buffers |
| `code/src/Matrix/Operand.cs` | SetBackend static seam, sync/invalidate at CPU boundaries, `Stack` op, `_previousN` |
| `code/src/Serialization/GgufSerializer.cs` | `Write` extended with KV params; `ReadWithMetadata` added |
| `code/src/tiny-brain.csproj` | Added `ILGPU` + `ILGPU.Algorithms` 1.5.3 |
| `code/use-cases/slm/EmbeddingTable.cs` | Unchanged (CPU-only, compatible with V2 sync model) |
| `code/use-cases/slm/SlmModel.cs` | `ForwardBatch(int[][] contexts)` added |
| `code/use-cases/slm/TrainingData.cs` | `Batches(int batchSize)` added |
| `code/use-cases/slm/Tokenizer.cs` | `Tokenizer(IReadOnlyList<string>)` constructor + `Words` property |
| `code/use-cases/slm/DatasetLoader.cs` | `skipStories` param + `CountStories` method |
| `code/use-cases/slm/Program.cs` | Full rewrite: backend CLI, batch training, story cursor, Sample |
| `code/tools/gpu-diag/` | New — GPU diagnostic tool (part of solution) |
| `code/test/BackendComparisonTests.cs` | New — CPU vs GPU numerical agreement tests (16 tests) |

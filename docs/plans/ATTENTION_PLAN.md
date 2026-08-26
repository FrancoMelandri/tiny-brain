# Attention Layer Plan — Decoder-Only Transformer Style

## Overview

Add a single causal self-attention head (decoder-only, no multi-head for now) between the embedding and the existing MLP in `SlmModel`.

### Data flow after the change

```
EmbeddingTable  →  [T, d_model]
AttentionHead   →  [T, d_model]   (with causal mask)
slice last row  →  [1, d_model]   (predict next token from last position)
Brain (hidden)  →  [1, hiddenSize]
Brain (output)  →  [1, vocabSize]
softmax + NLL
```

---

## New Trainable Parameters

All four are `Operand` matrices initialized with small random values.

| Matrix | Shape | Role |
|---|---|---|
| `Wq` | `[d_model, d_head]` | project input to queries |
| `Wk` | `[d_model, d_head]` | project input to keys |
| `Wv` | `[d_model, d_head]` | project input to values |
| `Wo` | `[d_head, d_model]` | project attention output back to model dim |

`d_head` can equal `d_model` for a single-head setup, or be smaller (`d_model / num_heads`) when multi-head is added later.

---

## New `Operand` Operations

All operations satisfy the existing `_previous = (Left, Right)` binary-node constraint (each op is unary or binary).

| Op | Signature | Why needed |
|---|---|---|
| `Transpose()` | `[m,n]` → `[n,m]` | compute `Q · Kᵀ` |
| `Scale(double s)` | `[m,n]` → `[m,n]` | divide scores by `√d_head` — no new params |
| `MaskFill(bool[,] mask, double fill)` | `[m,n]` → `[m,n]` | causal mask — use `-1e9`, not `-∞` |
| `RowSoftmax()` | `[T,T]` → `[T,T]` | per-row softmax on score matrix |
| `Add(Operand other)` | `[m,n] + [m,n]` → `[m,n]` | residual connection (full-shape, not broadcast) |
| `SliceRow(int i)` | `[T,d]` → `[1,d]` | extract last position after attention |

`AddBias` already handles `[m,n] + [1,n]` broadcast and is unchanged.

### `RowSoftmax` backward (trickiest op)

For each row `i`, the backward is:

```
dL/dz_j = p_j * (dL/dp_j - Σ_k p_k * dL/dp_k)
```

The backward closure must capture the output `p` values (same pattern as the existing `Softmax()`).

---

## Attention Forward Pass (all tracked `Operand` ops)

```
Q  = X.MatMul(Wq)                        // [T, d_model] × [d_model, d_head] → [T, d_head]
K  = X.MatMul(Wk)                        // [T, d_head]
V  = X.MatMul(Wv)                        // [T, d_head]

scores  = Q.MatMul(K.Transpose())        // [T, d_head] × [d_head, T] → [T, T]
scores  = scores.Scale(1 / √d_head)      // [T, T] — fixed scalar, no new params
scores  = scores.MaskFill(causalMask, -1e9)   // upper triangle → -1e9
weights = scores.RowSoftmax()            // [T, T], each row sums to 1

ctx = weights.MatMul(V)                  // [T, T] × [T, d_head] → [T, d_head]
out = ctx.MatMul(Wo)                     // [T, d_head] × [d_head, d_model] → [T, d_model]
out = out.Add(X)                         // residual: [T, d_model] + [T, d_model]
```

---

## Files to Add / Modify

### New file
- `code/src/Attention/AttentionHead.cs`
  - Holds `Wq`, `Wk`, `Wv`, `Wo` as `Operand` fields
  - Constructor receives `(int dModel, int dHead, int contextSize)` and builds the causal mask
  - `Forward(Operand x)` implements the forward pass above
  - `ParameterMatrices` exposes all four weight matrices for save/load

### Modify
- `code/src/Matrix/Operand.cs` — add the six new ops with their `_backward` closures
- `code/use-cases/slm/EmbeddingTable.cs` — add `LookupSequence(int[] indices)` returning `[T, d_model]`; keep `LookupFlat` for backward compat
- `code/use-cases/slm/SlmModel.cs`
  - Constructor gains `int dHead` parameter
  - Add `AttentionHead _attention`
  - `Forward` calls `LookupSequence` → `_attention.Forward` → `SliceRow(T-1)` → `_brain.Forward`
  - `ParameterMatrices` includes `_attention.ParameterMatrices`

---

## Critical Risks

1. **Causal mask is fixed at construction.** Build it once as `bool[T,T]` in `AttentionHead`'s constructor (upper triangle = true). Do not recompute every forward pass.

2. **`-1e9` not `-∞` for mask fill.** `exp(-∞)` propagates `NaN` into softmax backward (0 × ∞). `-1e9` gives numerically zero attention weight with clean gradients.

3. **Gradient accumulation through the residual `Add`.** Both the attention output and the original `X` receive gradient. `X.Gradient` already accumulates from the Q/K/V projections; `Add` just adds more into it — this is correct since gradients accumulate (`+=`).

4. **`SliceRow` backward is a scatter.** Only the sliced row receives gradient; all other rows get zero. The closure must capture the row index and the full `[T, d_model]` shape.

5. **Parameter save/load ordering.** The existing `FlatParameters` walks `ParameterMatrices` in order. Adding attention params changes the layout and will break any existing `parameters.txt`. Delete the file on first run after the change.

6. **Gradient clipping.** The training loop sums `GradientNormSquared()` over all param matrices. Add `_attention`'s matrices to that sum.

---

## Suggested Build Order

1. Add `Transpose`, `Scale`, `Add` to `Operand` + unit tests
2. Add `MaskFill` + `RowSoftmax` (harder) + unit tests
3. Add `SliceRow` + unit test
4. Implement `AttentionHead` + integration test with a small known input
5. Add `LookupSequence` to `EmbeddingTable`
6. Wire `SlmModel` together and run the use-case

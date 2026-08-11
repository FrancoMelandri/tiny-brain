# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build entire solution
dotnet build code/tiny-brain.sln

# Run all tests
dotnet test code/test/tiny-brain-test.csproj

# Run tests with coverage
dotnet test code/test/tiny-brain-test.csproj /p:CollectCoverage=true /p:CoverletOutput=TestResults/ /p:CoverletOutputFormat=lcov

# Run a single test by name
dotnet test code/test/tiny-brain-test.csproj --filter "FullyQualifiedName~<TestName>"

# Run the bigram use-case
dotnet run --project code/use-cases/biagram/biagram.csproj

# Run the playground
dotnet run --project code/use-cases/playground/playgroung.csproj
```

## Architecture

The library is a from-scratch implementation of a neural network with autograd (inspired by micrograd / cs231n). Target: .NET 8, namespace `TinyBrain`.

### Core abstraction: `Operand`

`Operand` (`code/src/Expression/Operand.cs`) is the fundamental building block — a scalar value that carries its gradient and a `_backward` closure. All arithmetic operators (`+`, `-`, `*`, `/`) and unary ops (`Exp`, `Log`, `Pow`, `Relu`, `Tanh`) are overloaded to return new `Operand` instances that record their `Previous` pair. Calling `Backpropagation()` on an `Operand` runs a non-recursive topological sort over the expression graph and then invokes each node's `_backward` in reverse order.

### Neural network layers

```
Brain  (MLP)
  └─ Layer[]            (one per hidden/output layer)
       └─ Neuron[]      (each neuron owns its weights + bias as Operands)
```

- `Neuron` (`code/src/Neuron/Neuron.cs`) computes `f(Σ(xᵢ·wᵢ) + b)` where `f` is determined by `ActivationType`.
- `Layer` (`code/src/Layers/Layer.cs`) fans inputs through all neurons; `Forward` returns one `Operand` per neuron.
- `Brain` (`code/src/MLP/Brain.cs`) chains layers via `Fold`; `Forward` auto-zeros gradients before each pass.

### Activation functions

`Activations.cs` holds the dispatch dictionary keyed on `ActivationType` (currently `None` and `Tanh`). `Tanh` is expressed in terms of `Exp`, `/`, and `-` on `Operand` so its gradient flows automatically through the graph — no hand-coded derivative needed.

### Functional style

The library depends on `tiny-fp` (functional primitives: `Option`, `Unit`, `Map`, `Tee`, `Fold`, `ForEach`). Prefer these combinators over imperative loops; `Unit` is used as the return type of side-effecting void-like operations.

### Use-cases

`code/use-cases/biagram/` implements a character-level bigram language model using a `Brain(27 → 27)` with `ActivationType.None` followed by a manual softmax. Parameters can be saved/loaded via `SaveParameters` / `LoadParameters` to `parameters.txt`.

`code/use-cases/playground/` is a scratch area for manual experiments.

### Test structure

Tests use NUnit 3 + Shouldly assertions. `MLPTrainingTests.Training` is a full end-to-end gradient-descent loop that runs until loss < 0.00001 or 10 000 steps — it is intentionally slow and exercises the entire autograd pipeline.

using System;

namespace TinyBrain;

public class MatrixLayer
{
    public MatrixOperand Weights { get; }
    public MatrixOperand Bias { get; }
    public ActivationType ActivationType { get; }

    public MatrixLayer(int inputSize, int outputSize, ActivationType activationType)
    {
        var scale = Math.Sqrt(2.0 / inputSize);  // He init
        Weights = MatrixOperand.OfRandom(inputSize, outputSize, scale);
        Bias = MatrixOperand.OfZero(1, outputSize);
        ActivationType = activationType;
    }

    public MatrixOperand[] ParameterMatrices => [Weights, Bias];

    public void ZeroGradients()
    {
        Weights.ZeroGradient();
        Bias.ZeroGradient();
    }

    public MatrixOperand Forward(MatrixOperand input)
    {
        var z = input.MatMul(Weights).AddBias(Bias);
        return ActivationType == ActivationType.Tanh ? z.Tanh() : z;
    }
}

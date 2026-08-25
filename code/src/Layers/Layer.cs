using System;

namespace TinyBrain;

public class Layer
{
    public Operand Weights { get; }
    public Operand Bias { get; }
    public ActivationType ActivationType { get; }

    public Layer(int inputSize, int outputSize, ActivationType activationType)
    {
        var scale = Math.Sqrt(2.0 / inputSize);
        Weights = Operand.OfRandom(inputSize, outputSize, scale);
        Bias = Operand.OfZero(1, outputSize);
        ActivationType = activationType;
    }

    public Operand[] ParameterMatrices => [Weights, Bias];

    public void ZeroGradients()
    {
        Weights.ZeroGradient();
        Bias.ZeroGradient();
    }

    public Operand Forward(Operand input)
    {
        var z = input.MatMul(Weights).AddBias(Bias);
        return ActivationType == ActivationType.Tanh ? z.Tanh() : z;
    }
}

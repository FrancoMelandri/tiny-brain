namespace TinyBrain;

public interface INeuron
{
    string Id { get; }
    Operand[] Weights { get; }
    Operand Bias { get; }
    Operand Forward(Operand[] inputs);
}
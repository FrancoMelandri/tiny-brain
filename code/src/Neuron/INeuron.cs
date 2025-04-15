using TinyFp;

namespace TinyBrain;

public interface INeuron
{
    string Id { get; }
    Operand[] Weights { get; }
    Operand Bias { get; }
    Unit ZeroGradient();
    Operand Forward(Operand[] inputs);
}
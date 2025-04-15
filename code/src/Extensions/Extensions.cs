using System.Collections.Generic;

namespace TinyBrain;

public static class Extensions
{
    public static IEnumerable<Operand> AsOperands(this object operands)
        => operands as IEnumerable<Operand>;

    public static Operand AsOperand(this object operands)
        => operands as Operand;
}
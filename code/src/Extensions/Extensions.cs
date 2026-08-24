using System.Collections.Generic;
using System.Linq;

namespace TinyBrain;

public static class Extensions
{
    public static IEnumerable<Operand> AsOperands(this object operands)
        => operands as IEnumerable<Operand>;

    public static Operand AsOperand(this object operands)
        => operands as Operand;

    public static Operand[] Softmax(this Operand[] logits)
    {
        var maxLogit = logits.Max(l => l.Data);
        var exps = logits.Select(l => (l - maxLogit).Exp()).ToArray();
        var sumOp = exps.Aggregate(Operand.Of(0.0), (a, e) => a + e);
        return exps.Select(e => e / sumOp).ToArray();
    }
}
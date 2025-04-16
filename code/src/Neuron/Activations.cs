using System;
using System.Collections.Generic;
using TinyFp.Extensions;

namespace TinyBrain;

public enum ActivationType
{
    None,
    Tanh
}

public static class Activations
{
    private static Dictionary<ActivationType, Func<Operand, Operand>> activationFunctions = new()
    {
        { ActivationType.None, operand => operand },
        { ActivationType.Tanh, operand => operand.Tanh() }
    };

    public static Operand Activation(this Neuron @this, Operand body)
        => activationFunctions[@this.ActivationType](body);

    public static Operand Tanh(this Operand @this)
        => @this
            .Map(operand => 2 * operand)
            .Map(operand => operand.Exp())
            .Map(operand => (operand - 1) / (operand + 1));   
}
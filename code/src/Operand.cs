using System;
using System.Collections.Generic;
using TinyFp;
using TinyFp.Extensions;
using static TinyBrain.Constants;

namespace TinyBrain;

public class Operand
{
    public (Option<Operand> Operand1, Option<Operand> Operand2) Previous { get; }
    public double Data { get; set;  }
    public double Gradient { get; set; }
    public string Label { get; set; }
    private Func<Unit> _backward; 

    private Operand()
    {
        Previous = (Option<Operand>.None(), Option<Operand>.None());
        Gradient = ZERO;
        Label = string.Empty;
        _backward = () => Unit.Default;
    }

    private Operand(double data)
        : this()
    {
        Data = data;
    }

    private Operand(double data, string label)
        : this(data)
    {
        Label = label;
    }

    private Operand(double data, (Operand, Operand) previous)
        : this(data)
    {
        Previous = (previous.Item1.ToOption(), previous.Item2.ToOption());
    }

    public static Operand Of(double data) => new(data);

    public static Operand Of(double data, string label) => new(data, label);

    public static Operand operator +(Operand a, Operand b)
        => new Operand(a.Data + b.Data, (a, b))
            .Tee(_ => _._backward = () =>
                Unit.Default
                    .Tee(__ => a.Gradient += GRADIENT_PLUS * _.Gradient)
                    .Tee(__ => b.Gradient += GRADIENT_PLUS * _.Gradient));

    public static Operand operator +(Operand a, double bval)
        => new Operand(bval)
            .Map(b => new Operand(a.Data + b.Data, (a, b))
                    .Tee(_ => _._backward = () =>
                        Unit.Default
                            .Tee(__ => a.Gradient += GRADIENT_PLUS * _.Gradient)
                            .Tee(__ => b.Gradient += GRADIENT_PLUS * _.Gradient)));

    public static Operand operator +(double aval, Operand b)
        => new Operand(aval)
            .Map(a => new Operand(a.Data + b.Data, (a, b))
                .Tee(_ => _._backward = () =>
                    Unit.Default
                        .Tee(__ => a.Gradient += GRADIENT_PLUS * _.Gradient)
                        .Tee(__ => b.Gradient += GRADIENT_PLUS * _.Gradient)));

    public static Operand operator -(Operand a, Operand b)
        => new Operand(a.Data - b.Data, (a, b))
            .Tee(_ => _._backward = () =>
                Unit.Default
                    .Tee(__ => a.Gradient += GRADIENT_MINUS * _.Gradient)
                    .Tee(__ => b.Gradient += GRADIENT_MINUS * _.Gradient));

    public static Operand operator -(Operand a, double bval)
        => new Operand(bval)
            .Map(b => new Operand(a.Data - b.Data, (a, b))
                .Tee(_ => _._backward = () =>
                    Unit.Default
                        .Tee(__ => a.Gradient += GRADIENT_MINUS * _.Gradient)
                        .Tee(__ => b.Gradient += GRADIENT_MINUS * _.Gradient)));

    public static Operand operator -(double aval, Operand b)
        => new Operand(aval)
            .Map(a => new Operand(a.Data - b.Data, (a, b))
                .Tee(_ => _._backward = () =>
                    Unit.Default
                        .Tee(__ => a.Gradient += GRADIENT_MINUS * _.Gradient)
                        .Tee(__ => b.Gradient += GRADIENT_MINUS * _.Gradient)));

    public static Operand operator *(Operand a, Operand b)
        => new Operand(a.Data * b.Data, (a, b))
            .Tee(_ => _._backward = () =>
                Unit.Default
                    .Tee(__ => a.Gradient += b.Data * _.Gradient)
                    .Tee(__ => b.Gradient += a.Data * _.Gradient));

    public static Operand operator *(Operand a, double bval)
        => new Operand(bval)
            .Map(b => new Operand(a.Data * b.Data, (a, b))
                                    .Tee(_ => _._backward = () =>
                                        Unit.Default
                                            .Tee(__ => a.Gradient += b.Data * _.Gradient)
                                            .Tee(__ => b.Gradient += a.Data * _.Gradient)));

    public static Operand operator *(double aval, Operand b)
        => new Operand(aval)
            .Map(a => new Operand(a.Data * b.Data, (a, b))
                .Tee(_ => _._backward = () =>
                    Unit.Default
                        .Tee(__ => a.Gradient += b.Data * _.Gradient)
                        .Tee(__ => b.Gradient += a.Data * _.Gradient)));

    public static Operand operator /(Operand a, Operand b)
        => new Operand(a.Data / b.Data, (a, b))
            .Tee(_ => _._backward = () =>
                Unit.Default
                    .Tee(__ => a.Gradient += 1/b.Data * _.Gradient)
                    .Tee(__ => b.Gradient += (-a.Data/Math.Pow(b.Data, 2)) * _.Gradient));

    public static Operand operator /(Operand a, double bval)
        => new Operand(bval)
            .Map(b => new Operand(a.Data / b.Data, (a, b))
                                    .Tee(_ => _._backward = () =>
                                        Unit.Default
                                            .Tee(__ => a.Gradient += 1/b.Data * _.Gradient)
                                            .Tee(__ => b.Gradient += (-a.Data/Math.Pow(b.Data, 2)) * _.Gradient)));

    public static Operand operator /(double aval, Operand b)
        => new Operand(aval)
            .Map(a => new Operand(a.Data / b.Data, (a, b))
                .Tee(_ => _._backward = () =>
                    Unit.Default
                        .Tee(__ => a.Gradient += 1/b.Data * _.Gradient)
                        .Tee(__ => b.Gradient += (-a.Data/Math.Pow(b.Data, 2)) * _.Gradient)));

    public Operand Exp()
        => new Operand(Math.Exp(Data), (this, null))
            .Tee(_ => _._backward = () =>
                    Unit.Default
                        .Tee(__ => Gradient += _.Data * _.Gradient));

    public Operand Pow(double exponent)
        => new Operand(Math.Pow(Data, exponent), (this, null))
            .Tee(_ => _._backward = () =>
                    Unit.Default
                        .Tee(__ => Gradient += exponent * Math.Pow(Data, exponent - 1) * _.Gradient));

    public Operand Activation()
        => this
            .Map(_ => 2 * this)
            .Map(_ => _.Exp())
            .Map(_ => (_ - 1) / (_ + 1));   

    public Operand Relu()
        => new Operand(
            Data < 0 ? 0 : Data,
            (this, null))
            .Tee(_ => _._backward = () =>
                Unit.Default
                    .Tee(__ => Gradient += _.Data * _.Gradient));

    private static List<Operand> BuildTopological(List<Operand> operands, Operand item)
        => operands
            .ToOption(_ => _.Contains(item))
            .Map(_ => _.Tee(_ => item.Previous.Operand1.OnSome(previous => BuildTopological(_, previous))))
            .Map(_ => _.Tee(_ => item.Previous.Operand2.OnSome(previous => BuildTopological(_, previous))))
            .Map(_ => _.Tee(_ => _.Add(item)))
            .OrElse(operands);

    public Unit Backpropagation()
        => Unit.Default
            .Tee(_ => Gradient = ONE)
            .Tee(_ => BuildTopological([], this)
                        .Map(operands => operands.Tee(_ => operands.Reverse()))
                        .ForEach(operand => operand._backward()));
}

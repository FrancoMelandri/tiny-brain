using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata.Ecma335;
using TinyFp;
using TinyFp.Extensions;
using static TinyFp.Extensions.Functional;
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
        => a + Of(bval);

    public static Operand operator +(double aval, Operand b)
        => Of(aval) + b;

    public static Operand operator -(Operand a, Operand b)
        => new Operand(a.Data - b.Data, (a, b))
            .Tee(_ => _._backward = () =>
                Unit.Default
                    .Tee(__ => a.Gradient += GRADIENT_MINUS * _.Gradient)
                    .Tee(__ => b.Gradient += GRADIENT_MINUS * _.Gradient));

    public static Operand operator -(Operand a, double bval)
        => a - Of(bval);

    public static Operand operator -(double aval, Operand b)
        => Of(aval) - b;

    public static Operand operator *(Operand a, Operand b)
        => new Operand(a.Data * b.Data, (a, b))
            .Tee(_ => _._backward = () =>
                Unit.Default
                    .Tee(__ => a.Gradient += b.Data * _.Gradient)
                    .Tee(__ => b.Gradient += a.Data * _.Gradient));

    public static Operand operator *(Operand a, double bval)
        => a * Of(bval);

    public static Operand operator *(double aval, Operand b)
        => Of(aval) * b ;

    public static Operand operator /(Operand a, Operand b)
        => new Operand(a.Data / b.Data, (a, b))
            .Tee(_ => _._backward = () =>
                Unit.Default
                    .Tee(__ => a.Gradient += 1/b.Data * _.Gradient)
                    .Tee(__ => b.Gradient += (-a.Data/Math.Pow(b.Data, 2)) * _.Gradient));

    public static Operand operator /(Operand a, double bval)
        => a / Of(bval);

    public static Operand operator /(double aval, Operand b)
        => Of(aval) / b;

    public Operand Exp()
        => new Operand(Math.Exp(Data), (this, null))
            .Tee(_ => _._backward = () =>
                    Unit.Default
                        .Tee(__ => Gradient += _.Data * _.Gradient));

    public Operand Log()
        => new Operand(Math.Log(Data), (this, null))
            .Tee(_ => _._backward = () =>
                    Unit.Default
                        .Tee(__ => Gradient += 1/_.Data * _.Gradient));

    public Operand Pow(double exponent)
        => new Operand(Math.Pow(Data, exponent), (this, null))
            .Tee(_ => _._backward = () =>
                    Unit.Default
                        .Tee(__ => Gradient += exponent * Math.Pow(Data, exponent - 1) * _.Gradient));

    public Operand Relu()
        => new Operand(
            Data < 0 ? 0 : Data,
            (this, null))
            .Tee(_ => _._backward = () =>
                Unit.Default
                    .Tee(__ => Gradient += _.Data * _.Gradient));

    private static HashSet<Operand> BuildTopologicalRecursive(HashSet<Operand> operands, Operand item)
        => operands
            .ToOption(_ => _.Contains(item))
            .Map(_ => _.Tee(_ => item.Previous.Operand1.OnSome(operand1 => BuildTopologicalRecursive(_, operand1))))
            .Map(_ => _.Tee(_ => item.Previous.Operand2.OnSome(operand2 => BuildTopologicalRecursive(_, operand2))))
            .Map(_ => _.Tee(_ => _.Add(item)))
            .OrElse(operands);

    private static List<Operand> BuildTopologicalNonRecursive(Operand root)
        => While(() => (HashSet: new HashSet<Operand>(), Stack: new Stack<Operand>().Tee(_ => _.Push(root))),
                _ => _.Stack.Count > 0,
                _ =>
                {
                    var current = _.Stack.Peek();
        
                    if (!_.HashSet.Contains(current))
                    {
                        if (current.Previous.Operand2.IsSome)
                        {
                            var operand2 = current.Previous.Operand2.Unwrap();
                            if (!_.HashSet.Contains(operand2))
                            {
                                _.Stack.Push(operand2);
                                return _;
                            }
                        }
            
                        if (current.Previous.Operand1.IsSome)
                        {
                            var operand1 = current.Previous.Operand1.Unwrap();
                            if (!_.HashSet.Contains(operand1))
                            {
                                _.Stack.Push(operand1);
                                return _;
                            }
                        }
                    }
                    _.HashSet.Add(current);
                    _.Stack.Pop();
                    return _;
                }
                )
            .Map(_ => _.Item1.ToList());
    
    public Unit Backpropagation()
        => Unit.Default
            .Tee(_ => Gradient = ONE)
            .Tee(_ => BuildTopologicalNonRecursive(this)
                        .Map(operands => operands.Tee(_ => operands.Reverse()))
                        .ForEach(operand => operand._backward()));
}

using Shouldly;

namespace TinyBrain.Test;

public class OperandTests
{
    [Test]
    public void Token_Sum_Is_Right()
    {
        var operand1 = Operand.Of(42.0);
        var operand2 = Operand.Of(66.0);
        var token = operand1 + operand2; 
        token.Data.ShouldBe(108.0);
        token.Previous.Operand1.Unwrap().ShouldBe(operand1);
        token.Previous.Operand2.Unwrap().ShouldBe(operand2);
    }

    [Test]
    public void Token_And_Constant_Sum_Is_Right()
    {
        var operand1 = Operand.Of(42.0);
        var operand2 = operand1 + 10; 
        var token3 = 10 + operand1; 
        operand2.Data.ShouldBe(52.0);
        token3.Data.ShouldBe(52.0);
        operand2.Previous.Operand1.Unwrap().ShouldBe(operand1);
        operand2.Previous.Operand2.Unwrap().Previous.Operand1.IsNone.ShouldBe(true);
        operand2.Previous.Operand2.Unwrap().Previous.Operand2.IsNone.ShouldBe(true);
        token3.Previous.Operand2.Unwrap().ShouldBe(operand1);
        token3.Previous.Operand2.Unwrap().Previous.Operand1.IsNone.ShouldBe(true);
        token3.Previous.Operand2.Unwrap().Previous.Operand2.IsNone.ShouldBe(true);
    }
    
    [Test]
    public void Token_Sub_Is_Right()
    {
        var operand1 = Operand.Of(66.0);
        var operand2 = Operand.Of(42.0);
        var token = operand1 - operand2; 
        token.Data.ShouldBe(24.0);
        token.Previous.Operand1.Unwrap().ShouldBe(operand1);
        token.Previous.Operand2.Unwrap().ShouldBe(operand2);
    }

    [Test]
    public void Token_Sub_Constant_Sum_Is_Right()
    {
        var operand1 = Operand.Of(42.0);
        var operand2 = operand1 - 10; 
        var token3 = 10 - operand1; 
        operand2.Data.ShouldBe(32.0);
        token3.Data.ShouldBe(-32.0);
        operand2.Previous.Operand1.Unwrap().ShouldBe(operand1);
        operand2.Previous.Operand2.Unwrap().Previous.Operand1.IsNone.ShouldBe(true);
        operand2.Previous.Operand2.Unwrap().Previous.Operand2.IsNone.ShouldBe(true);
        token3.Previous.Operand2.Unwrap().ShouldBe(operand1);
        token3.Previous.Operand2.Unwrap().Previous.Operand1.IsNone.ShouldBe(true);
        token3.Previous.Operand2.Unwrap().Previous.Operand2.IsNone.ShouldBe(true);
    }

    [Test]
    public void Token_Mul_Is_Right()
    {
        var operand1 = Operand.Of(42.0);
        var operand2 = Operand.Of(2.0);
        var token = operand1 * operand2; 
        token.Data.ShouldBe(84.0);
        token.Previous.Operand1.Unwrap().ShouldBe(operand1);
        token.Previous.Operand2.Unwrap().ShouldBe(operand2);
    }

    [Test]
    public void Token_And_Constant_Mul_Is_Right()
    {
        var operand1 = Operand.Of(42.0);
        var operand2 = operand1 * 2;
        var token3 = 2 * operand1;
        operand2.Data.ShouldBe(84.0);
        token3.Data.ShouldBe(84.0);
        operand2.Previous.Operand1.Unwrap().ShouldBe(operand1);
        operand2.Previous.Operand2.Unwrap().Previous.Operand1.IsNone.ShouldBe(true);
        operand2.Previous.Operand2.Unwrap().Previous.Operand2.IsNone.ShouldBe(true);
        token3.Previous.Operand2.Unwrap().ShouldBe(operand1);
        token3.Previous.Operand2.Unwrap().Previous.Operand1.IsNone.ShouldBe(true);
        token3.Previous.Operand2.Unwrap().Previous.Operand2.IsNone.ShouldBe(true);
    }
        [Test]
    public void Token_Div_Is_Right()
    {
        var operand1 = Operand.Of(42.0);
        var operand2 = Operand.Of(2.0);
        var token = operand1 / operand2; 
        token.Data.ShouldBe(21.0);
        token.Previous.Operand1.Unwrap().ShouldBe(operand1);
        token.Previous.Operand2.Unwrap().ShouldBe(operand2);
    }

    [Test]
    public void Token_And_Constant_Div_Is_Right()
    {
        var operand1 = Operand.Of(42.0);
        var operand2 = operand1 / 2;
        var token3 = 84 / operand1;
        operand2.Data.ShouldBe(21.0);
        token3.Data.ShouldBe(2.0);
        operand2.Previous.Operand1.Unwrap().ShouldBe(operand1);
        operand2.Previous.Operand2.Unwrap().Previous.Operand1.IsNone.ShouldBe(true);
        operand2.Previous.Operand2.Unwrap().Previous.Operand2.IsNone.ShouldBe(true);
        token3.Previous.Operand2.Unwrap().ShouldBe(operand1);
        token3.Previous.Operand2.Unwrap().Previous.Operand1.IsNone.ShouldBe(true);
        token3.Previous.Operand2.Unwrap().Previous.Operand2.IsNone.ShouldBe(true);
    }
    
    [Test]
    public void Complex_Right()
    {
        var a = Operand.Of(2.0);
        var b = Operand.Of(-3.0);
        var c = Operand.Of(10);
        var e = a*b;
        var d = e + c;
        
        d.Data.ShouldBe(4.0);
        d.Previous.Operand1.Unwrap().Data.ShouldBe(-6);
        d.Previous.Operand2.Unwrap().Data.ShouldBe(10);

        d.Previous.Operand2.Unwrap().Previous.Operand1.IsNone.ShouldBe(true);
        d.Previous.Operand2.Unwrap().Previous.Operand2.IsNone.ShouldBe(true);
        
        d.Previous.Operand1.Unwrap().Previous.Operand1.Unwrap().Data.ShouldBe(2);
        d.Previous.Operand1.Unwrap().Previous.Operand2.Unwrap().Data.ShouldBe(-3);
        
        d.Previous.Operand1.Unwrap().Previous.Operand1.Unwrap().Previous.Operand1.IsNone.ShouldBe(true);
        d.Previous.Operand1.Unwrap().Previous.Operand1.Unwrap().Previous.Operand2.IsNone.ShouldBe(true);
        d.Previous.Operand1.Unwrap().Previous.Operand2.Unwrap().Previous.Operand1.IsNone.ShouldBe(true);
        d.Previous.Operand1.Unwrap().Previous.Operand2.Unwrap().Previous.Operand2.IsNone.ShouldBe(true);
    }
    
    [Test]
    public void Neuron()
    {
        // bidimensional
        // x1*w1 + x2*w2 + b
        
        // inputs
        var x1 = Operand.Of(2.0);
        var x2 =Operand.Of(0);
        
        // weights
        var w1 = Operand.Of(-3.0);
        var w2 =Operand.Of(1);
        
        // BIAS
        var b = Operand.Of(6.8813735870195432);

        var x1w1 = x1*w1;
        var x2w2 = x2*w2;

        var x1w1x2w2 = x1w1 + x2w2;
        var n = x1w1x2w2 + b;
        
        var o = n.Tanh();
        o.Backpropagation();
        
        o.Data.ShouldBe(0.7071067811865476);
        n.Gradient.ShouldBe(0.4999999999999999);
        
        x1w1x2w2.Gradient.ShouldBe(0.4999999999999999);
        b.Gradient.ShouldBe(0.4999999999999999);
        
        x1w1.Gradient.ShouldBe(0.4999999999999999);
        x2w2.Gradient.ShouldBe(0.4999999999999999);

        x2.Gradient.ShouldBe(0.4999999999999999);
        w2.Gradient.ShouldBe(0);
        x1.Gradient.ShouldBe(-1.4999999999999996);
        w1.Gradient.ShouldBe(0.9999999999999998);
    }

    [Test]
    public void Neuron_Breaking_Activation()
    {
        // bidimensional
        // x1*w1 + x2*w2 + b
        
        // inputs
        var x1 = Operand.Of(2.0, "x1");
        var x2 = Operand.Of(0, "x2");
        
        // weights
        var w1 = Operand.Of(-3.0, "w1");
        var w2 =Operand.Of(1, "w2");
        
        // BIAS
        var b = Operand.Of(6.8813735870195432, "bias");
        
        var x1w1 = x1*w1;
        x1w1.Label = "x1w1";
        var x2w2 = x2*w2;
        x2w2.Label = "x2w2";

        var x1w1x2w2 = x1w1 + x2w2;
        x1w1x2w2.Label = "x1w1x2w2";
        var n = x1w1x2w2 + b;
        n.Label = "n";
        var n2 = n * 2;
        n2.Label = "n2";
        var e = n2.Exp();
        e.Label = "e";
        var e1 = e - 1;
        e1.Label = "e1";
        var e2 = e + 1;
        e2.Label = "e2";
        
        var o = e1/e2;
        o.Label = "o";
        
        
        o.Gradient = 1.0;
        o.Backpropagation();
        o.Data.ShouldBe(0.7071067811865476);
        n.Gradient.ShouldBe(0.4999999999999999);
        
        b.Gradient.ShouldBe(0.4999999999999999);
        x1w1x2w2.Gradient.ShouldBe(0.4999999999999999);
        
        x1w1.Gradient.ShouldBe(0.4999999999999999);
        x2w2.Gradient.ShouldBe(0.4999999999999999);

        x2.Gradient.ShouldBe(0.4999999999999999);
        w2.Gradient.ShouldBe(0);
        x1.Gradient.ShouldBe(-1.4999999999999996);
        w1.Gradient.ShouldBe(0.9999999999999998);
    }

    [Test]
    public void Token_Exp_Is_Right()
    {
        var operand1 = Operand.Of(2.0);
        var operand2 = operand1.Exp();
        operand2.Data.ShouldBe(7.38905609893065);
        operand2.Previous.Operand1.Unwrap().ShouldBe(operand1);
        operand2.Previous.Operand2.IsNone.ShouldBe(true);
    }

    [Test]
    public void Token_Pow_Is_Right()
    {
        var operand1 = Operand.Of(2.0);
        var operand2 = operand1.Pow(3);
        
        operand2.Data.ShouldBe(8);
        operand2.Previous.Operand1.Unwrap().ShouldBe(operand1);
        operand2.Previous.Operand2.IsNone.ShouldBe(true);
    }
}
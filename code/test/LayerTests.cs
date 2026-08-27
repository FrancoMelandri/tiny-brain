using Shouldly;

namespace TinyBrain.Test;

public class LayerTests
{
    [Test]
    public void Layer_Params()
    {
        var layer = new Layer(4, 3, ActivationType.Tanh);

        layer.Weights.Rows.ShouldBe(4);
        layer.Weights.Cols.ShouldBe(3);
        layer.Bias.Rows.ShouldBe(1);
        layer.Bias.Cols.ShouldBe(3);
        layer.ParameterMatrices.Length.ShouldBe(2);
    }

    [Test]
    public void Layer_Forward_Tanh_Output_In_Range()
    {
        var layer = new Layer(4, 3, ActivationType.Tanh);
        var input = Operand.Of(new float[,] { { 1.0f, 2.0f, 3.0f, 4.0f } });
        var output = layer.Forward(input);

        output.Rows.ShouldBe(1);
        output.Cols.ShouldBe(3);
        for (var j = 0; j < 3; j++)
        {
            output.Data[j].ShouldBeGreaterThan(-1.0f);
            output.Data[j].ShouldBeLessThan(1.0f);
        }
    }

    [Test]
    public void Layer_Forward_None_Unconstrained()
    {
        var layer = new Layer(2, 3, ActivationType.None);
        var input = Operand.Of(new float[,] { { 10.0f, 10.0f } });
        var output = layer.Forward(input);

        output.Rows.ShouldBe(1);
        output.Cols.ShouldBe(3);
    }
}

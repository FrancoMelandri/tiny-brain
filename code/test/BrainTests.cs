using Shouldly;

namespace TinyBrain.Test;

public class BrainTests
{
    [Test]
    public void Brain_Topology()
    {
        var brain = new Brain("test", 3, [4, 4, 1]);

        brain.Layers.Length.ShouldBe(3);
        brain.Layers[0].Weights.Rows.ShouldBe(3);
        brain.Layers[0].Weights.Cols.ShouldBe(4);
        brain.Layers[1].Weights.Rows.ShouldBe(4);
        brain.Layers[1].Weights.Cols.ShouldBe(4);
        brain.Layers[2].Weights.Rows.ShouldBe(4);
        brain.Layers[2].Weights.Cols.ShouldBe(1);
        brain.ParameterMatrices.Length.ShouldBe(6); // weights + bias per layer × 3 layers
    }

    [Test]
    public void Brain_Forward_Output_Shape()
    {
        var brain = new Brain("test", 3, [4, 4, 1]);
        var input = Operand.Of(new double[,] { { 1.0, 2.0, 3.0 } });
        var output = brain.Forward(input);

        output.Rows.ShouldBe(1);
        output.Cols.ShouldBe(1);
    }
}

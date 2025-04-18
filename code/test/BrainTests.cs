using Shouldly;

namespace TinyBrain.Test;

public class BrainTests
{
    [Test]
    public void MPL_Ok()
    {
        var mpl = new Brain("test", 4, [3, 4, 1]);
        var o = mpl.Forward([Operand.Of(1), Operand.Of(2), Operand.Of(3), Operand.Of(4)]);
        
        mpl.Layers.Length.ShouldBe(3);
        mpl.Layers[0].Neurons.Length.ShouldBe(3);
        mpl.Layers[0].Neurons[0].Weights.Length.ShouldBe(4);
        
        mpl.Layers[1].Neurons.Length.ShouldBe(4);
        mpl.Layers[1].Neurons[0].Weights.Length.ShouldBe(3);

        mpl.Layers[2].Neurons.Length.ShouldBe(1);
        mpl.Layers[2].Neurons[0].Weights.Length.ShouldBe(4);
        
        o.Length.ShouldBe(1);
        o[0].Data.ShouldBeLessThanOrEqualTo(1);
        o[0].Data.ShouldBeGreaterThanOrEqualTo(-1);
        
        mpl.Parameters.Length.ShouldBe(36);
    }
}
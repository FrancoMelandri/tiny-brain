using System.Linq;

namespace TinyBrain;

public class Brain
{
    public string Id { get; }
    public Layer[] Layers { get; }

    public Brain(string id, int inputSize, int[] layerSizes, ActivationType activationType = ActivationType.Tanh)
    {
        Id = id;
        int[] sizes = [inputSize, ..layerSizes];
        Layers = new Layer[sizes.Length - 1];
        for (var i = 0; i < Layers.Length; i++)
            Layers[i] = new Layer(sizes[i], sizes[i + 1], activationType);
    }

    public Operand[] ParameterMatrices => Layers.SelectMany(l => l.ParameterMatrices).ToArray();

    public Operand Forward(Operand input)
    {
        foreach (var layer in Layers)
            layer.ZeroGradients();

        var current = input;
        foreach (var layer in Layers)
            current = layer.Forward(current);
        return current;
    }
}

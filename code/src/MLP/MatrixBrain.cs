using System.Linq;

namespace TinyBrain;

public class MatrixBrain
{
    public string Id { get; }
    public MatrixLayer[] Layers { get; }

    public MatrixBrain(string id, int inputSize, int[] layerSizes, ActivationType activationType)
    {
        Id = id;
        int[] sizes = [inputSize, ..layerSizes];
        Layers = new MatrixLayer[sizes.Length - 1];
        for (var i = 0; i < Layers.Length; i++)
            Layers[i] = new MatrixLayer(sizes[i], sizes[i + 1], activationType);
    }

    public MatrixOperand[] ParameterMatrices => Layers.SelectMany(l => l.ParameterMatrices).ToArray();

    public MatrixOperand Forward(MatrixOperand input)
    {
        foreach (var layer in Layers)
            layer.ZeroGradients();

        var current = input;
        foreach (var layer in Layers)
            current = layer.Forward(current);
        return current;
    }
}

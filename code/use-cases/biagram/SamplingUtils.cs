using System;
using TinyBrain;

public static class SamplingUtils
{
    public static int Multinomial(float[] probabilities)
    {
        var total = 0.0f;
        for (var i = 0; i < probabilities.Length; i++) total += probabilities[i];
        var r = (float)Random.Shared.NextDouble() * total;
        var cumulative = 0.0f;
        for (var i = 0; i < probabilities.Length; i++)
        {
            cumulative += probabilities[i];
            if (r < cumulative) return i;
        }
        return probabilities.Length - 1;
    }

    // Returns Operand [N, numClasses] — each row is a one-hot vector
    public static Operand OneHotMatrix(int[] indices, int numClasses)
    {
        var data = new float[indices.Length, numClasses];
        for (var i = 0; i < indices.Length; i++)
            data[i, indices[i]] = 1.0f;
        return Operand.Of(data);
    }
}

using System;
using System.Linq;
using TinyBrain;
using TinyFp.Extensions; // Required for Sum() and Any()

public static class SamplingUtils
{
    /// <summary>
    /// Draws a single sample index from a multinomial distribution defined by input probabilities/weights.
    /// This behaves like torch.multinomial(input, num_samples=1).
    /// </summary>
    /// <param name="probabilities">
    /// An array of non-negative doubles representing probabilities or unnormalized weights.
    /// They do not need to sum to 1, but must not be negative.
    /// </param>
    /// <returns>The 0-based index of the sampled category.</returns>
    /// <exception cref="ArgumentNullException">Thrown if probabilities is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown if probabilities is empty, contains negative values, or if all values sum to zero or less.
    /// </exception>
    public static int Multinomial(double[] probabilities)
    {
        // --- Input Validation ---
        if (probabilities == null)
        {
            throw new ArgumentNullException(nameof(probabilities));
        }
        if (probabilities.Length == 0)
        {
            throw new ArgumentException("Probabilities array cannot be empty.", nameof(probabilities));
        }
        // Check for any negative probabilities/weights
        if (probabilities.Any(p => p < 0))
        {
            throw new ArgumentException("Probabilities cannot contain negative values.", nameof(probabilities));
        }

        // Calculate the total sum of probabilities/weights
        double totalSum = probabilities.Sum();

        // If the sum is zero or less (e.g., all inputs are 0), sampling is not possible.
        if (totalSum <= 0)
        {
             // Optional: Check if only one element is positive, if so return its index directly
             int nonZeroCount = 0;
             int lastNonZeroIndex = -1;
             for(int i = 0; i < probabilities.Length; ++i)
             {
                 if(probabilities[i] > 0) // Check strictly positive
                 {
                     nonZeroCount++;
                     lastNonZeroIndex = i;
                 }
             }
             // If exactly one probability is positive, that must be the one selected.
             if(nonZeroCount == 1) {
                // This check should ideally happen before the totalSum <= 0 check
                // but we recalculate it here for clarity if sum was exactly 0 due to precision
                // or if we want to handle the [0, 5, 0] case explicitly.
                // Let's reorder for better logic flow.
                 // Re-evaluate the logic:
                 // If totalSum is positive, we proceed normally.
                 // If totalSum is zero or negative (only possible if all are zero due to prior check), throw.
                throw new ArgumentException("The sum of probabilities must be positive to sample.", nameof(probabilities));

             }
             // If we reach here and totalSum > 0, continue to sampling.
             // Let's restructure validation:
        }

        // --- Restructured Validation and Edge Case Handling ---
         double calculatedTotalSum = 0.0;
         int positiveProbCount = 0;
         int lastPositiveIndex = -1;

         for(int i = 0; i < probabilities.Length; ++i)
         {
             double prob = probabilities[i];
             if (prob < 0) throw new ArgumentException("Probabilities cannot contain negative values.", nameof(probabilities)); // Redundant check, but safe
             if (prob > 0)
             {
                 positiveProbCount++;
                 lastPositiveIndex = i;
             }
             calculatedTotalSum += prob;
         }

         if (calculatedTotalSum <= 0)
         {
              throw new ArgumentException("The sum of probabilities must be positive to sample.", nameof(probabilities));
         }

         // Optimization: If only one probability is non-zero, return its index immediately.
         if (positiveProbCount == 1)
         {
             return lastPositiveIndex;
         }


        // --- Cumulative Probability Calculation ---
        // Create an array to store the cumulative sum
        double[] cumulativeProbabilities = new double[probabilities.Length];
        double currentCumulativeSum = 0.0;
        for (int i = 0; i < probabilities.Length; i++)
        {
            currentCumulativeSum += probabilities[i];
            cumulativeProbabilities[i] = currentCumulativeSum;
        }

        // Ensure the total sum used for random generation is the final cumulative sum
        // This helps mitigate potential floating-point inaccuracies compared to using Sum() directly.
        double finalCumulativeSum = cumulativeProbabilities[probabilities.Length - 1];

        // --- Sampling ---
        // Generate a random double between 0.0 (inclusive) and finalCumulativeSum (exclusive)
        // Use Random.Shared for better performance and thread safety in newer .NET versions.
        // If using older .NET Framework, you might need to manage a static Random instance carefully.
        double randomValue = Random.Shared.NextDouble() * finalCumulativeSum;

        // Find the index where the random value falls within the cumulative range
        // This performs a linear search. For very large arrays, a binary search could be faster.
        for (int i = 0; i < cumulativeProbabilities.Length; i++)
        {
            // If the random value is less than the cumulative probability at this index,
            // it means the sample falls into the range corresponding to this index.
            if (randomValue < cumulativeProbabilities[i])
            {
                return i;
            }
        }

        // --- Fallback ---
        // This part should theoretically not be reached if finalCumulativeSum > 0.
        // However, due to floating-point precision, if randomValue ends up being exactly
        // equal to finalCumulativeSum, the loop might finish. In this case,
        // returning the last index is the logical choice.
        return probabilities.Length - 1;
    }
    
    /// <summary>
    /// Converts an array of class indices into their one-hot encoded representation.
    /// Mimics the behavior of torch.nn.functional.one_hot.
    /// </summary>
    /// <param name="indices">An array of non-negative integer class indices.</param>
    /// <param name="numClasses">The total number of classes. This determines the length of each one-hot vector.
    /// It must be greater than the maximum value in indices.</param>
    /// <returns>A jagged array (array of arrays), where each inner array is the one-hot representation
    /// of the corresponding input index. The inner arrays will have a length equal to numClasses.</returns>
    /// <exception cref="ArgumentNullException">Thrown if indices is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown if numClasses is not positive, or if any index in indices is less than 0 or greater than or equal to numClasses.
    /// </exception>
    public static Operand[][] OneHot(int[] indices, int numClasses)
    {
        // --- Input Validation ---
        if (indices == null)
        {
            throw new ArgumentNullException(nameof(indices));
        }
        if (numClasses <= 0)
        {
            // The number of classes must be at least 1.
            throw new ArgumentOutOfRangeException(nameof(numClasses), "Number of classes must be positive.");
        }

        // Validate each index against the number of classes
        for (int i = 0; i < indices.Length; i++)
        {
            int index = indices[i];
            if (index < 0 || index >= numClasses)
            {
                throw new ArgumentOutOfRangeException(nameof(indices),
                    $"Input index at position {i} ({index}) is out of the valid range [0, {numClasses - 1}].");
            }
        }

        // --- One-Hot Encoding ---

        // Create the outer array (jagged array) to hold the results.
        // Its length is the same as the number of input indices.
        Operand[][] oneHotResult = new Operand[indices.Length][];

        // Iterate through each input index
        for (int i = 0; i < indices.Length; i++)
        {
            int currentIndex = indices[i];

            // Create the inner array (the one-hot vector) for the current index.
            // Its length is determined by numClasses.
            // C# automatically initializes integer arrays with all zeros.
            Operand[] oneHotVector = new Operand[numClasses].Select(_ => Operand.Of(0.0)).ToArray();
            // Set the element at the position specified by the currentIndex to 1.
            oneHotVector[currentIndex] = Operand.Of(1.0);

            // Assign the completed one-hot vector to the result array.
            oneHotResult[i] = oneHotVector;
        }

        return oneHotResult;
    }
}
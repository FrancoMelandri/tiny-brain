using System;
using System.Collections.Generic;
using System.Linq;

namespace slm;

public class Tokenizer
{
    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<int, string> _indexToWord;

    public const int UnkIdx = 0;
    public const int BosIdx = 1;
    public const int EosIdx = 2;

    public Tokenizer(string text, int maxVocabSize = int.MaxValue)
    {
        _vocab = new Dictionary<string, int>
        {
            ["<UNK>"] = UnkIdx,
            ["<BOS>"] = BosIdx,
            ["<EOS>"] = EosIdx
        };
        _indexToWord = new Dictionary<int, string>
        {
            [UnkIdx] = "<UNK>",
            [BosIdx] = "<BOS>",
            [EosIdx] = "<EOS>"
        };

        var words = SplitWords(text)
            .GroupBy(w => w)
            .OrderByDescending(g => g.Count())
            .Take(maxVocabSize - 3)
            .Select(g => g.Key);

        foreach (var word in words)
        {
            var idx = _vocab.Count;
            _vocab[word] = idx;
            _indexToWord[idx] = word;
        }
    }

    public int VocabSize => _vocab.Count;

    public int[] Encode(string text)
        => SplitWords(text)
            .Select(w => _vocab.TryGetValue(w, out var idx) ? idx : UnkIdx)
            .ToArray();

    public string Decode(int[] indices)
        => string.Join(" ", indices
            .Where(i => i != BosIdx && i != EosIdx)
            .Select(i => _indexToWord.TryGetValue(i, out var w) ? w : "<UNK>"));

    public static IEnumerable<string> SplitWords(string text)
        => text
            .Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries)
            .Select(w => w.Trim('.', ',', '!', '?', ';', ':', '"', '\'', '(', ')', '-').ToLower())
            .Where(w => w.Length > 0);
}

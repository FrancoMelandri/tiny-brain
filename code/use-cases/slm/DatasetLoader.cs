using System.IO;
using System.Text;

namespace slm;

public static class DatasetLoader
{
    public static string LoadText(string path, int maxStories, int skipStories = 0)
    {
        var sb = new StringBuilder();
        var count = 0;
        using var reader = new StreamReader(path);
        reader.ReadLine(); // skip header
        for (var s = 0; s < skipStories && !reader.EndOfStream; s++)
        {
            var skipped = ReadQuotedField(reader);
            if (skipped.Length == 0) break;
        }
        while (count < maxStories && !reader.EndOfStream)
        {
            var story = ReadQuotedField(reader);
            if (story.Length == 0) break;
            sb.Append(story).Append(' ');
            count++;
        }
        return sb.ToString();
    }

    public static int CountStories(string path)
    {
        var count = 0;
        using var reader = new StreamReader(path);
        reader.ReadLine(); // skip header
        while (!reader.EndOfStream)
        {
            var story = ReadQuotedField(reader);
            if (story.Length == 0) break;
            count++;
        }
        return count;
    }

    private static string ReadQuotedField(StreamReader reader)
    {
        int ch;
        while ((ch = reader.Read()) != -1 && ch != '"') { }
        if (ch == -1) return string.Empty;

        var sb = new StringBuilder();
        while ((ch = reader.Read()) != -1)
        {
            if (ch == '"')
            {
                if (reader.Peek() == '"')
                {
                    reader.Read();
                    sb.Append('"');
                }
                else break;
            }
            else
                sb.Append((char)ch);
        }
        return sb.ToString();
    }
}

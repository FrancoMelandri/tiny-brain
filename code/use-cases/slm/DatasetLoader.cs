using System.IO;
using System.Text;

namespace slm;

public static class DatasetLoader
{
    public static string LoadText(string path, int maxStories)
    {
        var sb = new StringBuilder();
        var count = 0;
        using var reader = new StreamReader(path);
        reader.ReadLine(); // skip header
        while (count < maxStories && !reader.EndOfStream)
        {
            var story = ReadQuotedField(reader);
            if (story.Length == 0) break;
            sb.Append(story).Append(' ');
            count++;
        }
        return sb.ToString();
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

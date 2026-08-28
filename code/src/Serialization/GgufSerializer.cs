using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace TinyBrain;

// GGUF v3 binary serializer for float32 tensors.
//
// Layout:
//   [Header]      magic(4) + version(u32) + tensorCount(u64) + kvCount(u64)
//   [KV pairs]    key(str) + type(u32=8/STRING) + value(str)
//   [Tensor info] name(str) + nDims(u32=2) + dims(u64[2]) + type(u32=0/F32) + offset(u64)
//   [Padding]     to next 32-byte boundary
//   [Tensor data] float[] row-major, each tensor padded to 32 bytes
public static class GgufSerializer
{
    private static readonly byte[] Magic = Encoding.ASCII.GetBytes("GGUF");
    private const uint Version = 3;
    private const uint GgmlTypeF32 = 0;
    private const uint KvTypeString = 8;
    private const uint KvTypeUint64 = 10;
    private const int Alignment = 32;

    public static void Write(string path,
                             string modelName,
                             IEnumerable<(string name, Operand tensor)> tensors,
                             IEnumerable<(string key, string value)>? stringKv = null,
                             IEnumerable<(string key, ulong value)>? uint64Kv = null)
    {
        var tensorList = tensors.ToList();
        var extraStrKv  = stringKv?.ToList() ?? [];
        var extraU64Kv  = uint64Kv?.ToList() ?? [];

        // Compute data section offsets
        var offsets = new ulong[tensorList.Count];
        ulong dataOffset = 0;
        for (var i = 0; i < tensorList.Count; i++)
        {
            offsets[i] = dataOffset;
            var byteLen = (ulong)(tensorList[i].tensor.Data.Length * sizeof(float));
            dataOffset += byteLen;
            dataOffset = Pad(dataOffset, Alignment);
        }

        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write);
        using var w = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: false);

        // Header
        w.Write(Magic);
        w.Write(Version);
        w.Write((ulong)tensorList.Count);

        // KV count: 2 built-in string pairs + caller-supplied extras
        var builtIn = new (string key, string value)[]
        {
            ("general.architecture", modelName),
            ("general.name",         modelName),
        };
        w.Write((ulong)(builtIn.Length + extraStrKv.Count + extraU64Kv.Count));

        foreach (var (key, value) in builtIn)
        {
            WriteString(w, key);
            w.Write(KvTypeString);
            WriteString(w, value);
        }
        foreach (var (key, value) in extraStrKv)
        {
            WriteString(w, key);
            w.Write(KvTypeString);
            WriteString(w, value);
        }
        foreach (var (key, value) in extraU64Kv)
        {
            WriteString(w, key);
            w.Write(KvTypeUint64);
            w.Write(value);
        }

        // Tensor info
        for (var i = 0; i < tensorList.Count; i++)
        {
            var (name, tensor) = tensorList[i];
            WriteString(w, name);
            w.Write((uint)2);                         // n_dims = 2
            w.Write((ulong)tensor.Rows);
            w.Write((ulong)tensor.Cols);
            w.Write(GgmlTypeF32);
            w.Write(offsets[i]);
        }

        // Padding to align tensor data section
        var headerEnd = stream.Position;
        var paddedStart = (long)Pad((ulong)headerEnd, Alignment);
        if (paddedStart > headerEnd)
            w.Write(new byte[paddedStart - headerEnd]);

        // Tensor data
        var floatBuf = new byte[sizeof(float)];
        foreach (var (_, tensor) in tensorList)
        {
            var dataLen = tensor.Data.Length * sizeof(float);
            var buf = new byte[dataLen];
            Buffer.BlockCopy(tensor.Data, 0, buf, 0, dataLen);
            w.Write(buf);

            // Pad to alignment
            var padBytes = (int)(Pad((ulong)dataLen, Alignment) - (ulong)dataLen);
            if (padBytes > 0)
                w.Write(new byte[padBytes]);
        }
    }

    public static IReadOnlyList<(string name, int rows, int cols, float[] data)> Read(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read);
        using var r = new BinaryReader(stream, Encoding.UTF8, leaveOpen: false);

        // Magic
        var magic = r.ReadBytes(4);
        if (!magic.SequenceEqual(Magic))
            throw new InvalidDataException("Not a GGUF file.");

        var version = r.ReadUInt32();
        if (version != Version)
            throw new InvalidDataException($"Unsupported GGUF version {version}; expected {Version}.");

        var tensorCount = (int)r.ReadUInt64();
        var kvCount = (int)r.ReadUInt64();

        // Skip KV pairs
        for (var i = 0; i < kvCount; i++)
        {
            ReadString(r);              // key
            var type = r.ReadUInt32();
            SkipKvValue(r, type);       // value
        }

        // Read tensor info
        var infos = new (string name, int rows, int cols, ulong offset)[tensorCount];
        for (var i = 0; i < tensorCount; i++)
        {
            var name = ReadString(r);
            var nDims = r.ReadUInt32();
            var dims = new ulong[nDims];
            for (var d = 0; d < nDims; d++)
                dims[d] = r.ReadUInt64();
            r.ReadUInt32(); // type (we always wrote F32)
            var offset = r.ReadUInt64();

            var rows = nDims >= 1 ? (int)dims[0] : 1;
            var cols = nDims >= 2 ? (int)dims[1] : 1;
            infos[i] = (name, rows, cols, offset);
        }

        // Advance to tensor data section (aligned)
        var headerEnd = stream.Position;
        var dataStart = (long)Pad((ulong)headerEnd, Alignment);
        stream.Seek(dataStart, SeekOrigin.Begin);

        // Read tensor data using stored offsets
        var result = new List<(string, int, int, float[])>();
        foreach (var (name, rows, cols, offset) in infos)
        {
            stream.Seek(dataStart + (long)offset, SeekOrigin.Begin);
            var count = rows * cols;
            var floatData = new float[count];
            var buf = r.ReadBytes(count * sizeof(float));
            Buffer.BlockCopy(buf, 0, floatData, 0, buf.Length);
            result.Add((name, rows, cols, floatData));
        }

        return result;
    }

    public static (
        IReadOnlyList<(string name, int rows, int cols, float[] data)> Tensors,
        Dictionary<string, string> StringKv,
        Dictionary<string, ulong> Uint64Kv
    ) ReadWithMetadata(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read);
        using var r = new BinaryReader(stream, Encoding.UTF8, leaveOpen: false);

        var magic = r.ReadBytes(4);
        if (!magic.SequenceEqual(Magic))
            throw new InvalidDataException("Not a GGUF file.");

        var version = r.ReadUInt32();
        if (version != Version)
            throw new InvalidDataException($"Unsupported GGUF version {version}; expected {Version}.");

        var tensorCount = (int)r.ReadUInt64();
        var kvCount     = (int)r.ReadUInt64();

        var strKv  = new Dictionary<string, string>();
        var u64Kv  = new Dictionary<string, ulong>();

        for (var i = 0; i < kvCount; i++)
        {
            var key  = ReadString(r);
            var type = r.ReadUInt32();
            if (type == KvTypeString)
                strKv[key] = ReadString(r);
            else if (type == KvTypeUint64)
                u64Kv[key] = r.ReadUInt64();
            else
                SkipKvValue(r, type);
        }

        var infos = new (string name, int rows, int cols, ulong offset)[tensorCount];
        for (var i = 0; i < tensorCount; i++)
        {
            var name  = ReadString(r);
            var nDims = r.ReadUInt32();
            var dims  = new ulong[nDims];
            for (var d = 0; d < nDims; d++)
                dims[d] = r.ReadUInt64();
            r.ReadUInt32();
            var offset = r.ReadUInt64();
            var rows   = nDims >= 1 ? (int)dims[0] : 1;
            var cols   = nDims >= 2 ? (int)dims[1] : 1;
            infos[i]   = (name, rows, cols, offset);
        }

        var headerEnd  = stream.Position;
        var dataStart  = (long)Pad((ulong)headerEnd, Alignment);
        stream.Seek(dataStart, SeekOrigin.Begin);

        var tensors = new List<(string, int, int, float[])>();
        foreach (var (name, rows, cols, offset) in infos)
        {
            stream.Seek(dataStart + (long)offset, SeekOrigin.Begin);
            var count     = rows * cols;
            var floatData = new float[count];
            var buf       = r.ReadBytes(count * sizeof(float));
            Buffer.BlockCopy(buf, 0, floatData, 0, buf.Length);
            tensors.Add((name, rows, cols, floatData));
        }

        return (tensors, strKv, u64Kv);
    }

    private static void WriteString(BinaryWriter w, string s)
    {
        var bytes = Encoding.UTF8.GetBytes(s);
        w.Write((ulong)bytes.Length);
        w.Write(bytes);
    }

    private static string ReadString(BinaryReader r)
    {
        var len = (int)r.ReadUInt64();
        var bytes = r.ReadBytes(len);
        return Encoding.UTF8.GetString(bytes);
    }

    private static void SkipKvValue(BinaryReader r, uint type)
    {
        switch (type)
        {
            case 0: r.ReadByte(); break;           // UINT8
            case 1: r.ReadSByte(); break;          // INT8
            case 2: r.ReadUInt16(); break;         // UINT16
            case 3: r.ReadInt16(); break;          // INT16
            case 4: r.ReadUInt32(); break;         // UINT32
            case 5: r.ReadInt32(); break;          // INT32
            case 6: r.ReadSingle(); break;         // FLOAT32
            case 7: r.ReadByte(); break;           // BOOL
            case 8: ReadString(r); break;          // STRING
            case 10: r.ReadUInt64(); break;        // UINT64
            case 11: r.ReadInt64(); break;         // INT64
            case 12: r.ReadDouble(); break;        // FLOAT64
            case 9:                                // ARRAY
                var elemType = r.ReadUInt32();
                var count = (int)r.ReadUInt64();
                for (var i = 0; i < count; i++)
                    SkipKvValue(r, elemType);
                break;
            default:
                throw new InvalidDataException($"Unknown KV value type {type}.");
        }
    }

    private static ulong Pad(ulong value, int alignment)
    {
        var a = (ulong)alignment;
        return (value + a - 1) / a * a;
    }
}

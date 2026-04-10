using System.Buffers.Binary;
using System.Net.Sockets;

namespace TranscriptionClient;

/// <summary>
/// Wire protocol shared with the Python server.
///
/// Frame layout:  [type: 1 byte] [length: 4 bytes big-endian] [payload: N bytes]
///
/// MSG_AUDIO payload layout:
///   [client_send_time_ms: 8 bytes big-endian int64] [raw PCM int16 bytes…]
/// </summary>
internal static class Protocol
{
    public const int DefaultPort = 43007;

    public const byte MsgConfig        = 1;
    public const byte MsgAudio         = 2;
    public const byte MsgTranscription = 3;
    public const byte MsgStats         = 4;
    public const byte MsgError         = 5;

    /// <summary>Size of the timestamp prefix prepended to every audio payload.</summary>
    public const int AudioTimestampSize = 8;

    private const int HeaderSize = 5; // 1 (type) + 4 (length)

    // ?? Audio payload helpers ?????????????????????????????????????????????????

    /// <summary>
    /// Prepend the current UTC millisecond timestamp to the PCM bytes so the
    /// server can measure network round-trip lag.
    /// </summary>
    public static byte[] PackAudioPayload(byte[] pcm)
    {
        long nowMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        var result = new byte[AudioTimestampSize + pcm.Length];
        BinaryPrimitives.WriteInt64BigEndian(result.AsSpan(0, AudioTimestampSize), nowMs);
        pcm.CopyTo(result, AudioTimestampSize);
        return result;
    }

    // ?? Send ??????????????????????????????????????????????????????????????????

    public static async Task SendMessageAsync(NetworkStream stream, byte msgType, byte[] payload, CancellationToken ct)
    {
        var header = new byte[HeaderSize];
        header[0] = msgType;
        BinaryPrimitives.WriteUInt32BigEndian(header.AsSpan(1), (uint)payload.Length);

        await stream.WriteAsync(header, ct);
        if (payload.Length > 0)
            await stream.WriteAsync(payload, ct);
    }

    // ?? Receive ???????????????????????????????????????????????????????????????

    /// <returns>
    ///   (msgType, payload) on success; (0, null) on clean disconnect;
    ///   throws <see cref="OperationCanceledException"/> when <paramref name="ct"/> is cancelled.
    /// </returns>
    public static async Task<(byte msgType, byte[]? payload)> RecvMessageAsync(NetworkStream stream, CancellationToken ct)
    {
        var header = await RecvExactlyAsync(stream, HeaderSize, ct);
        if (header is null)
            return (0, null);

        byte msgType = header[0];
        uint length = BinaryPrimitives.ReadUInt32BigEndian(header.AsSpan(1));

        if (length == 0)
            return (msgType, Array.Empty<byte>());

        var payload = await RecvExactlyAsync(stream, (int)length, ct);
        if (payload is null)
            return (0, null);

        return (msgType, payload);
    }

    // ?? Helpers ???????????????????????????????????????????????????????????????

    private static async Task<byte[]?> RecvExactlyAsync(NetworkStream stream, int count, CancellationToken ct)
    {
        var buffer = new byte[count];
        int received = 0;
        while (received < count)
        {
            int n = await stream.ReadAsync(buffer.AsMemory(received, count - received), ct);
            if (n == 0)
                return null; // clean disconnect
            received += n;
        }
        return buffer;
    }
}

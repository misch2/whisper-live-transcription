using NAudio.Wave;
using System.IO;
using System.Net.Sockets;
using System.Text.Json;

namespace TranscriptionClient;

/// <summary>
/// Captures audio from a WaveIn device, streams PCM frames to the transcription
/// server, and raises events with incoming transcription results.
/// Mirrors the logic of transcription_client.py.
/// </summary>
internal sealed class TranscriptionService : IDisposable
{
    // ?? Audio constants (must match server expectations) ??????????????????????
    private const int SampleRate = 16_000;
    private const int Channels = 1;
    private const int BitsPerSample = 16;
    private const int StepInSec = 1;          // seconds of audio per network message
    private const int ChunkBytes = SampleRate * Channels * (BitsPerSample / 8) * StepInSec;

    // ?? Events ????????????????????????????????????????????????????????????????

    /// <summary>Raised on the calling (UI) thread context with each live/final transcription.</summary>
    public event Action<string, bool>? TranscriptionReceived; // (text, isFinal)

    /// <summary>Raised on the calling (UI) thread context with each stats update from the server.</summary>
    public event Action<LagStats>? StatsReceived;

    /// <summary>Raised when the service stops, carrying an optional reason string.</summary>
    public event Action<string?>? Stopped;

    // ?? State ?????????????????????????????????????????????????????????????????
    private TcpClient? _tcpClient;
    private NetworkStream? _stream;
    private WaveInEvent? _waveIn;
    private CancellationTokenSource? _cts;

    private readonly object _audioLock = new();
    private readonly List<byte> _audioBuffer = new();
    private bool _disposed;
    private readonly SynchronizationContext _syncCtx;

    public TranscriptionService()
    {
        _syncCtx = SynchronizationContext.Current
                   ?? new SynchronizationContext();
    }

    // ?? Public API ????????????????????????????????????????????????????????????

    /// <summary>
    /// Connect to the server, configure the session and start capturing audio.
    /// Returns immediately; all I/O runs on background threads / tasks.
    /// </summary>
    public async Task StartAsync(string host, int port, int deviceNumber,
                                  string model = "turbo", string language = "en")
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_tcpClient is not null)
            throw new InvalidOperationException("Already running.");

        _cts = new CancellationTokenSource();
        var ct = _cts.Token;

        // ?? Connect ???????????????????????????????????????????????????????????
        _tcpClient = new TcpClient();
        await _tcpClient.ConnectAsync(host, port, ct);
        _stream = _tcpClient.GetStream();

        // ?? Send config ???????????????????????????????????????????????????????
        var config = new
        {
            sample_rate = SampleRate,
            channels = Channels,
            step_in_sec = StepInSec,
            model,
            language,
        };
        var configJson = JsonSerializer.SerializeToUtf8Bytes(config);
        await Protocol.SendMessageAsync(_stream, Protocol.MsgConfig, configJson, ct);

        // ?? Start audio capture ???????????????????????????????????????????????
        _waveIn = new WaveInEvent
        {
            DeviceNumber = deviceNumber,
            WaveFormat = new WaveFormat(SampleRate, BitsPerSample, Channels),
            BufferMilliseconds = StepInSec * 1000,
        };
        _waveIn.DataAvailable += OnDataAvailable;
        _waveIn.RecordingStopped += OnRecordingStopped;
        _waveIn.StartRecording();

        // ?? Start receiver loop ???????????????????????????????????????????????
        _ = Task.Run(() => ReceiveLoopAsync(ct), ct);

        // ?? Start sender loop ?????????????????????????????????????????????????
        _ = Task.Run(() => SendLoopAsync(ct), ct);
    }

    public void Stop()
    {
        _cts?.Cancel();
        _waveIn?.StopRecording();
        _stream?.Close();
        _tcpClient?.Close();
    }

    // ?? Audio capture callbacks ???????????????????????????????????????????????

    private void OnDataAvailable(object? sender, WaveInEventArgs e)
    {
        if (e.BytesRecorded == 0) return;
        lock (_audioLock)
        {
            _audioBuffer.AddRange(e.Buffer.AsSpan(0, e.BytesRecorded).ToArray());
        }
    }

    private void OnRecordingStopped(object? sender, StoppedEventArgs e) { /* handled via CTS */ }

    // ?? Sender loop ???????????????????????????????????????????????????????????

    private async Task SendLoopAsync(CancellationToken ct)
    {
        try
        {
            while (!ct.IsCancellationRequested)
            {
                await Task.Delay(StepInSec * 1000, ct);

                byte[]? chunk = null;
                lock (_audioLock)
                {
                    if (_audioBuffer.Count >= ChunkBytes)
                    {
                        chunk = _audioBuffer.GetRange(0, ChunkBytes).ToArray();
                        _audioBuffer.RemoveRange(0, ChunkBytes);
                    }
                    else if (_audioBuffer.Count > 0)
                    {
                        // Pad with silence if we have less than a full chunk
                        var padded = new byte[ChunkBytes];
                        _audioBuffer.CopyTo(padded);
                        chunk = padded;
                        _audioBuffer.Clear();
                    }
                }

                if (chunk is not null && _stream is not null)
                    await Protocol.SendMessageAsync(_stream, Protocol.MsgAudio, Protocol.PackAudioPayload(chunk), ct);
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex) when (ex is IOException or SocketException)
        {
            RaiseStopped("Lost connection to server.");
        }
    }

    // ?? Receiver loop ?????????????????????????????????????????????????????????

    private async Task ReceiveLoopAsync(CancellationToken ct)
    {
        try
        {
            while (!ct.IsCancellationRequested && _stream is not null)
            {
                var (msgType, payload) = await Protocol.RecvMessageAsync(_stream, ct);

                if (payload is null) // disconnect
                {
                    if (!ct.IsCancellationRequested)
                        RaiseStopped("Server disconnected.");
                    return;
                }

                if (msgType == Protocol.MsgError && payload.Length > 0)
                {
                    try
                    {
                        using var doc = JsonDocument.Parse(payload);
                        var error = doc.RootElement.TryGetProperty("error", out var e)
                            ? e.GetString() ?? "Unknown error"
                            : "Unknown error";
                        RaiseStopped($"Server error: {error}");
                    }
                    catch (JsonException)
                    {
                        RaiseStopped("Server error (malformed message).");
                    }
                    return;
                }

                if (msgType == Protocol.MsgTranscription && payload.Length > 0)
                {
                    try
                    {
                        using var doc = JsonDocument.Parse(payload);
                        var text = doc.RootElement.GetProperty("text").GetString() ?? string.Empty;
                        var isFinal = doc.RootElement.TryGetProperty("is_final", out var f) && f.GetBoolean();
                        RaiseTranscription(text, isFinal);
                    }
                    catch (JsonException) { /* ignore malformed messages */ }
                }
                else if (msgType == Protocol.MsgStats && payload.Length > 0)
                {
                    try
                    {
                        using var doc = JsonDocument.Parse(payload);
                        var stats = new LagStats(
                            NetworkLagMs:       doc.RootElement.GetProperty("network_lag_ms").GetInt32(),
                            QueueLagMs:         doc.RootElement.GetProperty("queue_lag_ms").GetInt32(),
                            TranscriptionLagMs: doc.RootElement.GetProperty("transcription_lag_ms").GetInt32(),
                            QueueDepth:         doc.RootElement.GetProperty("queue_depth").GetInt32(),
                            ChunksSkipped:      doc.RootElement.GetProperty("chunks_skipped").GetInt32());
                        RaiseStats(stats);
                    }
                    catch (JsonException) { /* ignore malformed messages */ }
                }
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex) when (ex is IOException or SocketException)
        {
            if (!ct.IsCancellationRequested)
                RaiseStopped("Lost connection to server.");
        }
    }

    // ?? Event helpers (marshal to UI thread) ??????????????????????????????????

    private void RaiseTranscription(string text, bool isFinal) =>
        _syncCtx.Post(_ => TranscriptionReceived?.Invoke(text, isFinal), null);

    private void RaiseStats(LagStats stats) =>
        _syncCtx.Post(_ => StatsReceived?.Invoke(stats), null);

    private void RaiseStopped(string? reason) =>
        _syncCtx.Post(_ => Stopped?.Invoke(reason), null);

    // ?? IDisposable ???????????????????????????????????????????????????????????

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Stop();
        _waveIn?.Dispose();
        _cts?.Dispose();
        _stream?.Dispose();
        _tcpClient?.Dispose();
    }
}

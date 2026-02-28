namespace TranscriptionClient;

/// <summary>Lag statistics reported by the server after each transcription cycle.</summary>
internal sealed record LagStats(
    int NetworkLagMs,
    int QueueLagMs,
    int TranscriptionLagMs,
    int QueueDepth,
    int ChunksSkipped);

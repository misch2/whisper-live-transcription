namespace TranscriptionClient;

/// <summary>Simple DTO used for ComboBox binding.</summary>
internal sealed class AudioDeviceInfo
{
    public int DeviceNumber { get; init; }
    public string Name { get; init; } = string.Empty;

    public override string ToString() => Name;
}

using System.IO;
using System.Text.Json;

namespace TranscriptionClient;

/// <summary>
/// Shared mutable state that flows between MainWindow and SetupWindow.
/// Persisted as JSON in the user's roaming AppData folder.
/// </summary>
public class AppSettings
{
    private static readonly string FilePath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
        "WhisperLiveTranscription",
        "settings.json");

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
    };

    public string Host { get; set; } = "localhost";
    public int Port { get; set; } = 43007;
    public string DeviceFilter { get; set; } = "Voicemeeter Out B1";
    public int DeviceNumber { get; set; } = 0;
    public string DeviceName { get; set; } = "(System default)";
    public bool AutoScroll { get; set; } = true;

    // ── Main window geometry ──────────────────────────────────────────────────
    public double? WindowLeft { get; set; } = null;
    public double? WindowTop { get; set; } = null;
    public double? WindowWidth { get; set; } = null;
    public double? WindowHeight { get; set; } = null;
    public string WindowState { get; set; } = nameof(System.Windows.WindowState.Normal);

    /// <summary>
    /// Loads settings from disk. Throws on failure so the caller can report the error.
    /// </summary>
    public static AppSettings Load()
    {
        if (!File.Exists(FilePath))
            return new AppSettings();

        string json = File.ReadAllText(FilePath);
        return JsonSerializer.Deserialize<AppSettings>(json, JsonOptions) ?? new AppSettings();
    }

    /// <summary>
    /// Saves settings to disk. Throws on failure so the caller can report the error.
    /// </summary>
    public void Save()
    {
        Directory.CreateDirectory(Path.GetDirectoryName(FilePath)!);
        string json = JsonSerializer.Serialize(this, JsonOptions);
        File.WriteAllText(FilePath, json);
    }
}

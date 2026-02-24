using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;
using NAudio.Wave;

namespace TranscriptionClient;

public partial class MainWindow : Window
{
    private TranscriptionService? _service;
    private List<AudioDeviceInfo> _allDevices = new();

    // Colours (match XAML palette)
    private static readonly SolidColorBrush GreenBrush  = new(Color.FromRgb(0x50, 0xFA, 0x7B));
    private static readonly SolidColorBrush RedBrush    = new(Color.FromRgb(0xFF, 0x55, 0x55));
    private static readonly SolidColorBrush AccentBrush = new(Color.FromRgb(0xBD, 0x93, 0xF9));
    private static readonly SolidColorBrush DimBrush    = new(Color.FromRgb(0x62, 0x72, 0xA4));
    private static readonly SolidColorBrush FgBrush     = new(Color.FromRgb(0xF8, 0xF8, 0xF2));

    public MainWindow()
    {
        InitializeComponent();
        Loaded += OnLoaded;
    }

    // ── Initialisation ────────────────────────────────────────────────────────

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        RefreshDeviceList(filter: string.Empty);

        // Pre-select the first device whose name starts with the default prefix
        const string defaultPrefix = "Voicemeeter Out B1";
        var preferred = _allDevices.FirstOrDefault(d =>
            d.Name.StartsWith(defaultPrefix, StringComparison.OrdinalIgnoreCase));
        CmbDevice.SelectedItem = preferred ?? CmbDevice.Items[0];

        TxtDeviceFilter.Text = defaultPrefix;
    }

    // ── Device list ───────────────────────────────────────────────────────────

    private void RefreshDeviceList(string filter)
    {
        _allDevices.Clear();
        int count = WaveInEvent.DeviceCount;
        for (int i = 0; i < count; i++)
        {
            var caps = WaveInEvent.GetCapabilities(i);
            _allDevices.Add(new AudioDeviceInfo { DeviceNumber = i, Name = caps.ProductName });
        }

        var filtered = string.IsNullOrWhiteSpace(filter)
            ? _allDevices
            : _allDevices.Where(d => d.Name.StartsWith(filter, StringComparison.OrdinalIgnoreCase)).ToList();

        // Always add a "System default" entry at index 0
        var items = new List<AudioDeviceInfo>
        {
            new() { DeviceNumber = -1, Name = "(System default)" }
        };
        items.AddRange(filtered.Count > 0 ? filtered : _allDevices);

        var previousName = (CmbDevice.SelectedItem as AudioDeviceInfo)?.Name;
        CmbDevice.ItemsSource = items;

        var restored = items.FirstOrDefault(d => d.Name == previousName);
        CmbDevice.SelectedItem = restored ?? items[0];
    }

    private void TxtDeviceFilter_TextChanged(object sender, TextChangedEventArgs e)
    {
        RefreshDeviceList(TxtDeviceFilter.Text.Trim());
    }

    // ── Connect ───────────────────────────────────────────────────────────────

    private async void BtnConnect_Click(object sender, RoutedEventArgs e)
    {
        if (!int.TryParse(TxtPort.Text.Trim(), out int port) || port < 1 || port > 65535)
        {
            MessageBox.Show("Please enter a valid port number (1–65535).",
                            "Invalid port", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        string host = TxtHost.Text.Trim();
        if (string.IsNullOrWhiteSpace(host))
        {
            MessageBox.Show("Please enter a server host.", "Invalid host",
                            MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        var selected = CmbDevice.SelectedItem as AudioDeviceInfo;
        int deviceNumber = selected?.DeviceNumber ?? 0;
        // WaveInEvent uses 0 for default; -1 from our sentinel → use 0
        if (deviceNumber < 0) deviceNumber = 0;

        SetConnecting();

        _service = new TranscriptionService();
        _service.TranscriptionReceived += OnTranscriptionReceived;
        _service.Stopped               += OnServiceStopped;

        try
        {
            await _service.StartAsync(host, port, deviceNumber);
            SetConnected(host, port, selected?.Name ?? "default");
        }
        catch (Exception ex)
        {
            _service.Dispose();
            _service = null;
            SetDisconnected();
            MessageBox.Show($"Could not connect to {host}:{port}\n\n{ex.Message}",
                            "Connection failed", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    // ── Disconnect ────────────────────────────────────────────────────────────

    private void BtnDisconnect_Click(object sender, RoutedEventArgs e)
    {
        _service?.Stop();
        _service?.Dispose();
        _service = null;
        SetDisconnected();
    }

    // ── Clear / Copy ──────────────────────────────────────────────────────────

    private void BtnClear_Click(object sender, RoutedEventArgs e)
    {
        TranscriptPanel.Children.Clear();
        TxtLive.Text = "(live transcription will appear here)";
    }

    private void BtnCopyAll_Click(object sender, RoutedEventArgs e)
    {
        var lines = TranscriptPanel.Children
            .OfType<TextBlock>()
            .Select(tb => tb.Text);
        Clipboard.SetText(string.Join(Environment.NewLine, lines));
    }

    // ── Transcription events ──────────────────────────────────────────────────

    private void OnTranscriptionReceived(string text, bool isFinal)
    {
        if (isFinal)
        {
            // Commit the live text as a final paragraph
            if (!string.IsNullOrWhiteSpace(TxtLive.Text) &&
                TxtLive.Text != "(live transcription will appear here)")
            {
                AddFinalLine(TxtLive.Text);
            }
            TxtLive.Text = string.Empty;
        }
        else
        {
            TxtLive.Text = $"[{DateTime.Now:HH:mm:ss}]  {text}";
        }
    }

    private void AddFinalLine(string text)
    {
        var tb = new TextBlock
        {
            Text         = text,
            Foreground   = FgBrush,
            FontSize     = 14,
            TextWrapping = TextWrapping.Wrap,
            Margin       = new Thickness(0, 0, 0, 8),
        };
        TranscriptPanel.Children.Add(tb);

        if (ChkAutoScroll.IsChecked == true)
            TranscriptScroller.ScrollToBottom();
    }

    // ── Service stopped ───────────────────────────────────────────────────────

    private void OnServiceStopped(string? reason)
    {
        _service?.Dispose();
        _service = null;
        SetDisconnected();

        if (!string.IsNullOrEmpty(reason))
            AppendStatusMessage(reason);
    }

    // ── UI state helpers ──────────────────────────────────────────────────────

    private void SetConnecting()
    {
        BtnConnect.IsEnabled    = false;
        BtnDisconnect.IsEnabled = false;
        SetStatus("Connecting…", DimBrush);
    }

    private void SetConnected(string host, int port, string device)
    {
        BtnConnect.IsEnabled    = false;
        BtnDisconnect.IsEnabled = true;
        SetStatus($"Connected to {host}:{port}  •  {device}", GreenBrush);
    }

    private void SetDisconnected()
    {
        BtnConnect.IsEnabled    = true;
        BtnDisconnect.IsEnabled = false;
        SetStatus("Disconnected", DimBrush);
    }

    private void SetStatus(string text, SolidColorBrush dotColor)
    {
        TxtStatus.Text  = text;
        StatusDot.Fill  = dotColor;
    }

    private void AppendStatusMessage(string msg)
    {
        TxtStatus.Text = msg;
        StatusDot.Fill = RedBrush;
    }
}
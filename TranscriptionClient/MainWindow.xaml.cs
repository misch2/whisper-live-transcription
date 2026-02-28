using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace TranscriptionClient;

public partial class MainWindow : Window
{
    private TranscriptionService? _service;
    private AppSettings _settings = LoadSettingsOrDefault();

    // Thresholds for colour-coding network / queue lag
    private const int LagWarnMs = 300;
    private const int LagBadMs  = 800;

    // The last child of TranscriptPanel; always a live (interim) block
    private TextBlock _liveBlock = null!;

    // ── Theme-aware brush helpers ─────────────────────────────────────────────

    private SolidColorBrush ThemeBrush(string key) =>
        (SolidColorBrush)Application.Current.Resources[key];

    private SolidColorBrush GreenBrush => ThemeBrush("GreenBrush");
    private SolidColorBrush RedBrush   => ThemeBrush("RedBrush");
    private SolidColorBrush AmberBrush => ThemeBrush("AmberBrush");
    private SolidColorBrush DimBrush   => ThemeBrush("FgDimBrush");
    private SolidColorBrush FgBrush    => ThemeBrush("FgBrush");

    public MainWindow()
    {
        InitializeComponent();
        Loaded += OnLoaded;
        Closing += OnClosing;
    }

    // ── Window geometry persistence ───────────────────────────────────────────

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        ThemeManager.Initialize(_settings.Theme);

        if (_settings.WindowLeft.HasValue && _settings.WindowTop.HasValue)
        {
            Left = _settings.WindowLeft.Value;
            Top  = _settings.WindowTop.Value;
        }

        if (_settings.WindowWidth.HasValue && _settings.WindowWidth.Value >= MinWidth)
            Width = _settings.WindowWidth.Value;

        if (_settings.WindowHeight.HasValue && _settings.WindowHeight.Value >= MinHeight)
            Height = _settings.WindowHeight.Value;

        if (Enum.TryParse<WindowState>(_settings.WindowState, out var state) &&
            state != System.Windows.WindowState.Minimized)
        {
            WindowState = state;
        }

        ResetLiveBlock();
    }

    private void OnClosing(object? sender, System.ComponentModel.CancelEventArgs e)
    {
        // Save normal bounds even when closing from maximized/minimized
        if (WindowState == System.Windows.WindowState.Normal)
        {
            _settings.WindowLeft = Left;
            _settings.WindowTop = Top;
            _settings.WindowWidth = Width;
            _settings.WindowHeight = Height;
        }
        _settings.WindowState = WindowState.ToString();

        try
        {
            _settings.Save();
        }
        catch (Exception ex)
        {
            MessageBox.Show(
                $"Window position could not be saved.\n\n{ex.Message}",
                "Settings error", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    // ── Settings persistence ──────────────────────────────────────────────────

    private static AppSettings LoadSettingsOrDefault()
    {
        try
        {
            return AppSettings.Load();
        }
        catch (Exception ex)
        {
            MessageBox.Show(
                $"Failed to load settings, using defaults.\n\n{ex.Message}",
                "Settings error", MessageBoxButton.OK, MessageBoxImage.Warning);
            return new AppSettings();
        }
    }

    // ── Setup window ──────────────────────────────────────────────────────────

    private void BtnSetup_Click(object sender, RoutedEventArgs e)
    {
        var setup = new SetupWindow(_settings) { Owner = this };
        if (setup.ShowDialog() == true)
        {
            _settings = setup.Settings;
            ThemeManager.Apply(_settings.Theme);
            try
            {
                _settings.Save();
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    $"Settings were applied but could not be saved to disk.\n\n{ex.Message}",
                    "Settings error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
    }

    // ── Connect ───────────────────────────────────────────────────────────────

    private async void BtnConnect_Click(object sender, RoutedEventArgs e)
    {
        int deviceNumber = _settings.DeviceNumber;
        if (deviceNumber < 0) deviceNumber = 0;

        SetConnecting();

        _service = new TranscriptionService();
        _service.TranscriptionReceived += OnTranscriptionReceived;
        _service.StatsReceived += OnStatsReceived;
        _service.Stopped += OnServiceStopped;

        try
        {
            await _service.StartAsync(_settings.Host, _settings.Port, deviceNumber);
            SetConnected(_settings.Host, _settings.Port, _settings.DeviceName);
        }
        catch (Exception ex)
        {
            _service.Dispose();
            _service = null;
            SetDisconnected();
            MessageBox.Show($"Could not connect to {_settings.Host}:{_settings.Port}\n\n{ex.Message}",
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
        ResetLiveBlock();
    }

    private void BtnCopyAll_Click(object sender, RoutedEventArgs e)
    {
        // Exclude the trailing live block from the copy
        var lines = TranscriptPanel.Children
            .OfType<TextBlock>()
            .Where(tb => tb != _liveBlock && !string.IsNullOrWhiteSpace(tb.Text))
            .Select(tb => tb.Text);
        Clipboard.SetText(string.Join(Environment.NewLine, lines));
    }

    // ── Transcription events ──────────────────────────────────────────────────

    private void OnTranscriptionReceived(string text, bool isFinal)
    {
        if (isFinal)
        {
            // Commit: turn the live block into a final line, then add a fresh live block
            CommitLiveBlock();
        }
        else
        {
            _liveBlock.Text = $"[{DateTime.Now:HH:mm:ss}]  {text}";
            if (_settings.AutoScroll)
                TranscriptScroller.ScrollToBottom();
        }
    }

    // ── Stats events ──────────────────────────────────────────────────────────

    private void OnStatsReceived(LagStats stats)
    {
        StatsBar.Visibility = Visibility.Visible;

        TxtNetLag.Text           = $"{stats.NetworkLagMs} ms";
        TxtNetLag.Foreground     = LagBrush(stats.NetworkLagMs);

        TxtQueueLag.Text         = $"{stats.QueueLagMs} ms";
        TxtQueueLag.Foreground   = LagBrush(stats.QueueLagMs);

        TxtTranscriptionLag.Text      = $"{stats.TranscriptionLagMs} ms";
        TxtTranscriptionLag.Foreground = LagBrush(stats.TranscriptionLagMs);

        TxtQueueDepth.Text       = stats.QueueDepth.ToString();
        TxtQueueDepth.Foreground = stats.QueueDepth > 2 ? RedBrush : FgBrush;

        TxtChunksSkipped.Text       = stats.ChunksSkipped.ToString();
        TxtChunksSkipped.Foreground = stats.ChunksSkipped > 0 ? RedBrush : FgBrush;
    }

    private SolidColorBrush LagBrush(int ms) =>
        ms >= LagBadMs  ? RedBrush   :
        ms >= LagWarnMs ? AmberBrush :
                          GreenBrush;

    // ── Live block helpers ────────────────────────────────────────────────────

    private void ResetLiveBlock()
    {
        _liveBlock = MakeLiveBlock();
        TranscriptPanel.Children.Add(_liveBlock);
    }

    private void CommitLiveBlock()
    {
        if (string.IsNullOrWhiteSpace(_liveBlock.Text))
            return;

        // Restyle the existing block to look like a final line
        _liveBlock.Foreground = FgBrush;
        _liveBlock.FontStyle = FontStyles.Normal;
        _liveBlock.Margin = new Thickness(0, 0, 0, 8);

        // Append a new live block after it
        _liveBlock = MakeLiveBlock();
        TranscriptPanel.Children.Add(_liveBlock);

        if (_settings.AutoScroll)
            TranscriptScroller.ScrollToBottom();
    }

    private TextBlock MakeLiveBlock() => new()
    {
        Text = string.Empty,
        Foreground = DimBrush,
        FontSize = 14,
        FontStyle = FontStyles.Italic,
        TextWrapping = TextWrapping.Wrap,
    };

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
        BtnConnect.IsEnabled = false;
        BtnDisconnect.IsEnabled = false;
        BtnSetup.IsEnabled = false;
        SetStatus("Connecting…", DimBrush);
    }

    private void SetConnected(string host, int port, string device)
    {
        BtnConnect.IsEnabled = false;
        BtnDisconnect.IsEnabled = true;
        BtnSetup.IsEnabled = false;
        SetStatus($"Connected to {host}:{port}  •  {device}", GreenBrush);
    }

    private void SetDisconnected()
    {
        BtnConnect.IsEnabled = true;
        BtnDisconnect.IsEnabled = false;
        BtnSetup.IsEnabled = true;
        SetStatus("Disconnected", DimBrush);
        StatsBar.Visibility = Visibility.Collapsed;
    }

    private void SetStatus(string text, SolidColorBrush dotColor)
    {
        TxtStatus.Text = text;
        StatusDot.Fill = dotColor;
    }

    private void AppendStatusMessage(string msg)
    {
        TxtStatus.Text = msg;
        StatusDot.Fill = RedBrush;
    }
}
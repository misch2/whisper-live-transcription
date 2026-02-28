using NAudio.Wave;
using System.Windows;
using System.Windows.Controls;

namespace TranscriptionClient;

public partial class SetupWindow : Window
{
    private List<AudioDeviceInfo> _allDevices = new();

    public AppSettings Settings { get; }

    public SetupWindow(AppSettings current)
    {
        InitializeComponent();

        // Work on a copy so Cancel truly cancels
        Settings = new AppSettings
        {
            Host = current.Host,
            Port = current.Port,
            DeviceFilter = current.DeviceFilter,
            DeviceNumber = current.DeviceNumber,
            DeviceName = current.DeviceName,
            AutoScroll = current.AutoScroll,
            Theme = current.Theme,
        };

        TxtHost.Text = Settings.Host;
        TxtPort.Text = Settings.Port.ToString();
        TxtDeviceFilter.Text = Settings.DeviceFilter;
        ChkAutoScroll.IsChecked = Settings.AutoScroll;

        RbThemeSystem.IsChecked = Settings.Theme == AppThemeMode.System;
        RbThemeLight.IsChecked  = Settings.Theme == AppThemeMode.Light;
        RbThemeDark.IsChecked   = Settings.Theme == AppThemeMode.Dark;

        // Live-preview the theme as the user clicks a radio button
        RbThemeSystem.Checked += (_, _) => ThemeManager.Apply(AppThemeMode.System);
        RbThemeLight.Checked  += (_, _) => ThemeManager.Apply(AppThemeMode.Light);
        RbThemeDark.Checked   += (_, _) => ThemeManager.Apply(AppThemeMode.Dark);

        Loaded += OnLoaded;
    }

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        RefreshDeviceList(TxtDeviceFilter.Text.Trim(), selectName: Settings.DeviceName);
    }

    private void RefreshDeviceList(string filter, string? selectName = null)
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

        var items = new List<AudioDeviceInfo>
        {
            new() { DeviceNumber = -1, Name = "(System default)" }
        };
        items.AddRange(filtered.Count > 0 ? filtered : _allDevices);

        var previousName = selectName ?? (CmbDevice.SelectedItem as AudioDeviceInfo)?.Name;
        CmbDevice.ItemsSource = items;

        var restored = items.FirstOrDefault(d => d.Name == previousName);
        CmbDevice.SelectedItem = restored ?? items[0];
    }

    private void TxtDeviceFilter_TextChanged(object sender, TextChangedEventArgs e)
    {
        RefreshDeviceList(TxtDeviceFilter.Text.Trim());
    }

    private void BtnOk_Click(object sender, RoutedEventArgs e)
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

        Settings.Host = host;
        Settings.Port = port;
        Settings.DeviceFilter = TxtDeviceFilter.Text.Trim();
        Settings.DeviceNumber = selected?.DeviceNumber ?? -1;
        Settings.DeviceName = selected?.Name ?? "(System default)";
        Settings.AutoScroll = ChkAutoScroll.IsChecked == true;
        Settings.Theme = RbThemeLight.IsChecked == true ? AppThemeMode.Light
                       : RbThemeDark.IsChecked  == true ? AppThemeMode.Dark
                                                        : AppThemeMode.System;

        DialogResult = true;
    }

    private void BtnCancel_Click(object sender, RoutedEventArgs e)
    {
        // Revert any live-preview theme change
        ThemeManager.Apply(Settings.Theme);
        DialogResult = false;
    }
}

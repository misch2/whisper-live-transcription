using Microsoft.Win32;
using System.Windows;

namespace TranscriptionClient;

/// <summary>
/// Applies and watches Light / Dark / System themes at runtime.
/// Call <see cref="Initialize"/> once from MainWindow.OnLoaded.
/// </summary>
public static class ThemeManager
{
    private const string RegistryKey =
        @"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize";
    private const string RegistryValue = "AppsUseLightTheme";

    private static AppThemeMode _currentMode = AppThemeMode.System;

    public static void Initialize(AppThemeMode mode) => Apply(mode);

    /// <summary>Switches to the requested mode and starts/stops registry watching.</summary>
    public static void Apply(AppThemeMode mode)
    {
        _currentMode = mode;

        SystemEvents.UserPreferenceChanged -= OnUserPreferenceChanged;
        if (mode == AppThemeMode.System)
            SystemEvents.UserPreferenceChanged += OnUserPreferenceChanged;

        ApplyResolved(Resolve(mode));
    }

    private static void OnUserPreferenceChanged(object sender, UserPreferenceChangedEventArgs e)
    {
        if (e.Category == UserPreferenceCategory.General)
            ApplyResolved(Resolve(AppThemeMode.System));
    }

    private static bool Resolve(AppThemeMode mode) => mode switch
    {
        AppThemeMode.Light => true,
        AppThemeMode.Dark  => false,
        _                  => ReadSystemUsesLight(),
    };

    private static bool ReadSystemUsesLight()
    {
        try
        {
            using var key = Registry.CurrentUser.OpenSubKey(RegistryKey);
            if (key?.GetValue(RegistryValue) is int value)
                return value != 0;
        }
        catch { /* fall through */ }
        return true;
    }

    private static void ApplyResolved(bool light)
    {
        var uri = new Uri(
            light ? "pack://application:,,,/Themes/Light.xaml"
                  : "pack://application:,,,/Themes/Dark.xaml",
            UriKind.Absolute);

        var dict = new ResourceDictionary { Source = uri };

        var app = Application.Current;
        var existing = app.Resources.MergedDictionaries
            .FirstOrDefault(d => d.Source?.OriginalString.Contains("/Themes/") == true);
        if (existing != null)
            app.Resources.MergedDictionaries.Remove(existing);

        app.Resources.MergedDictionaries.Add(dict);
    }
}

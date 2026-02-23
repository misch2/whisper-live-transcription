import wave
import threading
from typing import List, Optional

class WaveRecorder:
    """Handle WAV audio recording and saving"""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width  # 2 bytes for int16
        
        self.audio_data: List[bytes] = []
        self.output_path: Optional[str] = None
        self.lock = threading.Lock()
        self.wave_file: Optional[wave.Wave_write] = None
        self.is_file_open = False
    
    def set_output_path(self, path: str):
        """Set the output file path for the WAV file and open it for writing"""
        self.output_path = path
        try:
            self.wave_file = wave.open(path, 'wb')
            self.wave_file.setnchannels(self.channels)
            self.wave_file.setsampwidth(self.sample_width)
            self.wave_file.setframerate(self.sample_rate)
            self.is_file_open = True
            print(f"WAV file opened for continuous writing: {path}")
        except Exception as e:
            print(f"Error opening WAV file: {e}")
            self.wave_file = None
            self.is_file_open = False
    
    def add_audio_chunk(self, audio_chunk: bytes):
        """Add an audio chunk and write it immediately to the WAV file"""
        with self.lock:
            self.audio_data.append(audio_chunk)
            
            # Write immediately to file if it's open
            if self.is_file_open and self.wave_file:
                try:
                    self.wave_file.writeframes(audio_chunk)
                except Exception as e:
                    print(f"Error writing audio chunk: {e}")
                    self.is_file_open = False
    
    def save_to_file(self) -> bool:
        """Close the WAV file (audio has been written continuously)"""
        if not self.is_file_open or not self.wave_file:
            print("No WAV file is currently open.")
            return False
        
        try:
            with self.lock:
                self.wave_file.close()
                self.wave_file = None
                self.is_file_open = False
            
            print(f"WAV file closed successfully: {self.output_path}")
            return True
            
        except Exception as e:
            print(f"Error closing WAV file: {e}")
            return False
    
    def get_duration_seconds(self) -> float:
        """Get the current recording duration in seconds"""
        with self.lock:
            if not self.audio_data:
                return 0.0
            
            total_bytes = sum(len(chunk) for chunk in self.audio_data)
            # Each sample is sample_width bytes, sample_rate samples per second
            total_samples = total_bytes // (self.sample_width * self.channels)
            return total_samples / self.sample_rate
    
    def clear(self):
        """Clear all recorded audio data"""
        with self.lock:
            self.audio_data.clear()
    
    def is_recording(self) -> bool:
        """Check if we have an output path set (indicating recording is enabled)"""
        return self.output_path is not None
    
    def close(self):
        """Properly close the WAV file if it's open"""
        if self.is_file_open and self.wave_file:
            try:
                with self.lock:
                    self.wave_file.close()
                    self.wave_file = None
                    self.is_file_open = False
                print(f"WAV file closed: {self.output_path}")
            except Exception as e:
                print(f"Error closing WAV file: {e}")
    
    def __del__(self):
        """Ensure file is closed when object is destroyed"""
        self.close()

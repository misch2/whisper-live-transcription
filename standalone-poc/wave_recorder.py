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
    
    def set_output_path(self, path: str):
        """Set the output file path for the WAV file"""
        self.output_path = path
    
    def add_audio_chunk(self, audio_chunk: bytes):
        """Add an audio chunk to the recording buffer"""
        with self.lock:
            self.audio_data.append(audio_chunk)
    
    def save_to_file(self) -> bool:
        """Save all collected audio data to WAV file"""
        if not self.audio_data or not self.output_path:
            print("No audio data or output path to save.")
            return False
        
        try:
            with self.lock:
                # Combine all audio chunks
                combined_audio = b''.join(self.audio_data)
            
            # Write to WAV file
            with wave.open(self.output_path, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.sample_width)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(combined_audio)
            
            print(f"Audio saved successfully to: {self.output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving audio: {e}")
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
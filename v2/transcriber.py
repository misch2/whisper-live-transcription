import threading
import numpy as np
import time
from faster_whisper import WhisperModel

class Transcriber(threading.Thread):
    def __init__(self, buffer, sample_rate=16000, chunk_seconds=30, whisper_model="turbo", whisper_device="cuda", whisper_compute_type="float16", whisper_threads=4, language="en"):
        super().__init__()
        self.buffer = buffer
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.model = WhisperModel(
            whisper_model,
            device=whisper_device,
            compute_type=whisper_compute_type,
            cpu_threads=whisper_threads
        )
        self.language = language
        self._stop_event = threading.Event()
        self.transcriptions = []
        self.last_read_pos = 0
        self.lag_warning_threshold = 60  # seconds
        self.latest_text = ""
        self.lock = threading.Lock()

    def run(self):
        while not self._stop_event.is_set():
            audio = self.buffer.read_latest(self.chunk_seconds)
            self.buffer.read_pos = self.buffer.write_pos  # update read position
            lag = self.buffer.get_lag()
            if lag > self.lag_warning_threshold:
                print(f"[WARNING] Transcription lag: {lag:.2f} seconds")
            if len(audio) < self.chunk_seconds * self.sample_rate:
                time.sleep(0.5)
                continue
            audio_float = audio.astype(np.float32) / 255.0
            segments, _ = self.model.transcribe(
                audio_float,
                language=self.language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=1000)
            )
            texts = [s.text for s in segments]
            combined = " ".join(texts)
            with self.lock:
                self.transcriptions.append(combined)
                self.latest_text = combined
            time.sleep(1)

    def stop(self):
        self._stop_event.set()

    def get_latest(self):
        with self.lock:
            return self.latest_text

    def get_all(self):
        with self.lock:
            return " ".join(self.transcriptions)

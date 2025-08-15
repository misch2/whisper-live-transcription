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
        self.transcriptions = []  # List of dicts: {text, timestamp, start_time, end_time, duration}
        self.last_read_pos = 0
        self.lag_warning_threshold = 60  # seconds
        self.latest_text = ""
        self.latest_metadata = None
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
            end_time = time.time()
            start_time = end_time - self.chunk_seconds
            segments, _ = self.model.transcribe(
                audio_float,
                language=self.language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=1000)
            )
            texts = [s.text for s in segments]
            combined = " ".join(texts)
            timestamp_str = time.strftime("%H:%M:%S", time.localtime(end_time)) + f".{int((end_time%1)*1000):03d}"
            metadata = {
                "text": combined,
                "timestamp": timestamp_str,
                "start_time": start_time,
                "end_time": end_time,
                "duration": self.chunk_seconds
            }
            with self.lock:
                self.transcriptions.append(metadata)
                self.latest_text = combined
                self.latest_metadata = metadata
            time.sleep(1)

    def stop(self):
        self._stop_event.set()

    def get_latest(self):
        with self.lock:
            return self.latest_text, self.latest_metadata

    def get_all(self):
        with self.lock:
            return list(self.transcriptions)

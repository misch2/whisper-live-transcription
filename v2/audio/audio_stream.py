import threading
import numpy as np
import pyaudio

class AudioStream(threading.Thread):
    def __init__(self, buffer, device_index=None, sample_rate=16000, chunk_size=16000, channels=1):
        super().__init__()
        self.buffer = buffer
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.device_index = device_index
        self._stop_event = threading.Event()

    def run(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
        )
        while not self._stop_event.is_set():
            data = stream.read(self.chunk_size, exception_on_overflow=False)
            arr = np.frombuffer(data, dtype=np.int16)
            self.buffer.write(arr)
        stream.stop_stream()
        stream.close()
        audio.terminate()

    def stop(self):
        self._stop_event.set()

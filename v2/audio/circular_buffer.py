import threading
import numpy as np
import time

class CircularBuffer:
    def __init__(self, max_seconds, sample_rate, dtype=np.int16):
        self.max_samples = max_seconds * sample_rate
        self.buffer = np.zeros(self.max_samples, dtype=dtype)
        self.sample_rate = sample_rate
        self.write_pos = 0
        self.read_pos = 0
        self.lock = threading.Lock()
        self.last_write_time = time.time()

    def write(self, data: np.ndarray):
        with self.lock:
            n = len(data)
            end_pos = (self.write_pos + n) % self.max_samples
            if end_pos < self.write_pos:
                self.buffer[self.write_pos:] = data[:self.max_samples-self.write_pos]
                self.buffer[:end_pos] = data[self.max_samples-self.write_pos:]
            else:
                self.buffer[self.write_pos:end_pos] = data
            self.write_pos = end_pos
            self.last_write_time = time.time()

    def read_latest(self, seconds: int):
        with self.lock:
            n = seconds * self.sample_rate
            end_pos = self.write_pos
            start_pos = (end_pos - n) % self.max_samples
            if start_pos < end_pos:
                return self.buffer[start_pos:end_pos].copy()
            else:
                return np.concatenate((self.buffer[start_pos:], self.buffer[:end_pos])).copy()

    def get_lag(self):
        with self.lock:
            lag_samples = (self.write_pos - self.read_pos) % self.max_samples
            return lag_samples / self.sample_rate

    def advance_read_pos(self, seconds: int):
        with self.lock:
            self.read_pos = (self.read_pos + seconds * self.sample_rate) % self.max_samples

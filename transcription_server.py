#!/usr/bin/env python3
"""
Whisper Live Transcription Server
══════════════════════════════════
Accepts a single TCP client at a time, receives raw PCM audio,
transcribes it with faster-whisper, streams results back, and
optionally saves the audio as a WAV file.

The Whisper model is loaded once on startup.  Processing only runs
while a client is connected — when the client disconnects the server
idles with near-zero CPU/GPU use until the next connection.

Usage
─────
  python transcription_server.py [--host HOST] [--port PORT] \\
                                 [--audio-output-path FOLDER]

Windows service (via NSSM)
──────────────────────────
  nssm install WhisperTranscription "<venv>\\python.exe" "transcription_server.py --host 0.0.0.0 --port 43007"
  nssm start WhisperTranscription
"""

import argparse
import json
import os
import queue
import socket
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from faster_whisper import WhisperModel

from wave_recorder import WaveRecorder
from protocol import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    MSG_AUDIO,
    MSG_CONFIG,
    MSG_TRANSCRIPTION,
    recv_message,
    send_message,
)

# ── Whisper settings ─────────────────────────────────────────────────────────
WHISPER_LANGUAGE = "en"
WHISPER_THREADS = 4
WHISPER_MODEL = "turbo"
WHISPER_COMPUTE_TYPE = "float16"

# ── Transcription window ─────────────────────────────────────────────────────
WINDOW_LENGTH_SEC = 6
MAX_SENTENCE_CHARACTERS = 80


# ══════════════════════════════════════════════════════════════════════════════
#  Server
# ══════════════════════════════════════════════════════════════════════════════
class TranscriptionServer:
    """Single-client TCP server for live audio transcription."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        audio_output_path: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.audio_output_path = audio_output_path

        self.running = False
        self.whisper: Optional[WhisperModel] = None
        self._server_socket: Optional[socket.socket] = None
        self._client_socket: Optional[socket.socket] = None

    # ── Model loading ─────────────────────────────────────────────────────
    def _load_model(self):
        try:
            import ctranslate2
            cuda_devices = ctranslate2.get_cuda_device_count()
            if cuda_devices > 0:
                device = "cuda"
                print(f"GPU detected ({cuda_devices} device(s)), using CUDA.")
            else:
                device = "cpu"
                print("No CUDA GPU detected, falling back to CPU.")
        except Exception:
            device = "cpu"
            print("Could not query CUDA devices, falling back to CPU.")

        compute_type = WHISPER_COMPUTE_TYPE if device == "cuda" else "int8"

        home_dir = os.path.expanduser("~")
        models_dir = os.path.join(home_dir, ".cache", "huggingface", "hub")
        print(
            f"Loading Whisper model '{WHISPER_MODEL}' "
            f"(device={device}, compute={compute_type}, threads={WHISPER_THREADS})..."
        )
        print(f"Model cache: {models_dir}")
        self.whisper = WhisperModel(
            WHISPER_MODEL,
            device=device,
            compute_type=compute_type,
            cpu_threads=WHISPER_THREADS,
            download_root=models_dir,
        )
        print("Whisper model ready.\n")

    # ── Accept loop ───────────────────────────────────────────────────────
    def start(self):
        """Load the model and start accepting connections (blocks)."""
        self._load_model()
        self.running = True

        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.settimeout(1.0)  # lets us check self.running
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(1)

        print(f"Listening on {self.host}:{self.port}")
        print("Waiting for client...\n")

        while self.running:
            try:
                client_sock, addr = self._server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            self._client_socket = client_sock
            print(f"\n{'=' * 60}")
            print(f"  Client connected: {addr}")
            print(f"{'=' * 60}")

            try:
                self._handle_client(client_sock)
            except Exception as exc:
                print(f"\nError while handling client: {exc}")
            finally:
                self._client_socket = None

            print(f"\nClient {addr} disconnected.")
            print("Waiting for client...\n")

        if self._server_socket:
            self._server_socket.close()
        print("Server stopped.")

    def stop(self):
        """Signal the server to shut down (can be called from any thread)."""
        self.running = False
        for s in (self._client_socket, self._server_socket):
            if s:
                try:
                    s.close()
                except OSError:
                    pass

    # ── Per-client session ────────────────────────────────────────────────
    def _handle_client(self, sock: socket.socket):
        audio_q: queue.Queue = queue.Queue()
        stop = threading.Event()
        stats: Dict[str, List[float]] = {"transcription": []}

        # Optional WAV recorder
        wav = self._open_wav_recorder()

        # Reader thread: socket ──► audio_q
        reader = threading.Thread(
            target=self._reader_thread,
            args=(sock, audio_q, wav, stop),
            daemon=True,
        )
        reader.start()

        # Process audio (runs in this thread)
        self._processor_loop(sock, audio_q, stop, stats)

        # ── Cleanup ───────────────────────────────────────────────────────
        stop.set()
        reader.join(timeout=3.0)

        if wav:
            wav.close()
            dur = wav.get_duration_seconds()
            print(f"\nWAV saved ({dur:.1f}s): {wav.output_path}")

        if stats["transcription"]:
            avg = np.mean(stats["transcription"])
            std = np.std(stats["transcription"])
            print(
                f"Processed {len(stats['transcription'])} chunks — "
                f"avg {avg:.3f}s ± {std:.3f}s per chunk"
            )

        try:
            sock.close()
        except OSError:
            pass

    # ── Reader thread ─────────────────────────────────────────────────────
    def _reader_thread(self, sock, audio_q, wav, stop):
        """Blocking read loop — runs in its own thread."""
        try:
            while not stop.is_set():
                msg_type, payload = recv_message(sock)
                if msg_type is None:
                    break
                if msg_type == MSG_CONFIG and payload:
                    try:
                        cfg = json.loads(payload.decode("utf-8"))
                        print(f"Client config: {cfg}")
                    except json.JSONDecodeError:
                        pass
                elif msg_type == MSG_AUDIO and payload:
                    audio_q.put(payload)
                    if wav:
                        wav.add_audio_chunk(payload)
        except Exception as exc:
            if not stop.is_set():
                print(f"\nReader error: {exc}")
        finally:
            stop.set()
            audio_q.put(None)  # sentinel to unblock processor

    # ── Processor loop ────────────────────────────────────────────────────
    def _processor_loop(self, sock, audio_q, stop, stats):
        """Consume audio chunks, transcribe, and send results back."""
        window: List[bytes] = []

        while not stop.is_set():
            try:
                chunk = audio_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if chunk is None:
                break

            t0 = time.time()

            # Sliding-window management
            if len(window) >= WINDOW_LENGTH_SEC:
                window.clear()
                self._send_transcription(sock, "", is_final=True, stop=stop)

            window.append(chunk)

            # Build numpy array from window
            audio_bytes = b"".join(window)
            audio_array = (
                np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 255.0
            )

            # Transcribe
            assert self.whisper is not None
            segments, _ = self.whisper.transcribe(
                audio_array,
                language=WHISPER_LANGUAGE,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=1000),
            )
            text = " ".join(s.text for s in segments).ljust(MAX_SENTENCE_CHARACTERS)

            elapsed = time.time() - t0
            stats["transcription"].append(elapsed)

            # Mirror on server console
            ts = time.strftime("%H:%M:%S")
            print(f"\r{ts} {text}", end="", flush=True)

            self._send_transcription(sock, text, is_final=False, stop=stop)

    # ── Send helper ───────────────────────────────────────────────────────
    @staticmethod
    def _send_transcription(sock, text, *, is_final, stop):
        payload = json.dumps({"text": text, "is_final": is_final}).encode("utf-8")
        try:
            send_message(sock, MSG_TRANSCRIPTION, payload)
        except OSError:
            stop.set()

    # ── WAV helper ────────────────────────────────────────────────────────
    def _open_wav_recorder(self) -> Optional[WaveRecorder]:
        if not self.audio_output_path:
            return None
        os.makedirs(self.audio_output_path, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.audio_output_path, f"recording_{ts}.wav")
        rec = WaveRecorder(sample_rate=16000, channels=1)
        rec.set_output_path(path)
        return rec


# ══════════════════════════════════════════════════════════════════════════════
#  CLI entry-point
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Whisper Live Transcription Server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind address (default: %(default)s)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP port (default: %(default)s)")
    parser.add_argument(
        "--audio-output-path",
        type=str,
        help="Folder to save WAV recordings (one file per client session)",
    )
    args = parser.parse_args()

    server = TranscriptionServer(
        host=args.host,
        port=args.port,
        audio_output_path=args.audio_output_path,
    )
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n\nShutting down …")
        server.stop()


if __name__ == "__main__":
    main()

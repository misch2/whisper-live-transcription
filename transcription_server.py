#!/usr/bin/env python3
"""
Whisper Live Transcription Server
══════════════════════════════════
Accepts a single TCP client at a time, receives raw PCM audio,
transcribes it with faster-whisper, streams results back, and
optionally saves the audio as a WAV file.

The Whisper model is loaded on demand when the first client connects and
unloaded automatically after the server has been idle (no connected client)
for MODEL_IDLE_TIMEOUT_SEC seconds (default: 60).

Usage
─────
  python transcription_server.py [--host HOST] [--port PORT] \\
                                 [--audio-output-path FOLDER]

Windows service (via NSSM)
──────────────────────────
  nssm install WhisperTranscription "<venv>\\python.exe" "transcription_server.py --host 0.0.0.0 --port 43007"
  nssm start WhisperTranscription

  AppStdout / AppStderr do NOT need to be configured — the server
  automatically writes timestamped logs to:
    %LOCALAPPDATA%\\WhisperLiveTranscription\\server.log
    %LOCALAPPDATA%\\WhisperLiveTranscription\\server_error.log
"""

import argparse
import gc
import io
import json
import os
import queue
import socket
import sys
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
    MSG_ERROR,
    MSG_TRANSCRIPTION,
    MSG_STATS,
    recv_message,
    send_message,
    unpack_audio_payload,
)

# ── Whisper settings ─────────────────────────────────────────────────────────
DEFAULT_WHISPER_LANGUAGE = "en"
DEFAULT_WHISPER_MODEL = "turbo"
WHISPER_THREADS = 4
WHISPER_COMPUTE_TYPE = "float16"

# ── Transcription window ─────────────────────────────────────────────────────
WINDOW_LENGTH_SEC = 6
MAX_SENTENCE_CHARACTERS = 80


# ══════════════════════════════════════════════════════════════════════════════
#  Logging redirect
# ══════════════════════════════════════════════════════════════════════════════
class TimestampedTee(io.TextIOBase):
    """
    Wraps an existing stream so every *complete line* written to it gets a
    ``[YYYY-MM-DD HH:MM:SS] `` prefix and is simultaneously written to a log
    file.  The original stream is kept as a tee so interactive runs continue
    to show output on the terminal.
    """

    def __init__(self, original: io.TextIOBase, log_file: io.TextIOBase) -> None:
        super().__init__()
        self._orig = original
        self._log = log_file
        self._buf = ""
        self._lock = threading.Lock()

    # TextIOBase.write must return the number of characters accepted.
    def write(self, text: str) -> int:
        if not text:
            return 0
        with self._lock:
            self._buf += text
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                self._emit(line + "\n")
        return len(text)

    def flush(self) -> None:
        with self._lock:
            if self._buf:
                self._emit(self._buf)
                self._buf = ""
        self._orig.flush()
        self._log.flush()

    def _emit(self, line: str) -> None:
        stamped = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {line}"
        self._orig.write(stamped)
        self._orig.flush()
        self._log.write(stamped)
        self._log.flush()

    # Preserve encoding/errors so libraries that inspect sys.stdout still work.
    @property
    def encoding(self):
        return self._orig.encoding

    @property
    def errors(self):
        return self._orig.errors


def setup_logging() -> None:
    """
    Redirect sys.stdout and sys.stderr to TimestampedTee instances that write
    to ``%LOCALAPPDATA%\\WhisperLiveTranscription\\server.log`` (stdout) and
    ``server_error.log`` (stderr).  The directory is created if absent.
    """
    local_app_data = os.environ.get("LOCALAPPDATA") or os.path.join(
        os.path.expanduser("~"), "AppData", "Local"
    )
    log_dir = os.path.join(local_app_data, "WhisperLiveTranscription")
    os.makedirs(log_dir, exist_ok=True)

    stdout_log = open(
        os.path.join(log_dir, "server.log"), "a", encoding="utf-8", buffering=1
    )
    stderr_log = open(
        os.path.join(log_dir, "server_error.log"), "a", encoding="utf-8", buffering=1
    )

    sys.stdout = TimestampedTee(sys.__stdout__, stdout_log)
    sys.stderr = TimestampedTee(sys.__stderr__, stderr_log)

    print(f"Logging to: {log_dir}")


# ══════════════════════════════════════════════════════════════════════════════
#  Server
# ══════════════════════════════════════════════════════════════════════════════
class TranscriptionServer:
    """Single-client TCP server for live audio transcription."""

    # How long to keep the model in memory after the last client disconnects.
    MODEL_IDLE_TIMEOUT_SEC = 60

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
        self._loaded_model_name: Optional[str] = None
        self._server_socket: Optional[socket.socket] = None
        self._client_socket: Optional[socket.socket] = None
        self._unload_timer: Optional[threading.Timer] = None
        self._model_lock = threading.Lock()

    # ── Model loading ─────────────────────────────────────────────────────
    def _load_model(self, model_name: str):
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
            f"Loading Whisper model '{model_name}' "
            f"(device={device}, compute={compute_type}, threads={WHISPER_THREADS})..."
        )
        print(f"Model cache: {models_dir}")
        self.whisper = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=WHISPER_THREADS,
            download_root=models_dir,
        )
        self._loaded_model_name = model_name
        print("Whisper model ready.\n")

    def _ensure_model_loaded(self, model_name: str = DEFAULT_WHISPER_MODEL):
        """Load or reload the model if needed (thread-safe)."""
        with self._model_lock:
            if self.whisper is None or self._loaded_model_name != model_name:
                if self.whisper is not None:
                    print(f"Model change requested: '{self._loaded_model_name}' -> '{model_name}'")
                    self.whisper = None
                    gc.collect()
                self._load_model(model_name)

    def _unload_model(self):
        """Release the model and free memory."""
        with self._model_lock:
            if self.whisper is not None:
                print("No client connected for 60 s — unloading Whisper model.")
                self.whisper = None
                self._loaded_model_name = None
                gc.collect()

    def _cancel_unload_timer(self):
        if self._unload_timer is not None:
            self._unload_timer.cancel()
            self._unload_timer = None

    def _schedule_unload(self):
        self._cancel_unload_timer()
        self._unload_timer = threading.Timer(self.MODEL_IDLE_TIMEOUT_SEC, self._unload_model)
        self._unload_timer.daemon = True
        self._unload_timer.start()

    # ── Accept loop ───────────────────────────────────────────────────────
    def start(self):
        """Start accepting connections (blocks). Model is loaded on demand."""
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

            self._cancel_unload_timer()

            self._client_socket = client_sock
            print(f"\n{'=' * 60}")
            print(f"  Client connected: {addr}")
            print(f"{'=' * 60}")

            try:
                self._handle_client(client_sock)
            except Exception as exc:
                print(f"\nError while handling client: {exc}")
                self._send_error(client_sock, str(exc))
            finally:
                self._client_socket = None

            print(f"\nClient {addr} disconnected.")
            self._schedule_unload()
            print(f"Model will be unloaded in {self.MODEL_IDLE_TIMEOUT_SEC} s if no new client connects.")
            print("Waiting for client...\n")

        self._cancel_unload_timer()
        if self._server_socket:
            self._server_socket.close()
        print("Server stopped.")

    def stop(self):
        """Signal the server to shut down (can be called from any thread)."""
        self.running = False
        self._cancel_unload_timer()
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
        # Client-requested settings (populated from CONFIG message)
        session: Dict[str, str] = {
            "model": DEFAULT_WHISPER_MODEL,
            "language": DEFAULT_WHISPER_LANGUAGE,
        }

        # Optional WAV recorder
        wav = self._open_wav_recorder()

        # Reader thread: socket ──► audio_q
        reader = threading.Thread(
            target=self._reader_thread,
            args=(sock, audio_q, wav, stop, session),
            daemon=True,
        )
        reader.start()

        # Wait for the reader to parse the CONFIG message before loading the model
        # (give it a short window — the CONFIG is the first message the client sends)
        for _ in range(50):  # up to 500 ms
            if session.get("_config_received") or stop.is_set():
                break
            time.sleep(0.01)

        self._ensure_model_loaded(session["model"])
        print(f"Session language: {session['language']}, model: {session['model']}")

        # Process audio (runs in this thread)
        self._processor_loop(sock, audio_q, stop, stats, session)

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
    def _reader_thread(self, sock, audio_q, wav, stop, session):
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
                        if "model" in cfg:
                            session["model"] = cfg["model"]
                        if "language" in cfg:
                            session["language"] = cfg["language"]
                        session["_config_received"] = True
                    except json.JSONDecodeError:
                        pass
                elif msg_type == MSG_AUDIO and payload:
                    server_recv_ms = int(time.time() * 1000)
                    try:
                        client_ts_ms, pcm_bytes = unpack_audio_payload(payload)
                    except Exception:
                        # Older client without timestamp prefix — treat whole payload as PCM
                        client_ts_ms = server_recv_ms
                        pcm_bytes = payload
                    audio_q.put((pcm_bytes, client_ts_ms, server_recv_ms))
                    if wav:
                        wav.add_audio_chunk(pcm_bytes)
        except Exception as exc:
            if not stop.is_set():
                print(f"\nReader error: {exc}")
        finally:
            stop.set()
            audio_q.put(None)  # sentinel to unblock processor

    # ── Processor loop ────────────────────────────────────────────────────
    def _processor_loop(self, sock, audio_q, stop, stats, session):
        """Consume audio chunks, transcribe, and send results back."""
        window: List[bytes] = []
        chunks_skipped = 0

        while not stop.is_set():
            try:
                item = audio_q.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:
                break

            chunk, client_ts_ms, server_recv_ms = item
            queue_dequeue_ms = int(time.time() * 1000)

            # Sliding-window management
            if len(window) >= WINDOW_LENGTH_SEC:
                window.clear()
                chunks_skipped += 1
                self._send_transcription(sock, "", is_final=True, stop=stop)

            window.append(chunk)

            # Build numpy array from window
            audio_bytes = b"".join(window)
            audio_array = (
                np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
            )

            # Transcribe
            assert self.whisper is not None
            t_transcribe_start = time.time()
            segments, _ = self.whisper.transcribe(
                audio_array,
                language=session["language"],
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=1000),
            )
            text = " ".join(s.text for s in segments).ljust(MAX_SENTENCE_CHARACTERS)
            t_transcribe_end = time.time()

            transcription_lag_ms = int((t_transcribe_end - t_transcribe_start) * 1000)
            network_lag_ms = max(0, server_recv_ms - client_ts_ms)
            queue_lag_ms = max(0, queue_dequeue_ms - server_recv_ms)
            queue_depth = audio_q.qsize()

            stats["transcription"].append(transcription_lag_ms / 1000.0)

            self._send_transcription(sock, text, is_final=False, stop=stop)
            self._send_stats(
                sock,
                network_lag_ms=network_lag_ms,
                queue_lag_ms=queue_lag_ms,
                transcription_lag_ms=transcription_lag_ms,
                queue_depth=queue_depth,
                chunks_skipped=chunks_skipped,
                stop=stop,
            )

    # ── Send helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _send_transcription(sock, text, *, is_final, stop):
        payload = json.dumps({"text": text, "is_final": is_final}).encode("utf-8")
        try:
            send_message(sock, MSG_TRANSCRIPTION, payload)
        except OSError:
            stop.set()

    @staticmethod
    def _send_error(sock, message):
        payload = json.dumps({"error": message}).encode("utf-8")
        try:
            send_message(sock, MSG_ERROR, payload)
        except OSError:
            pass

    @staticmethod
    def _send_stats(sock, *, network_lag_ms, queue_lag_ms, transcription_lag_ms,
                    queue_depth, chunks_skipped, stop):
        payload = json.dumps({
            "network_lag_ms": network_lag_ms,
            "queue_lag_ms": queue_lag_ms,
            "transcription_lag_ms": transcription_lag_ms,
            "queue_depth": queue_depth,
            "chunks_skipped": chunks_skipped,
        }).encode("utf-8")
        try:
            send_message(sock, MSG_STATS, payload)
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
    setup_logging()

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

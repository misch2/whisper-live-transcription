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

Standalone
──────────
  python transcription_server.py [--host HOST] [--port PORT] \\
                                 [--audio-output-path FOLDER]

Windows service  (requires pywin32)
───────────────
  python transcription_server.py install   # install the service
  python transcription_server.py remove    # remove it
  python transcription_server.py start     # start via SCM
  python transcription_server.py stop      # stop via SCM
  python transcription_server.py debug     # run interactively for debugging

When installed as a service, settings are read from
``transcription_server_config.json`` next to this script.
"""

import argparse
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

# ── Resolve imports relative to this script ───────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "standalone-poc"))

from wave_recorder import WaveRecorder  # type: ignore[import-untyped]  # noqa: E402

from protocol import (  # noqa: E402
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

# ── Service helpers ───────────────────────────────────────────────────────────
SERVICE_CONFIG_FILE = os.path.join(_SCRIPT_DIR, "transcription_server_config.json")
_SERVICE_COMMANDS = {"install", "update", "remove", "start", "stop", "restart", "debug"}


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
        home_dir = os.path.expanduser("~")
        models_dir = os.path.join(home_dir, ".cache", "huggingface", "hub")
        print(
            f"Loading Whisper model '{WHISPER_MODEL}' "
            f"(threads={WHISPER_THREADS}, compute={WHISPER_COMPUTE_TYPE}) …"
        )
        print(f"Model cache: {models_dir}")
        self.whisper = WhisperModel(
            WHISPER_MODEL,
            device="cuda",
            compute_type=WHISPER_COMPUTE_TYPE,
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
        print("Waiting for client …\n")

        while self.running:
            try:
                client_sock, addr = self._server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            self._client_socket = client_sock
            print(f"\n{'═' * 60}")
            print(f"  Client connected: {addr}")
            print(f"{'═' * 60}")

            try:
                self._handle_client(client_sock)
            except Exception as exc:
                print(f"\nError while handling client: {exc}")
            finally:
                self._client_socket = None

            print(f"\nClient {addr} disconnected.")
            print("Waiting for client …\n")

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
#  Windows service  (optional — requires pywin32)
# ══════════════════════════════════════════════════════════════════════════════
try:
    import win32serviceutil
    import win32service
    import win32event
    import servicemanager

    class TranscriptionWindowsService(win32serviceutil.ServiceFramework):
        _svc_name_ = "WhisperTranscription"
        _svc_display_name_ = "Whisper Live Transcription Server"
        _svc_description_ = (
            "Accepts audio streams over TCP, transcribes with Whisper, "
            "and optionally saves WAV recordings."
        )

        def __init__(self, args):
            super().__init__(args)
            self.stop_event = win32event.CreateEvent(None, 0, 0, None)
            self.server: Optional[TranscriptionServer] = None

        def SvcStop(self):
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            win32event.SetEvent(self.stop_event)
            if self.server:
                self.server.stop()

        def SvcDoRun(self):
            servicemanager.LogMsg(
                servicemanager.EVENTLOG_INFORMATION_TYPE,
                servicemanager.PYS_SERVICE_STARTED,
                (self._svc_name_, ""),
            )
            cfg = _load_service_config()
            self.server = TranscriptionServer(
                host=cfg.get("host", DEFAULT_HOST),
                port=cfg.get("port", DEFAULT_PORT),
                audio_output_path=cfg.get("audio_output_path"),
            )
            self.server.start()

    _HAS_SERVICE = True

except ImportError:
    _HAS_SERVICE = False


def _load_service_config() -> dict:
    """Read the JSON config file (used by the Windows service)."""
    if os.path.exists(SERVICE_CONFIG_FILE):
        with open(SERVICE_CONFIG_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_default_service_config():
    """Write a starter config file if none exists."""
    if not os.path.exists(SERVICE_CONFIG_FILE):
        cfg = {
            "host": DEFAULT_HOST,
            "port": DEFAULT_PORT,
            "audio_output_path": None,
        }
        with open(SERVICE_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"Default config written to {SERVICE_CONFIG_FILE}")
        print("Edit it before starting the service.\n")


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
    # If the first argument is a Windows-service verb, delegate to pywin32.
    if len(sys.argv) > 1 and sys.argv[1].lower() in _SERVICE_COMMANDS:
        if not _HAS_SERVICE:
            print("pywin32 is required for Windows service support.")
            print("Install it with:  pip install pywin32")
            sys.exit(1)
        if sys.argv[1].lower() == "install":
            _save_default_service_config()
        win32serviceutil.HandleCommandLine(TranscriptionWindowsService)
    else:
        main()

#!/usr/bin/env python3
"""
Whisper Live Transcription Client
══════════════════════════════════
Captures audio from the sound card, streams it over TCP to the
transcription server, and displays the live transcription text.

Usage
─────
  python transcription_client.py [--host HOST] [--port PORT]
                                 [--device DEVICE_PREFIX]
                                 [--list-devices]
"""

import argparse
import json
import socket
import sys
import threading
import time
from typing import Optional

import colorama
import numpy as np
import pyaudio
from termcolor import colored

colorama.init()

from protocol import (
    DEFAULT_PORT,
    MSG_AUDIO,
    MSG_CONFIG,
    MSG_ERROR,
    MSG_TRANSCRIPTION,
    recv_message,
    send_message,
)

# ── Audio capture settings ────────────────────────────────────────────────────
STEP_IN_SEC = 1       # Seconds of audio per network message
NB_CHANNELS = 1
RATE = 16000
CHUNK = RATE          # Samples per read (= 1 second at 16 kHz)

# Default device name prefix to look for
DEFAULT_DEVICE_PREFIX = "Voicemeeter Out B1"


# ══════════════════════════════════════════════════════════════════════════════
class TranscriptionClient:
    """Captures audio, sends it to the server, and prints transcriptions."""

    def __init__(self, host: str, port: int, device_prefix: Optional[str] = None,
                 model: Optional[str] = None, language: Optional[str] = None):
        self.host = host
        self.port = port
        self.device_prefix = device_prefix or DEFAULT_DEVICE_PREFIX
        self.model = model or "turbo"
        self.language = language or "en"

        self.sock: Optional[socket.socket] = None
        self.running = False

    # ── Connection ────────────────────────────────────────────────────────
    def _connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        print(f"Connected to server at {self.host}:{self.port}")

        # Tell the server about our audio format
        config = {
            "sample_rate": RATE,
            "channels": NB_CHANNELS,
            "step_in_sec": STEP_IN_SEC,
            "model": self.model,
            "language": self.language,
        }
        send_message(self.sock, MSG_CONFIG, json.dumps(config).encode("utf-8"))

    # ── Main entry ────────────────────────────────────────────────────────
    def start(self):
        """Connect and start capture + display (blocks until stopped)."""
        self._connect()
        self.running = True

        # Receiver thread: reads transcriptions from server and prints them
        receiver = threading.Thread(target=self._receiver_thread, daemon=True)
        receiver.start()

        # Audio capture thread
        capture = threading.Thread(target=self._capture_thread, daemon=True)
        capture.start()

        # Main thread just waits
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            raise  # propagate to caller

    def stop(self):
        """Shut down cleanly."""
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass

    # ── Audio capture (runs in thread) ────────────────────────────────────
    def _capture_thread(self):
        assert self.sock is not None, "must call _connect() first"
        pa = pyaudio.PyAudio()
        device_index = self._find_device(pa)

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=NB_CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK,
        )

        print("Recording started …")
        print("-" * 60)

        try:
            while self.running:
                audio_data = b""
                for _ in range(STEP_IN_SEC):
                    try:
                        audio_data += stream.read(RATE, exception_on_overflow=False)
                    except OSError:
                        if not self.running:
                            return
                        raise
                    if not self.running:
                        return

                try:
                    send_message(self.sock, MSG_AUDIO, audio_data)
                except OSError:
                    print("\nLost connection to server.")
                    self.running = False
                    return
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    # ── Receiver (runs in thread) ─────────────────────────────────────────
    def _receiver_thread(self):
        """Read transcription messages from the server and display them."""
        assert self.sock is not None, "must call _connect() first"
        while self.running:
            msg_type, payload = recv_message(self.sock)
            if msg_type is None:
                if self.running:
                    print("\nServer disconnected.")
                self.running = False
                return

            if msg_type == MSG_ERROR and payload:
                try:
                    data = json.loads(payload.decode("utf-8"))
                    error = data.get("error", "Unknown error")
                    print(f"\n{colored('Server error:', 'red')} {error}")
                except json.JSONDecodeError:
                    print(f"\n{colored('Server error:', 'red')} (malformed)")
                self.running = False
                return

            if msg_type == MSG_TRANSCRIPTION and payload:
                try:
                    data = json.loads(payload.decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                text = data.get("text", "")
                is_final = data.get("is_final", False)

                if is_final:
                    print()  # newline — paragraph break
                else:
                    ts = time.strftime("%H:%M:%S", time.localtime())
                    print(colored(ts, "green") + " " + text, end="\r", flush=True)

    # ── Device selection ──────────────────────────────────────────────────
    def _find_device(self, pa: pyaudio.PyAudio) -> Optional[int]:
        """Find the audio input device whose name starts with *self.device_prefix*."""
        hostapi = pa.get_default_host_api_info()
        count = pa.get_device_count()

        for i in range(count):
            info = pa.get_device_info_by_index(i)
            if (
                int(info["maxInputChannels"]) > 0
                and info["hostApi"] == hostapi["index"]
                and str(info["name"]).startswith(self.device_prefix)
            ):
                print(f"Using input device #{i}: {info['name']}")
                print(f"  Channels: {info['maxInputChannels']}  "
                      f"Sample rate: {info['defaultSampleRate']}")
                return i

        # Fallback to system default
        print(f"Device '{self.device_prefix}' not found — using system default.")
        default = pa.get_default_input_device_info()
        print(f"  Default device: {default['name']}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════════════════════
def list_devices():
    """Print all input-capable audio devices and exit."""
    pa = pyaudio.PyAudio()
    print("Available input devices:\n")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if int(info["maxInputChannels"]) > 0:
            api = pa.get_host_api_info_by_index(int(info["hostApi"]))["name"]
            print(f"  [{i:>2}] {info['name']}")
            print(f"       API: {api}  |  Channels: {info['maxInputChannels']}  "
                  f"|  Rate: {int(info['defaultSampleRate'])} Hz")
    pa.terminate()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Whisper Live Transcription Client")
    parser.add_argument("--host", default="localhost", help="Server host (default: %(default)s)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port (default: %(default)s)")
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE_PREFIX,
        help="Audio input device name prefix (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="turbo",
        help="Whisper model name (default: %(default)s)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Transcription language code, e.g. en, cs, de (default: %(default)s)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    client = TranscriptionClient(
        host=args.host, port=args.port, device_prefix=args.device,
        model=args.model, language=args.language,
    )
    try:
        client.start()
    except KeyboardInterrupt:
        print("\n\nDisconnecting …")
    except ConnectionRefusedError:
        print(f"\nCannot connect to server at {args.host}:{args.port}")
        print("Is the server running?")
        sys.exit(1)
    finally:
        client.stop()


if __name__ == "__main__":
    main()

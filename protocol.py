"""
TCP protocol for audio streaming and transcription.

Message wire format
───────────────────
  [type: 1 byte] [length: 4 bytes big-endian] [payload: N bytes]

Message types
─────────────
  MSG_CONFIG (1)        Client → Server   JSON with audio parameters
  MSG_AUDIO  (2)        Client → Server   8-byte client timestamp (ms, big-endian int64)
                                          followed by raw PCM int16 audio data
  MSG_TRANSCRIPTION (3) Server → Client   JSON with transcription text
  MSG_STATS (4)         Server → Client   JSON with lag statistics
"""

import socket
import struct
from typing import Optional, Tuple

# ── Message types ─────────────────────────────────────────────────────────────
MSG_CONFIG = 1
MSG_AUDIO = 2
MSG_TRANSCRIPTION = 3
MSG_STATS = 4

# ── Network defaults ─────────────────────────────────────────────────────────
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 43007

# ── Framing ───────────────────────────────────────────────────────────────────
_HEADER_FMT = "!BI"  # unsigned char + unsigned int
HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 5 bytes

# ── Audio timestamp prefix ────────────────────────────────────────────────────
# Every MSG_AUDIO payload starts with an 8-byte big-endian int64 carrying the
# client's wall-clock time in milliseconds, followed by the raw PCM bytes.
AUDIO_TIMESTAMP_SIZE = 8
_AUDIO_TS_FMT = "!q"  # signed int64 big-endian


def pack_audio_payload(client_ts_ms: int, pcm_bytes: bytes) -> bytes:
    """Prepend the 8-byte timestamp header to a raw PCM chunk."""
    return struct.pack(_AUDIO_TS_FMT, client_ts_ms) + pcm_bytes


def unpack_audio_payload(payload: bytes) -> Tuple[int, bytes]:
    """Split timestamp and PCM from a MSG_AUDIO payload."""
    client_ts_ms = struct.unpack_from(_AUDIO_TS_FMT, payload, 0)[0]
    pcm_bytes = payload[AUDIO_TIMESTAMP_SIZE:]
    return client_ts_ms, pcm_bytes


def send_message(sock: socket.socket, msg_type: int, payload: bytes) -> None:
    """Send a length-prefixed message over a TCP socket."""
    header = struct.pack(_HEADER_FMT, msg_type, len(payload))
    sock.sendall(header + payload)


def recv_exactly(sock: socket.socket, n: int) -> Optional[bytes]:
    """
    Receive exactly *n* bytes from *sock*.

    Returns ``None`` when the peer disconnects or the socket is closed
    from another thread (any ``OSError`` is treated as a disconnect).
    """
    data = b""
    while len(data) < n:
        try:
            chunk = sock.recv(n - len(data))
        except OSError:
            return None
        if not chunk:
            return None
        data += chunk
    return data


def recv_message(sock: socket.socket) -> Tuple[Optional[int], Optional[bytes]]:
    """
    Receive one framed message.

    Returns ``(msg_type, payload)`` on success or ``(None, None)`` when
    the connection has been closed.
    """
    header = recv_exactly(sock, HEADER_SIZE)
    if header is None:
        return None, None
    msg_type, length = struct.unpack(_HEADER_FMT, header)
    if length == 0:
        return msg_type, b""
    payload = recv_exactly(sock, length)
    if payload is None:
        return None, None
    return msg_type, payload

"""
TCP protocol for audio streaming and transcription.

Message wire format
───────────────────
  [type: 1 byte] [length: 4 bytes big-endian] [payload: N bytes]

Message types
─────────────
  MSG_CONFIG (1)        Client → Server   JSON with audio parameters
  MSG_AUDIO  (2)        Client → Server   Raw PCM int16 audio data
  MSG_TRANSCRIPTION (3) Server → Client   JSON with transcription text
"""

import socket
import struct
from typing import Optional, Tuple

# ── Message types ─────────────────────────────────────────────────────────────
MSG_CONFIG = 1
MSG_AUDIO = 2
MSG_TRANSCRIPTION = 3

# ── Network defaults ─────────────────────────────────────────────────────────
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 43007

# ── Framing ───────────────────────────────────────────────────────────────────
_HEADER_FMT = "!BI"  # unsigned char + unsigned int
HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 5 bytes


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

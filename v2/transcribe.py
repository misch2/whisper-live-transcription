import time
from audio.circular_buffer import CircularBuffer
from audio.audio_stream import AudioStream
from transcriber import Transcriber

SAMPLE_RATE = 16000
BUFFER_SECONDS = 120
CHUNK_SECONDS = 30
CHANNELS = 1

# Whisper LLM parameters
WHISPER_MODEL = "turbo"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"
WHISPER_THREADS = 4
WHISPER_LANGUAGE = "en"


def main():
    print("\n====================================")
    print("  Whisper Live Transcription v2")
    print("====================================\n")
    buffer = CircularBuffer(BUFFER_SECONDS, SAMPLE_RATE)
    audio_thread = AudioStream(buffer, sample_rate=SAMPLE_RATE, chunk_size=SAMPLE_RATE, channels=CHANNELS)
    transcriber_thread = Transcriber(
        buffer,
        sample_rate=SAMPLE_RATE,
        chunk_seconds=CHUNK_SECONDS,
        whisper_model=WHISPER_MODEL,
        whisper_device=WHISPER_DEVICE,
        whisper_compute_type=WHISPER_COMPUTE_TYPE,
        whisper_threads=WHISPER_THREADS,
        language=WHISPER_LANGUAGE
    )

    # Display audio device info
    import pyaudio
    pa = pyaudio.PyAudio()
    default_device_index = pa.get_default_input_device_info()["index"]
    default_device_info = pa.get_device_info_by_index(default_device_index)
    print("Selected audio device:")
    print(f"  Index: {default_device_index}")
    print(f"  Name: {default_device_info['name']}")
    # print(f"  Channels: {CHANNELS}")
    # print(f"  Default Sample Rate: {default_device_info['defaultSampleRate']}")
    # print()
    # print("Sound sampling parameters:")
    # print(f"  SAMPLE_RATE: {SAMPLE_RATE}")
    # print(f"  BUFFER_SECONDS: {BUFFER_SECONDS}")
    # print(f"  CHUNK_SECONDS (transcription window): {CHUNK_SECONDS}")
    print()
    print("Whisper LLM parameters:")
    print(f"  Model: {WHISPER_MODEL}")
    print(f"  Device: {WHISPER_DEVICE}")
    print(f"  Compute type: {WHISPER_COMPUTE_TYPE}")
    print(f"  Threads: {WHISPER_THREADS}")
    print(f"  Language: {WHISPER_LANGUAGE}")
    print()
    pa.terminate()

    audio_thread.start()
    transcriber_thread.start()

    print("Live transcription started. Press Ctrl+C to stop.")
    try:
        last_output = ""
        last_metadata = None
        while True:
            latest, metadata = transcriber_thread.get_latest()
            lag = buffer.get_lag()
            if latest and latest != last_output:
                ts = metadata["timestamp"] if metadata else ""
                print(f"[{ts}] (lag: {lag:.2f}s) {latest}")
                last_output = latest
                last_metadata = metadata
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping...")
        audio_thread.stop()
        transcriber_thread.stop()
        audio_thread.join()
        transcriber_thread.join()

if __name__ == "__main__":
    main()

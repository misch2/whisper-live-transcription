import time
from audio.circular_buffer import CircularBuffer
from audio.audio_stream import AudioStream
from transcriber import Transcriber

SAMPLE_RATE = 16000
BUFFER_SECONDS = 120
CHUNK_SECONDS = 30
CHANNELS = 1


def main():
    buffer = CircularBuffer(BUFFER_SECONDS, SAMPLE_RATE)
    # Get default audio device info
    import pyaudio
    pa = pyaudio.PyAudio()
    default_device_info = pa.get_default_input_device_info()
    device_index = default_device_info["index"]
    device_name = default_device_info["name"]
    device_channels = default_device_info["maxInputChannels"]
    device_rate = default_device_info["defaultSampleRate"]
    pa.terminate()

    print("Selected audio device:")
    print(f"  Index: {device_index}")
    print(f"  Name: {device_name}")
    print(f"  Channels: {device_channels}")
    print(f"  Default Sample Rate: {device_rate}")

    print("Sound sampling parameters:")
    print(f"  Sample Rate: {SAMPLE_RATE}")
    print(f"  Buffer Duration: {BUFFER_SECONDS} seconds")
    print(f"  Chunk Duration: {CHUNK_SECONDS} seconds")
    print(f"  Channels: {CHANNELS}")

    transcriber_thread = Transcriber(buffer, sample_rate=SAMPLE_RATE, chunk_seconds=CHUNK_SECONDS)
    print("Whisper LLM parameters:")
    print(f"  Model: {transcriber_thread.model.model_name}")
    print(f"  Device: {transcriber_thread.model.device}")
    print(f"  Compute Type: {transcriber_thread.model.compute_type}")
    print(f"  Threads: {transcriber_thread.model.cpu_threads}")
    print(f"  Language: {transcriber_thread.language}")

    audio_thread = AudioStream(buffer, device_index=device_index, sample_rate=SAMPLE_RATE, chunk_size=SAMPLE_RATE, channels=CHANNELS)
    audio_thread.start()
    transcriber_thread.start()

    print("Live transcription started. Press Ctrl+C to stop.")
    try:
        last_output = ""
        while True:
            latest = transcriber_thread.get_latest()
            if latest and latest != last_output:
                print(latest)
                last_output = latest
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping...")
        audio_thread.stop()
        transcriber_thread.stop()
        audio_thread.join()
        transcriber_thread.join()

if __name__ == "__main__":
    main()

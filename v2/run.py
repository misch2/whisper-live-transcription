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
    audio_thread = AudioStream(buffer, sample_rate=SAMPLE_RATE, chunk_size=SAMPLE_RATE, channels=CHANNELS)
    transcriber_thread = Transcriber(buffer, sample_rate=SAMPLE_RATE, chunk_seconds=CHUNK_SECONDS)

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

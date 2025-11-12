import time
from audio.circular_buffer import CircularBuffer
from audio.audio_stream import AudioStream
from transcriber import Transcriber

SAMPLE_RATE = 16000
RECORDING_CHUNK_SIZE = int(SAMPLE_RATE / 4)
BUFFER_SECONDS = 120
CHANNELS = 1

# Whisper LLM parameters
CHUNK_SECONDS = 30
WHISPER_MODEL = "turbo"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"
WHISPER_THREADS = 4
WHISPER_LANGUAGE = "en"

AUDIO_DEVICE_NAME_STARTS_WITH = "Voicemeeter Out B1"

def main():
    print("\n====================================")
    print("  Whisper Live Transcription v2")
    print("====================================\n")
    buffer = CircularBuffer(BUFFER_SECONDS, SAMPLE_RATE)

    import pyaudio
    pa = pyaudio.PyAudio()
    voicemeeter_index = None
    voicemeeter_info = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0 and info["name"].startswith(AUDIO_DEVICE_NAME_STARTS_WITH):
            voicemeeter_index = i
            voicemeeter_info = info
            break
    if voicemeeter_index is not None:
        print("Selected audio device:")
        print(f"  Index: {voicemeeter_index}")
        print(f"  Name: {voicemeeter_info['name']}")
        print()
    else:
        print("Voicemeeter Out B1 device not found. Using default input device.")
        voicemeeter_index = pa.get_default_input_device_info()["index"]
        voicemeeter_info = pa.get_device_info_by_index(voicemeeter_index)
        print(f"  Index: {voicemeeter_index}")
        print(f"  Name: {voicemeeter_info['name']}")
        print()

    print("Initializing audio stream...")
    audio_thread = AudioStream(buffer, device_index=voicemeeter_index, sample_rate=SAMPLE_RATE, chunk_size=RECORDING_CHUNK_SIZE, channels=CHANNELS)
    print("Initializing transcriber...")
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
        import matcher
        history = []
        while True:
            latest = transcriber_thread.get_latest()
            result = matcher.match_transcription(latest, history, buffer, SAMPLE_RATE)
            for line in result["lines"]:
                print(f"[{result['timestamp']} {result['lag']:.2f}s] {line.ljust(80)}")
            if latest:
                if not history or latest["text"] != history[-1]["text"]:
                    history.append(latest)
                if result["sentence_end"]:
                    history = []
            # time.sleep(0.5)
            # time.sleep(5)   # FIXME test only
    except KeyboardInterrupt:
        print("Stopping...")
        audio_thread.stop()
        transcriber_thread.stop()
        audio_thread.join()
        transcriber_thread.join()

if __name__ == "__main__":
    main()

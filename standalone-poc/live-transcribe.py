import queue
import re
import threading
import time
import os
from typing import Dict, List

import numpy as np
import pyaudio
from faster_whisper import WhisperModel

import keyboard
from termcolor import colored

# Yeah I could do this config with argparse, but I won't...

# Audio settings
STEP_IN_SEC: int = 1    # We'll increase the processable audio data by this
LENGHT_IN_SEC: int = 6    # We'll process this amount of audio data together maximum
NB_CHANNELS = 1
RATE = 16000
CHUNK = RATE

# Whisper settings
WHISPER_LANGUAGE = "en"
WHISPER_THREADS = 4
WHISPER_MODEL = "turbo"    # "base", "small", "medium", "large", "large-v1", "large-v2", "tiny", "tiny.en", "base.en", "small.en", "medium.en", "large-v1.en", "large-v2.en", "turbo"
#WHISPER_MODEL = "tiny"  # test only
WHISPER_COMPUTE_TYPE = "float16"    # "int8", "float16", "float32", "int4", "int4_float16", "int4_float32"

# Visualization (expected max number of characters for LENGHT_IN_SEC audio)
MAX_SENTENCE_CHARACTERS = 80

# import os
# os.add_dll_directory("c:\\Program Files\\NVIDIA\\CUDNN\\v9.8\\bin\\12.8")

# print("CUDA_VISIBLE_DEVICES: ", os.environ.get("CUDA_VISIBLE_DEVICES"))

# This queue holds all the 1-second audio chunks
audio_queue = queue.Queue()

# This queue holds all the chunks that will be processed together
# If the chunk is filled to the max, it will be emptied
length_queue = queue.Queue(maxsize=LENGHT_IN_SEC)

# Get user home directory
home_dir = os.path.expanduser("~")

models_dir = os.path.join(home_dir, ".cache", "huggingface", "hub")

print("Initializing Whisper model '%s' with %d threads and compute type '%s'..." % (WHISPER_MODEL, WHISPER_THREADS, WHISPER_COMPUTE_TYPE))
print("Models will be downloaded to: %s" % models_dir)
# Whisper model
# whisper = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=WHISPER_THREADS, download_root=models_dir)
# whisper = WhisperModel("turbo", device="cuda", compute_type="float16", cpu_threads=WHISPER_THREADS, download_root=models_dir)
whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type=WHISPER_COMPUTE_TYPE, cpu_threads=WHISPER_THREADS, download_root=models_dir)

print("Whisper model initialized")

# Global flag to signal threads to stop
stop_threads = False

def producer_thread():
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=NB_CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,    # 1 second of audio
    )

    print("-" * 80)
    print("Microphone initialized, recording started...")
    print("-" * 80)
    print("TRANSCRIPTION using '%s' model" % WHISPER_MODEL)
    print("-" * 80)

    global stop_threads
    while not stop_threads:
        audio_data = b""
        for _ in range(STEP_IN_SEC):
            chunk = stream.read(RATE)    # Read 1 second of audio data
            audio_data += chunk
            if stop_threads:
                break

        audio_queue.put(audio_data)    # Put the 5-second audio data into the queue
        # print('(audio)', end='', flush=True)

    print("\nStopping audio producer thread...")

# Thread which gets items from the queue and prints its length
def consumer_thread(stats):
    global stop_threads
    while not stop_threads:
        if length_queue.qsize() >= LENGHT_IN_SEC:
            with length_queue.mutex:
                length_queue.queue.clear()
                print()

        audio_data = audio_queue.get()
        if stop_threads:
            break
        
        transcription_start_time = time.time()
        length_queue.put(audio_data)

        # Concatenate audio data in the lenght_queue
        audio_data_to_process = b""
        for i in range(length_queue.qsize()):
            # We index it so it won't get removed
            audio_data_to_process += length_queue.queue[i]

        # convert the bytes data toa  numpy array
        audio_data_array: np.ndarray = np.frombuffer(audio_data_to_process, np.int16).astype(np.float32) / 255.0
        # audio_data_array = np.expand_dims(audio_data_array, axis=0)

        segment_length_seconds = (audio_data_array.shape[0] / RATE)
        # print("transcribing %d seconds of audio data" % segment_length_seconds)
        segments, _ = whisper.transcribe(audio_data_array,
                                         language=WHISPER_LANGUAGE,
                                         beam_size=5,
                                         vad_filter=True,
                                         vad_parameters=dict(min_silence_duration_ms=1000))
        segments = [s.text for s in segments]

        transcription_end_time = time.time()

        # print("Raw segments in %d sec segment: %s", segment_length_seconds, segments)

        transcription = " ".join(segments)

        # print(colored("Raw segments in %d sec segment: %s", "grey"), segment_length_seconds, transcription, end='\r', flush=True)


        # remove anything from the text which is between () or [] --> these are non-verbal background noises/music/etc.
        # transcription = re.sub(r"\[.*\]", "", transcription)
        # transcription = re.sub(r"\(.*\)", "", transcription)
        # We do this for the more clean visualization (when the next transcription we print would be shorter then the one we printed)
        transcription = transcription.ljust(MAX_SENTENCE_CHARACTERS, " ")

        transcription_postprocessing_end_time = time.time()

        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(colored(current_time, "green") + " " + transcription, end='\r', flush=True)

        audio_queue.task_done()

        overall_elapsed_time = transcription_postprocessing_end_time - transcription_start_time
        transcription_elapsed_time = transcription_end_time - transcription_start_time
        postprocessing_elapsed_time = transcription_postprocessing_end_time - transcription_end_time
        stats["overall"].append(overall_elapsed_time)
        stats["transcription"].append(transcription_elapsed_time)
        stats["postprocessing"].append(postprocessing_elapsed_time)

    print("\nStopping consumer thread...")

if __name__ == "__main__":
    stats: Dict[str, List[float]] = {"overall": [], "transcription": [], "postprocessing": []}

    producer = threading.Thread(target=producer_thread)
    producer.start()

    consumer = threading.Thread(target=consumer_thread, args=(stats,))
    consumer.start()


    abort_key = ''; #'q'
    # print("Press q or Esc to exit")
    try:
        # Wait for 'q' to be pressed
        print("Press '%s' or ^C to stop the threads." % abort_key)
        if (abort_key == ''):
            while not stop_threads:
                time.sleep(0.1)
        else:
            keyboard.wait(abort_key)
            print("Exiting due to abort key (%s) pressed" % abort_key)
            stop_threads = True
            
    except KeyboardInterrupt as e:
        print("Exiting due to keyboard interrupt: %s" % str(e))
        stop_threads = True

    # print out the statistics
    print("Number of processed chunks: ", len(stats["overall"]))
    print(f"Overall time: avg: {np.mean(stats['overall']):.4f}s, std: {np.std(stats['overall']):.4f}s")
    print(
        f"Transcription time: avg: {np.mean(stats['transcription']):.4f}s, std: {np.std(stats['transcription']):.4f}s"
    )
    print(
        f"Postprocessing time: avg: {np.mean(stats['postprocessing']):.4f}s, std: {np.std(stats['postprocessing']):.4f}s"
    )
    # We need to add the step_in_sec to the latency as we need to wait for that chunk of audio
    print(f"The average latency is {np.mean(stats['overall'])+STEP_IN_SEC:.4f}s")

    # Wait for all threads to finish
    producer.join()
    consumer.join()

    print("All threads stopped, exiting...")
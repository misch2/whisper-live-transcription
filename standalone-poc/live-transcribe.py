import queue
import re
import threading
import time
import os
import argparse
from datetime import datetime
from typing import Dict, List

import numpy as np
import pyaudio
from faster_whisper import WhisperModel

import keyboard
from termcolor import colored
from wave_recorder import WaveRecorder

# Audio settings
STEP_IN_SEC: int = 1  # We'll increase the processable audio data by this
LENGHT_IN_SEC: int = 6  # We'll process this amount of audio data together maximum
NB_CHANNELS = 1
RATE = 16000
CHUNK = RATE

# Whisper settings
WHISPER_LANGUAGE = "en"
WHISPER_THREADS = 4
WHISPER_MODEL = "turbo"  # "base", "small", "medium", "large", "large-v1", "large-v2", "tiny", "tiny.en", "base.en", "small.en", "medium.en", "large-v1.en", "large-v2.en", "turbo"
# WHISPER_MODEL = "tiny"  # test only
WHISPER_COMPUTE_TYPE = (
    "float16"  # "int8", "float16", "float32", "int4", "int4_float16", "int4_float32"
)

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

print(
    "Initializing Whisper model '%s' with %d threads and compute type '%s'..."
    % (WHISPER_MODEL, WHISPER_THREADS, WHISPER_COMPUTE_TYPE)
)
print("Models will be downloaded to: %s" % models_dir)
# Whisper model
# whisper = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=WHISPER_THREADS, download_root=models_dir)
# whisper = WhisperModel("turbo", device="cuda", compute_type="float16", cpu_threads=WHISPER_THREADS, download_root=models_dir)
whisper = WhisperModel(
    WHISPER_MODEL,
    device="cuda",
    compute_type=WHISPER_COMPUTE_TYPE,
    cpu_threads=WHISPER_THREADS,
    download_root=models_dir,
)

print("Whisper model initialized")

# Global flag to signal threads to stop
stop_threads = False

# Global variable for audio recording
wave_recorder = WaveRecorder(sample_rate=RATE, channels=NB_CHANNELS)


def producer_thread(save_audio_path=None):
    audio = pyaudio.PyAudio()
    global wave_recorder

    # Initialize WAV saving if path is provided
    if save_audio_path:
        wave_recorder.set_output_path(save_audio_path)

    # list_pyaudio_devices(audio)

    print("-" * 80)

    # Get the default input device index
    microphone_device_index = get_virtual_audio_mix_device_index(audio)

    stream = audio.open(
        format=pyaudio.paInt16,
        channels=NB_CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=microphone_device_index,
        frames_per_buffer=CHUNK,  # 1 second of audio
    )

    print("Microphone initialized, recording started...")
    print("-" * 80)
    print("TRANSCRIPTION using '%s' model" % WHISPER_MODEL)
    print("-" * 80)

    global stop_threads
    while not stop_threads:
        audio_data = b""
        for _ in range(STEP_IN_SEC):
            chunk = stream.read(RATE)  # Read 1 second of audio data
            audio_data += chunk
            if stop_threads:
                break

        audio_queue.put(audio_data)  # Put the audio data into the queue for transcription
        
        # If saving is enabled, collect audio data in memory
        if save_audio_path:
            wave_recorder.add_audio_chunk(audio_data)

    # Close the audio stream
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Close WAV file if it was being written to
    if save_audio_path:
        wave_recorder.close()
    
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
        audio_data_array: np.ndarray = (
            np.frombuffer(audio_data_to_process, np.int16).astype(np.float32) / 255.0
        )
        # audio_data_array = np.expand_dims(audio_data_array, axis=0)

        segment_length_seconds = audio_data_array.shape[0] / RATE
        # print("transcribing %d seconds of audio data" % segment_length_seconds)
        segments, _ = whisper.transcribe(
            audio_data_array,
            language=WHISPER_LANGUAGE,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=1000),
        )
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
        print(
            colored(current_time, "green") + " " + transcription, end="\r", flush=True
        )

        audio_queue.task_done()

        overall_elapsed_time = (
            transcription_postprocessing_end_time - transcription_start_time
        )
        transcription_elapsed_time = transcription_end_time - transcription_start_time
        postprocessing_elapsed_time = (
            transcription_postprocessing_end_time - transcription_end_time
        )
        stats["overall"].append(overall_elapsed_time)
        stats["transcription"].append(transcription_elapsed_time)
        stats["postprocessing"].append(postprocessing_elapsed_time)

    print("\nStopping consumer thread...")


def get_virtual_audio_mix_device_index(p):
    hostapi = p.get_default_host_api_info()

    device_count = p.get_device_count()

    index = 0
    for i in range(device_count):
        # Get device info
        device_info = p.get_device_info_by_index(i)
        #print(device_info)

        # Check if the device supports input
        if device_info["maxInputChannels"] > 0 and device_info["hostApi"] == hostapi["index"] and device_info["name"].startswith("Voicemeeter Out B1"):
        #if device_info["maxInputChannels"] > 0 and device_info["name"].startswith("Voicemeeter Out B1"):
            print(f"Using default input device index: {i}")
            print(f"  Name: {device_info['name']}")
            print(f"  Max Input Channels: {device_info['maxInputChannels']}")
            print(f"  Default Sample Rate: {device_info['defaultSampleRate']}")
            index = i
            break

    return index


def finalize_audio_saving(output_path):
    """Close the WAV file (audio has been written continuously)"""
    global wave_recorder
    
    wave_recorder.save_to_file()

def list_pyaudio_devices(p):
    hostapi_count = p.get_host_api_count()
    print("Host API count: %d" % hostapi_count)
    for i in range(hostapi_count):
        # Get host API info
        hostapi_info = p.get_host_api_info_by_index(i)
        print(f"Host API ID: {i}")
        print(f"  Name: {hostapi_info['name']}")
        print(f"  Device count: {hostapi_info['deviceCount']}")
        print(f"  Default input device: {hostapi_info['defaultInputDevice']}")
        print(f"  Default output device: {hostapi_info['defaultOutputDevice']}")
        print(hostapi_info)
        print()

    print("Default Host API: %s" % p.get_default_host_api_info())
    print("Default Input Device: %s" % p.get_default_input_device_info())

    device_count = p.get_device_count()

    for i in range(device_count):
        # Get device info
        device_info = p.get_device_info_by_index(i)

        # Check if the device supports input
        # if device_info["maxInputChannels"] > 0:
        if device_info["maxInputChannels"] == 1:
            print(f"Device ID: {i}")
            print(f"  Name: {device_info['name']}")
            print(f"  Max Input Channels: {device_info['maxInputChannels']}")
            print(f"  Default Sample Rate: {device_info['defaultSampleRate']}")

            print(device_info)
            print()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Live audio transcription using Whisper')
    parser.add_argument('--audio-output-path', type=str, help='Folder path to save audio output file with datetime-based name')
    args = parser.parse_args()

    # Generate datetime-based filename if output path is provided
    audio_file_path = None
    if args.audio_output_path:
        # Ensure the directory exists
        os.makedirs(args.audio_output_path, exist_ok=True)
        
        # Generate datetime-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        audio_file_path = os.path.join(args.audio_output_path, filename)
        
        # print(f"Audio will be saved to: {audio_file_path}")

    stats: Dict[str, List[float]] = {
        "overall": [],
        "transcription": [],
        "postprocessing": [],
    }

    producer = threading.Thread(target=producer_thread, args=(audio_file_path,))
    producer.start()

    consumer = threading.Thread(target=consumer_thread, args=(stats,))
    consumer.start()

    abort_key = ""
    #'q'
    # print("Press q or Esc to exit")
    try:
        # Wait for 'q' to be pressed
        print("Press '%s' or ^C to stop the threads." % abort_key)
        if abort_key == "":
            while not stop_threads:
                time.sleep(0.1)
        else:
            keyboard.wait(abort_key)
            print("Exiting due to abort key (%s) pressed" % abort_key)
            stop_threads = True

    except KeyboardInterrupt as e:
        print("Exiting due to keyboard interrupt: %s" % str(e))
        stop_threads = True
        # Ensure WAV file is properly closed
        if audio_file_path:
            wave_recorder.close()

    # print out the statistics
    print("Number of processed chunks: ", len(stats["overall"]))
    print(
        f"Overall time: avg: {np.mean(stats['overall']):.4f}s, std: {np.std(stats['overall']):.4f}s"
    )
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

    # Finalize audio saving if path was provided
    if audio_file_path:
        print("Finalizing audio file...")
        finalize_audio_saving(audio_file_path)

    print("All threads stopped, exiting...")

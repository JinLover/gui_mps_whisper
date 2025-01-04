#! python3.7

import sounddevice
import argparse
import numpy as np
import speech_recognition as sr
import whisper
import wave

from queue import Queue
from time import sleep
from sys import platform

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="turbo", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large", "turbo"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()
    return args

def gen_source():
    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)
    return source

def load_model(args):
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    return whisper.load_model(model, device="mps")

def main():
    # Parse command line arguments.
    args = parse_args()
    # Load / Download model
    model = load_model(args)
    print("Model loaded.\n")
    # Load audio source
    source = gen_source()
    if not source:
        return
    # Start listening
    listening(args, source, model)


def listening(args, source, audio_model):
    phrase_time = None
    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout
    sample_rate = 16000

    # 오디오 파일 저장
    audio_file_path = "live_audio.wav"
    wf = wave.open(audio_file_path, 'wb')
    wf.setnchannels(1)  # 모노 채널
    wf.setsampwidth(2)  # 16비트 오디오
    wf.setframerate(sample_rate)  # 샘플링 레이트

    full_transcription = ""

    def record_callback(_, audio: sr.AudioData) -> None:
        """Callback function to receive audio data."""
        data = audio.get_raw_data()
        data_queue.put(data)
        wf.writeframes(data)  # 모든 데이터를 파일에 저장

    with source:
        recorder.adjust_for_ambient_noise(source)
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    try:
        while True:
            if not data_queue.empty():
                # Queue에서 오디오 데이터를 가져와 변환
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()

                if audio_data:
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                    result = audio_model.transcribe(audio_np, fp16=True)
                    text = result['text'].strip()

                    full_transcription += text
                    print(text, end=' ')
            else:
                sleep(0.1)

    except KeyboardInterrupt:
        print("\nListening stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        wf.close()
        print("\nFinal Transcription:")
        print(full_transcription)

if __name__ == "__main__":
    main()
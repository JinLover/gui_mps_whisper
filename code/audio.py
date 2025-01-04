import sounddevice
import pyaudio
import whisper
import numpy as np
import torch
import wave

# PyAudio settings
RATE = 16000  # Sampling rate
CHUNK = 16000  # Buffer size
FORMAT = pyaudio.paInt16  # Sample format
CHANNELS = 1  # Mono audio
DEVICE = torch.device("mps")

# Load Whisper model
options = whisper.DecodingOptions()
model = whisper.load_model("tiny", device=DEVICE)
print("Model loaded.")

# Create PyAudio object
p = pyaudio.PyAudio()

# Function to listen to real-time audio and transcribe to text
def listen_and_transcribe():
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Listening... (Press Ctrl+C to stop)")

    # Read audio buffer
    audio_data = np.frombuffer(stream.read(CHUNK*30), dtype=np.int16)
    
    # Convert numpy array to writable float32 type
    audio_data = audio_data.astype(np.float32)

    # Convert audio data to format suitable for Whisper model
    # audio_tensor = torch.from_numpy(audio_data).to(DEVICE)
    # Generate Mel spectrogram
    mel = whisper.log_mel_spectrogram(audio_data, n_mels=model.dims.n_mels).to(DEVICE)
    print(type(mel), mel.shape)

    # Perform speech recognition
    result = model.decode(mel)

    # Print the transcribed text
    print(result.text)
    # while True:
    #     # Read audio buffer
    #     audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        
    #     # Convert numpy array to writable float32 type
    #     audio_data = audio_data.astype(np.float32)

    #     # Convert audio data to format suitable for Whisper model
    #     # audio_tensor = torch.from_numpy(audio_data).to(DEVICE)
    #     # Generate Mel spectrogram
    #     mel = whisper.log_mel_spectrogram(audio_data, n_mels=model.dims.n_mels).to(DEVICE)
    #     print(type(mel), mel.shape)

    #     # Perform speech recognition
    #     result = model.decode(mel)

    #     # Print the transcribed text
    #     print(result.text)

# Run the program
try:
    listen_and_transcribe()
except KeyboardInterrupt:
    print("\nProgram terminated.")
finally:
    p.terminate()
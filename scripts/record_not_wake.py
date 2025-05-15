import os
import sounddevice as sd
import scipy.io.wavfile as wav

SAMPLING_RATE = 16000
DURATION = 1.0
SAVE_DIR = "data/raw/not_wake_word"
os.makedirs(SAVE_DIR, exist_ok=True)

def record_sample(path: str):
    print(f"Recording not-wake -> {path}")
    audio = sd.rec(int(SAMPLING_RATE * DURATION), samplerate=SAMPLING_RATE, channels=1, dtype='int16')
    sd.wait()
    wav.write(path, SAMPLING_RATE, audio)
    print("Saved!")

if __name__ == "__main__":
    for i in range(40):
        fname = f"noise_{i+1:03d}.wav"
        record_sample(os.path.join(SAVE_DIR, fname))

import argparse
import sys
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from collections import deque
import torch

from src.model import TinyWakeWordCNN
from src.utils import compute_mel_spectrogram, create_tensor

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    model = TinyWakeWordCNN().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    FRAME_SIZE  = int(1.0 * 16000)
    STRIDE_SIZE = int(0.2 * 16000)
    audio_buffer = deque(maxlen=FRAME_SIZE)
    stream = sd.InputStream(callback=lambda indata, frames, time, status: audio_buffer.extend(indata[:,0]),
                            channels=1, samplerate=16000, blocksize=STRIDE_SIZE)

    plt.ion()
    fig, ax = plt.subplots()

    with stream:
        while True:
            if len(audio_buffer) < FRAME_SIZE:
                continue

            buffer = np.array(audio_buffer)
            mel = compute_mel_spectrogram(buffer)
            mel_tensor = create_tensor(mel, device)

            with torch.no_grad():
                prob = model(mel_tensor).item()

            bar_len = 40
            hashtags = "#" * round(prob * bar_len)
            spaces = " " * (bar_len - len(hashtags))
            sys.stdout.write(f"\r[{hashtags}{spaces}] {prob:.2f}")
            sys.stdout.flush()
            plt.pause(0.01)

if __name__ == "__main__":
    main()

import numpy as np
import librosa
import torch

SAMPLING_RATE = 16000
MEL_WINDOW_MS = 30
MEL_HOP_MS = 10
N_MELS = 40
N_FFT = 512
WIN_LENGTH = int(SAMPLING_RATE * MEL_WINDOW_MS / 1000) 
HOP_LENGTH = int(SAMPLING_RATE * MEL_HOP_MS / 1000) 
TARGET_FRAMES = 100

def compute_mel_spectrogram(audio):
    needed = TARGET_FRAMES * HOP_LENGTH + WIN_LENGTH
    if len(audio) < needed:
        audio = np.pad(audio, (0, needed - len(audio)), mode='constant')

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLING_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        center=False
    )
    return librosa.power_to_db(mel, ref=np.max)

def create_tensor(mel_spec, device):
    # mel_spec: (bins, frames) -> Tensor shape (1, 1, frames, bins)
    mel_tensor = torch.tensor(mel_spec.T, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return mel_tensor.to(device)
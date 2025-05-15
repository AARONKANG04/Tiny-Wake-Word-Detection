import os
import numpy as np
import librosa

SOURCE_WAKE_DIR = "data/raw/wake_word"
SOURCE_NOT_WAKE_DIR = "data/raw/not_wake_word"

WAKE_OUT_DIR = "data/processed/wake_word_mel"
NOT_WAKE_OUT_DIR = "data/processed/not_wake_word_mel"

os.makedirs(WAKE_OUT_DIR, exist_ok=True)
os.makedirs(NOT_WAKE_OUT_DIR, exist_ok=True)

SAMPLING_RATE = 16000
MEL_WINDOW_MS = 30
MEL_HOP_MS = 10
N_MELS = 40
N_FFT = 512
WIN_LENGTH = int(SAMPLING_RATE * MEL_WINDOW_MS / 1000)
HOP_LENGTH = int(SAMPLING_RATE * MEL_HOP_MS / 1000)
TARGET_FRAMES = 100

for src_dir, out_dir in [(SOURCE_WAKE_DIR, WAKE_OUT_DIR), (SOURCE_NOT_WAKE_DIR, NOT_WAKE_OUT_DIR)]:
    for fname in os.listdir(src_dir):
        if not fname.endswith(".wav"):
            continue

        path = os.path.join(src_dir, fname)
        y, sr = librosa.load(path, sr=SAMPLING_RATE)
        needed_frames = TARGET_FRAMES * HOP_LENGTH + WIN_LENGTH
        if len(y) < needed_frames:
            y = np.pad(y, (0, needed_frames - len(y)), mode="constant")

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
            center=False
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        base = os.path.splitext(fname)[0] + ".npy"
        out_path = os.path.join(out_dir, base)
        np.save(out_path, mel_db)
        print(f"Saved: {out_path}")

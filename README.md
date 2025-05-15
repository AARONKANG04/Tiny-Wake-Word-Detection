# Tiny Wake-Word Detector!

A lightweight, real-time wake-word detection system built with PyTorch. This project uses mel-spectrogram features and supports both CNN and LSTM models for efficient audio classification. Designed to work with low-resource audio clips (1 second), it enables hands-free interaction on edge devices or desktop systems.

---

## Project Structure

```
tiny-wake-word-detector/
├── scripts/             # Data collection and preprocessing tools
│   ├── record_wake.py
│   ├── record_not_wake.py
│   └── preprocess_audio.py
│
├── src/                 # Core training, inference, and model code
│   ├── model.py
│   ├── dataset.py
│   ├── utils.py
│   ├── train.py
│   └── inference.py
│
├── data/                # Audio data (not tracked in the repo -> must be created upon cloning the repository)
│   ├── raw/             # Raw .wav recordings
│   └── processed/       # Precomputed .npy mel-spectrogram data
│
├── experiments/         # Saved model checkpoints
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Record Samples

Record 1-second clips for your wake word and non wake words (background noise, background chatter, unrelated conversation, etc.):

```bash
python scripts/record_wake.py
python scripts/record_not_wake.py
```

### 3. Preprocess to Mel-Spectrograms
```bash
python scripts/preprocess_audio.py
```

This will save `.npy` spectrograms to `data/processed/`.

### 4. Train the Model
```bash
python -m src.train \
  --wake-dir data/processed/wake_word_mel \
  --not-wake-dir data/processed/not_wake_word_mel \
  --epochs 50 \
  --checkpoint experiments/cnn_model.pt
```

### 5. Run Real-Time Detection
```bash
python -m src.inference --model-path experiments/cnn_model.pt
```

You’ll see a live bar in your terminal showing detection confidence:
```
[############                  ] 0.34
```

---

## Model Architectures

- **CNN:** For fast 2D pattern recognition on mel-spectrograms.
- **LSTM:** (optional) For sequential modeling over time-frequency input.

You can switch models by modifying `train.py` and `inference.py`.

---

## Requirements

- Python 3.9+
- PyTorch
- librosa
- sounddevice
- matplotlib
- numpy
- scipy

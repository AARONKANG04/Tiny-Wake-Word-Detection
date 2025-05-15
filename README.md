# Tiny Wake-Word Detector!

A lightweight, real-time wake-word detection system built with PyTorch. 
This project processes 1-second audio snippets by converting them into mel-spectrograms, then classifies each as either a "wake word" or "non wake word" using a CNN or LSTM model. With fewer than a thousand parameters, the system is optimized for fast inference and minimal memory usage, making it ideal for deployment on edge devices and microcontrollers. 

---

## Project Structure

```
tiny-wake-word-detector/
├── scripts/             # Data collection and preprocessing
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
├── data/                # Audio data (not tracked by Git)
│   ├── raw/             # Raw .wav recordings
│   └── processed/       # Precomputed .npy mel-spectrograms
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

### 2. Record Training Samples

Record 1-second clips for your wake word and background data (e.g. speech, noise, silence):

```bash
python scripts/record_wake.py
python scripts/record_not_wake.py
```

### 3. Convert to Mel-Spectrograms
```bash
python scripts/preprocess_audio.py
```

This saves `.npy` spectrograms to `data/processed/`.

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

Edit `train.py` and `inference.py` to switch architectures.

---

## Requirements

- Python 3.9+
- PyTorch
- librosa
- sounddevice
- matplotlib
- numpy
- scipy

---

## Scientific and Mathematical Foundations
### Sound as Data
Sound is a pressure wave that travels through the air and reaches our ears. When you speak, for example, your vocal cords create vibrations that push air pressure at discrete time steps. Since we’re sampling this signal digitally, we typically get values at a fixed rate. In our implementation, we sample at a rate of 16 kHz (16,000 samples per second). This type of signal is known as a time-domain signal because it shows how pressure changes over time. But here’s the problem - a waveform isn’t very informative by itself. It’s just thousands and thousands of numbers. You can’t easily tell what pitches or rhythms are present. It’s like trying to understand a song by staring at the spreadsheet full of numbers. Your machine learning model will face the same problem as you and won’t do any better.

### From Time to Frequency
What we actually perceive as sound (e.g. pitch, tone, timbre) comes from frequencies, not just raw pressure changes. Every sound is made up of a combination of many frequencies at once. For example, when you say “hi,” the sound isn’t just one frequency, it’s a blend of many that change over time. If we can isolate these frequency components, we get a much more useful representation of the sound. This is where we turn to a powerful tool from math and signal processing - the Fourier Transform. 

### The Short-Time Fourier Transform (STFT)
The Fourier Transform decomposes a time-domain signal into its component frequencies, like figuring out what notes are being played in a chord. But it only gives you the full "ingredients list" for the entire signal - not when each frequency occurs. We want to know which frequencies are present at what times. That's where the Short-Time Fourier Transform (STFT) comes in. STFT slices the audio into short overlapping windows (25 milliseconds in this implementation) and applies the Fourier Transform to each window. This gives us a snapshot of the frequency information in each window of time. So instead of one big frequency summary, we now have a time-frequency map. Once we apply the STFT, we end up with a 2D matrix where:

- X-axis -> time
- Y-axis -> frequency (in Hz)
- Value at each cell -> the strength (amplitude of that frequency in that window of time)

This matrix is called a spectrogram. It’s a visual way to represent how the sound’s energy is distributed across both time and frequency. At this point, we’ve already gone from a 1D waveform to a 2D “image” that can be understood more easily by machine learning models - especially Convolutional Neural Networks (CNNs). But we can still do better by thinking about how we hear sound.

### Why the Mel Scale?
The human ear doesn’t perceive all frequencies equally - it is logarithmic. We are much more sensitive to lower frequencies than higher frequencies. For example, the difference between 500 Hz and 1000 Hz is very noticeable to us, but the same difference between higher frequencies such as between 7500 Hz and 8000 Hz is much harder to hear - even though they’re both 500 Hz apart. To mimic this perception, we convert the frequency axis of our spectrogram to the mel scale. The mel scale compresses high frequencies and stretches low ones, aligning with how we hear. The conversion from frequency (in Hz) to mel is done using this formula:

$$\text{mel}(f) = 2595 \times \log_{10}(1 + \frac{f}{700})$$

This logarithmic scaling is similar to what we see in music, for example how each octave doubles in frequency but still feels like the same jump in pitch. 


### Mel Filter Bank to Mel Spectrogram
To apply this idea in practice, we pass our spectrogram through a mel filter bank, which is a set of overlapping triangular filters that group together frequencies according to the mel scale. The result is a mel spectrogram, which:
- Keeps the time axis
- Replaces the frequency axis with mel bins (e.g. 40 perceptually spaced bands)
- Uses brightness or intensity to represent energy in each band

The mel spectrogram is the final, compact representation of sound that we feed into our model. It's essentially a compressed, perceptually meaningful image of sound. 


### Why This Works for Wake Word Detection
The mel spectrogram is perfect for the wake word detection task because:
- It captures the key features of speech while reducing noise
- It mimics human hearing
- And it looks like an image - which CNNs and LSTMs are great at processing

We use it to train a neural network to learn the visual patterns that correspond to the wake word (like "Hi Pi") and distinguish it from other background noise or speech. Instead of feeding in raw, messy audio, we give our model a clean, structured, and meaningful input, one that helps it learn faster and more accurately.

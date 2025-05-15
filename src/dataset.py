import os
import numpy as np
import torch
from torch.utils.data import Dataset

class WakeWordMelDataset(Dataset):
    def __init__(self, wake_dir, not_wake_dir, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform

        for f in os.listdir(wake_dir):
            if f.endswith(".npy"):
                self.samples.append(os.path.join(wake_dir, f))
                self.labels.append(1)

        for f in os.listdir(not_wake_dir):
            if f.endswith(".npy"):
                self.samples.append(os.path.join(not_wake_dir, f))
                self.labels.append(0)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        mel = np.load(self.samples[idx])
        label = self.labels[idx]
        mel_tensor = torch.tensor(mel.T, dtype=torch.float32).unsqueeze(0)
        return mel_tensor, torch.tensor(label, dtype=torch.float32)
    


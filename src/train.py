import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.model import TinyWakeWordCNN
from src.dataset import WakeWordMelDataset
from src.utils import create_tensor

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wake-dir", required=True)
    p.add_argument("--not-wake-dir", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--checkpoint", default="experiments/model.pt")
    return p.parse_args()

def main():
    args = parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    dataset = WakeWordMelDataset(args.wake_dir, args.not_wake_dir)
    labels = dataset.labels
    counts = np.bincount(labels)
    weights = 1. / counts
    sample_weights = [weights[l] for l in labels]

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    model = TinyWakeWordCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for mel, label in dataloader:
            mel, label = mel.to(device), label.to(device).unsqueeze(1)
            optimizer.zero_grad()
            prob = model(mel)
            loss = criterion(prob, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * mel.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.checkpoint)
            print(f"  -> saved new best model to {args.checkpoint}")

if __name__ == "__main__":
    main()

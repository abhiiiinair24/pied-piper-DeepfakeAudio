import random
import numpy as np
from pathlib import Path

import soundfile as sf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchaudio.transforms as T

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AudioDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, duration=5.0):
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)

        self.files = []
        self.labels = []

        for label_name, label in [("real", 0), ("fake", 1)]:
            for f in (self.root_dir / label_name).rglob("*.wav"):
                self.files.append(f)
                self.labels.append(label)

        self.resamplers = {}
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=128
        )
        self.db = T.AmplitudeToDB()

    def __len__(self):
        return len(self.files)

    def _load_audio(self, path):
        audio, sr = sf.read(str(path), always_2d=True)  # shape: [time, channels]
        audio = audio.mean(axis=1)  # mono
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # [1, time]

        if sr != self.sample_rate:
            if sr not in self.resamplers:
                self.resamplers[sr] = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = self.resamplers[sr](waveform)

        return waveform

    def __getitem__(self, idx):
        f = self.files[idx]
        label = self.labels[idx]

        waveform = self._load_audio(f)

        # fix length
        if waveform.shape[1] < self.num_samples:
            pad = self.num_samples - waveform.shape[1]
            waveform = nn.functional.pad(waveform, (0, pad))
        else:
            start = random.randint(0, waveform.shape[1] - self.num_samples)
            waveform = waveform[:, start:start + self.num_samples]

        x = self.db(self.mel(waveform))
        x = (x - x.mean()) / (x.std() + 1e-6)

        return x, torch.tensor(label, dtype=torch.float32)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(1)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    preds, labels = [], []

    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)

        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).float()

        preds.extend(pred.detach().cpu().numpy())
        labels.extend(y.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    return avg_loss, acc, prec, rec, f1


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    train_ds = AudioDataset("data/train")
    val_ds = AudioDataset("data/val")
    test_ds = AudioDataset("data/test")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    model = CNN().to(device)

    fake = sum(train_ds.labels)
    real = len(train_ds.labels) - fake
    pos_weight = torch.tensor([real / fake], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_f1 = 0.0

    for epoch in range(10):
        print(f"\nEpoch {epoch + 1}")

        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, False)

        print("Train:", train_metrics)
        print("Val  :", val_metrics)

        if val_metrics[-1] > best_f1:
            best_f1 = val_metrics[-1]
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model")

    print("\nTesting best model on test set")
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    test_metrics = run_epoch(model, test_loader, criterion, optimizer, device, False)
    print("Test:", test_metrics)


if __name__ == "__main__":
    main()
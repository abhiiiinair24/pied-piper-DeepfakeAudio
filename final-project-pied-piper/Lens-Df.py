import random
import numpy as np
from pathlib import Path
import pandas as pd
import soundfile as sf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LensDFDataset(Dataset):
    def __init__(self, root_dir, csv_file, sample_rate=16000, duration=5.0):
        self.root_dir = Path(root_dir)
        self.df = pd.read_csv(csv_file)

        # Drop unnamed index column if present
        unnamed_cols = [c for c in self.df.columns if str(c).startswith("Unnamed:")]
        if unnamed_cols:
            self.df = self.df.drop(columns=unnamed_cols)

        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)

        self.resamplers = {}
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=128
        )
        self.db = T.AmplitudeToDB()

    def __len__(self):
        return len(self.df)

    def _fix_path(self, path_str):
        filename = Path(path_str).name
        return self.root_dir / "wavs" / filename

    def _load_audio(self, path):
        audio, sr = sf.read(str(path), always_2d=True)
        audio = audio.mean(axis=1)
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

        if sr != self.sample_rate:
            if sr not in self.resamplers:
                self.resamplers[sr] = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = self.resamplers[sr](waveform)

        return waveform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path = self._fix_path(row["file"])
        if not path.exists():
            raise FileNotFoundError(f"Could not find audio file: {path}")

        label = 0 if row["label"] == "bonafide" else 1

        waveform = self._load_audio(path)

        if waveform.shape[1] < self.num_samples:
            pad = self.num_samples - waveform.shape[1]
            waveform = nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :self.num_samples]

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


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    preds, labels = [], []

    for x, y in tqdm(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device)

        logits = model(x)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).float()

        preds.extend(pred.cpu().numpy())
        labels.extend(y.cpu().numpy())

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    print("\nLens-DF Results:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    dataset = LensDFDataset(
        root_dir="data/Lens-DF/dev/dev",
        csv_file="data/Lens-DF/dev/dev/data.csv"
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = CNN().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))

    evaluate(model, loader, device)


if __name__ == "__main__":
    main()
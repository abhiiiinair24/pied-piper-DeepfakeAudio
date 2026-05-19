import random
import numpy as np
from pathlib import Path
import pandas as pd
import soundfile as sf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class UnlabeledAudioDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, duration=5.0):
        self.files = sorted(list(Path(root_dir).rglob("*.wav")))
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
        return len(self.files)

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
        f = self.files[idx]
        waveform = self._load_audio(f)

        if waveform.shape[1] < self.num_samples:
            pad = self.num_samples - waveform.shape[1]
            waveform = nn.functional.pad(waveform, (0, pad))
        else:
            start = 0
            waveform = waveform[:, start:start + self.num_samples]

        x = self.db(self.mel(waveform))
        x = (x - x.mean()) / (x.std() + 1e-6)

        return x, str(f)


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
def predict(model, loader, device, output_csv="rand_test_predictions.csv"):
    model.eval()
    results = []

    for x, paths in tqdm(loader):
        x = x.to(device, non_blocking=True)

        logits = model(x)
        probs = torch.sigmoid(logits)

        for prob, path in zip(probs.cpu().numpy(), paths):
            results.append({
                "filepath": path,
                "fake_probability": float(prob),
                "prediction": int(prob > 0.5)
            })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved predictions to {output_csv}")


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    dataset = UnlabeledAudioDataset("data/Rand_test/test/audio")
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = CNN().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))

    predict(model, loader, device, output_csv="rand_test_predictions.csv")


if __name__ == "__main__":
    main()
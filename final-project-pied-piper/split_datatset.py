import shutil
import random
from pathlib import Path
from tqdm import tqdm


def collect_files(base_dir):
    base_dir = Path(base_dir)
    return list(base_dir.rglob("*.wav"))


def split_envsdd(
    input_dir="data/development",
    output_dir="data",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    random.seed(seed)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    print("Collecting files")

    real_files = collect_files(input_dir / "real_audio")
    fake_files = collect_files(input_dir / "fake_audio")

    print(f"Real files: {len(real_files)}")
    print(f"Fake files: {len(fake_files)}")

    def split_files(files):
        random.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return (
            files[:n_train],
            files[n_train:n_train + n_val],
            files[n_train + n_val:]
        )

    real_train, real_val, real_test = split_files(real_files)
    fake_train, fake_val, fake_test = split_files(fake_files)

    splits = {
        "train": (real_train, fake_train),
        "val": (real_val, fake_val),
        "test": (real_test, fake_test),
    }

    print("\nCopying files\n")

    for split, (real_list, fake_list) in splits.items():
        real_out = output_dir / split / "real"
        fake_out = output_dir / split / "fake"

        real_out.mkdir(parents=True, exist_ok=True)
        fake_out.mkdir(parents=True, exist_ok=True)

        print(f"\n--- {split.upper()} ---")

        # REAL
        for f in tqdm(real_list, desc=f"{split} REAL"):
            new_name = f"{f.parent.name}_{f.name}"
            shutil.copy(f, real_out / new_name)

        # FAKE
        for f in tqdm(fake_list, desc=f"{split} FAKE"):
            new_name = f"{f.parent.name}_{f.name}"
            shutil.copy(f, fake_out / new_name)

    print("\nDone")


if __name__ == "__main__":
    split_envsdd()
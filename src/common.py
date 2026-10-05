from pathlib import Path
import pandas as pd
import librosa

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Dataset_Final"
FEAT = ROOT / "features"
SR = 16000


def load_meta():
    tr = pd.read_csv(DATA / "train.csv").assign(split="train")
    te = pd.read_csv(DATA / "test.csv").assign(split="test", label=float("nan"))
    df = pd.concat([tr, te], ignore_index=True)
    df["key"] = df.split + "/" + df.filename
    df["path"] = [str(DATA / k) for k in df.key]
    return df


def load_audio(path):
    return librosa.load(path, sr=SR, mono=True)[0]

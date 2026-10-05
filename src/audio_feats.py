"""Hand-crafted prosody/energy features + layer-wise mean-pooled WavLM embeddings."""
import numpy as np
import pandas as pd
import torch
import librosa
from transformers import AutoFeatureExtractor, WavLMModel
from common import FEAT, SR, load_meta, load_audio

HOP = 160  # 10 ms frames


def prosody(y):
    rms = librosa.feature.rms(y=y, frame_length=400, hop_length=HOP)[0]
    voiced = rms > max(0.02, 0.1 * np.percentile(rms, 95))
    # run-lengths of silence between speech = pauses
    edges = np.flatnonzero(np.diff(np.r_[1, voiced.astype(int), 1]))
    runs = np.diff(edges)[::2] * HOP / SR if len(edges) > 1 else np.array([])
    pauses = runs[runs > 0.3]
    flat = librosa.feature.spectral_flatness(y=y, hop_length=HOP)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP)[0]
    return dict(
        dur=len(y) / SR, rms_mean=rms.mean(), rms_std=rms.std(), voiced_ratio=voiced.mean(),
        n_pauses=len(pauses), pause_mean=pauses.mean() if len(pauses) else 0,
        pause_total=pauses.sum(), flatness=flat.mean(), zcr=zcr.mean(),
    )


def main():
    df = load_meta()
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").cuda().eval().half()
    cache = FEAT / "audio_cache"
    for i, (k, p) in enumerate(zip(df.key, df.path)):
        out = cache / (k.replace("/", "__") + ".npz")
        if out.exists():
            continue
        y = load_audio(p)
        x = fe(y, sampling_rate=SR, return_tensors="pt").input_values.cuda().half()
        with torch.no_grad():
            hs = model(x, output_hidden_states=True).hidden_states
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, emb=torch.stack([h[0].float().mean(0) for h in hs]).cpu().numpy(), **prosody(y))
        if i % 50 == 0:
            print(f"{i}/{len(df)}", flush=True)
    z = [np.load(cache / (k.replace("/", "__") + ".npz")) for k in df.key]
    rows = pd.DataFrame([{c: float(f[c]) for c in f.files if c != "emb"} for f in z])
    pd.concat([df[["key", "filename", "split", "label"]], rows], axis=1).to_csv(FEAT / "prosody.csv", index=False)
    np.save(FEAT / "wavlm.npy", np.stack([f["emb"] for f in z]))  # (N, 13 layers, 768)


if __name__ == "__main__":
    main()

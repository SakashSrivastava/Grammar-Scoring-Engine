"""Frozen self-supervised speech encoder -> per-layer [mean, std] pooled over time (resumable)."""
import sys
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel
from common import FEAT, SR, load_meta, load_audio

MODEL = sys.argv[1] if len(sys.argv) > 1 else "microsoft/wavlm-large"
TAG = MODEL.split("/")[-1]


@torch.no_grad()
def main():
    df = load_meta()
    fe = AutoFeatureExtractor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    cache = FEAT / f"{TAG}_cache"
    cache.mkdir(exist_ok=True)
    for i, (k, p) in enumerate(zip(df.key, df.path)):
        out = cache / (k.replace("/", "__") + ".npy")
        if out.exists():
            continue
        x = fe(load_audio(p), sampling_rate=SR, return_tensors="pt").input_values.cuda().half()
        hs = model(x, output_hidden_states=True).hidden_states
        np.save(out, torch.stack([torch.cat([h[0].float().mean(0), h[0].float().std(0)]) for h in hs]).cpu().numpy())
        if i % 100 == 0:
            print(f"{i}/{len(df)}", flush=True)
    np.save(FEAT / f"ssl_{TAG}.npy", np.stack([np.load(cache / (k.replace("/", "__") + ".npy")) for k in df.key]))


if __name__ == "__main__":
    main()

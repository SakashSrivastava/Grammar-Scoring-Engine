"""Mean-pooled Whisper encoder states (every 4th layer) over the real (unpadded) audio frames."""
import sys
import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperModel
from common import FEAT, SR, load_meta, load_audio

MODEL = sys.argv[1] if len(sys.argv) > 1 else "openai/whisper-large-v3-turbo"
TAG = MODEL.split("/")[-1]
CHUNK = 30 * SR


@torch.no_grad()
def main():
    df = load_meta()
    fe = WhisperFeatureExtractor.from_pretrained(MODEL)
    enc = WhisperModel.from_pretrained(MODEL, dtype=torch.float16).encoder.cuda().eval()
    cache = FEAT / f"{TAG}_cache"
    cache.mkdir(exist_ok=True)
    for i, (k, p) in enumerate(zip(df.key, df.path)):
        out = cache / (k.replace("/", "__") + ".npy")
        if out.exists():
            continue
        y = load_audio(p)
        chunks = [y[s:s + CHUNK] for s in range(0, len(y), CHUNK) if len(y[s:s + CHUNK]) > SR]
        x = fe(chunks, sampling_rate=SR, return_tensors="pt").input_features.cuda().half()
        hs = enc(x, output_hidden_states=True).hidden_states[::4]  # encoder frame = 20 ms
        keep = [len(c) // 320 for c in chunks]
        pooled = [torch.cat([h[j, :n] for j, n in enumerate(keep)]).float().mean(0) for h in hs]
        np.save(out, torch.stack(pooled).cpu().numpy())
        if i % 100 == 0:
            print(f"{i}/{len(df)}", flush=True)
    np.save(FEAT / f"{TAG}_enc.npy", np.stack([np.load(cache / (k.replace("/", "__") + ".npy")) for k in df.key]))


if __name__ == "__main__":
    main()

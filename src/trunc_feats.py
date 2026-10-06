"""Test-condition simulation: re-embed every clip longer than 50 s using only its first 45 s
(most test clips are ~45 s recordings cut mid-answer). Shorter clips keep their own embedding.
Writes features/t45_wavlm-large.npy (layers 20-21 mean) and features/t45_whisper.npy (layers 28-32)."""
import numpy as np
import pandas as pd
import torch
from transformers import AutoFeatureExtractor, AutoModel, WhisperFeatureExtractor, WhisperModel
from common import FEAT, SR, load_meta, load_audio

CUT = 45 * SR


@torch.no_grad()
def main():
    df = load_meta()
    dur = pd.read_csv(FEAT / "prosody.csv").dur.values
    wl = np.load(FEAT / "ssl_wavlm-large.npy")
    wl = wl[:, 20:22, : wl.shape[-1] // 2].mean(1)
    wh = np.load(FEAT / "whisper-large-v3-turbo_enc.npy")[:, 7:].mean(1)
    long_idx = np.flatnonzero(dur > 50)
    audio = {i: load_audio(df.path[i])[:CUT] for i in long_idx}

    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
    m = AutoModel.from_pretrained("microsoft/wavlm-large", dtype=torch.float16).cuda().eval()
    for i in long_idx:
        x = fe(audio[i], sampling_rate=SR, return_tensors="pt").input_values.cuda().half()
        hs = m(x, output_hidden_states=True).hidden_states
        wl[i] = torch.stack([hs[l][0].float().mean(0) for l in (20, 21)]).mean(0).cpu().numpy()
    del m
    torch.cuda.empty_cache()
    np.save(FEAT / "t45_wavlm-large.npy", wl)
    print("wavlm-large done", flush=True)

    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")
    enc = WhisperModel.from_pretrained("openai/whisper-large-v3-turbo", dtype=torch.float16).encoder.cuda().eval()
    for i in long_idx:
        y = audio[i]
        chunks = [y[s:s + 30 * SR] for s in range(0, len(y), 30 * SR) if len(y[s:s + 30 * SR]) > SR]
        x = fe(chunks, sampling_rate=SR, return_tensors="pt").input_features.cuda().half()
        hs = enc(x, output_hidden_states=True).hidden_states[28::4]  # layers 28, 32
        keep = [len(c) // 320 for c in chunks]
        wh[i] = torch.stack([torch.cat([h[j, :n] for j, n in enumerate(keep)]).float().mean(0) for h in hs]).mean(0).cpu().numpy()
    np.save(FEAT / "t45_whisper.npy", wh)
    print("whisper done", flush=True)


if __name__ == "__main__":
    main()

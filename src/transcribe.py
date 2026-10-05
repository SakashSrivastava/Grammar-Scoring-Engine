"""Whisper ASR over all clips -> features/transcripts.csv (resumable via a jsonl cache)."""
import json
import sys
import pandas as pd
import torch
from transformers import pipeline
from common import FEAT, load_meta, load_audio

MODEL = sys.argv[1] if len(sys.argv) > 1 else "openai/whisper-large-v3-turbo"
CACHE = FEAT / "transcripts.jsonl"

df = load_meta()
FEAT.mkdir(exist_ok=True)
done = {json.loads(l)["key"] for l in open(CACHE)} if CACHE.exists() else set()
asr = pipeline("automatic-speech-recognition", model=MODEL, torch_dtype=torch.float16, device="cuda:0")
with open(CACHE, "a") as f:
    for i, (fn, p) in enumerate(zip(df.key, df.path)):
        if fn in done:
            continue
        out = asr(load_audio(p), chunk_length_s=30, batch_size=4,
                  generate_kwargs={"language": "en", "task": "transcribe", "temperature": 0.0})
        f.write(json.dumps({"key": fn, "text": out["text"].strip()}) + "\n")
        f.flush()
        if i % 50 == 0:
            print(f"{i}/{len(df)}", flush=True)

texts = pd.read_json(CACHE, lines=True).drop_duplicates("key")
df[["key", "filename", "split", "label"]].merge(texts, on="key").to_csv(FEAT / "transcripts.csv", index=False)
print("done", flush=True)

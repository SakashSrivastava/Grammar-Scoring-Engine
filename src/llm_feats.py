"""Mean-pooled hidden states of a decoder LLM over each transcript (no fine-tuning)."""
import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from common import FEAT

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B"
TAG = MODEL.split("/")[-1]


@torch.no_grad()
def main():
    texts = pd.read_csv(FEAT / "transcripts.csv").fillna({"text": ""}).text.tolist()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    n = model.config.num_hidden_layers
    layers = list(range(n // 4, n + 1, n // 8))  # middle-to-top layers
    out = []
    for t in texts:
        ids = tok(t or ".", return_tensors="pt", truncation=True, max_length=512).to("cuda")
        hs = model(**ids, output_hidden_states=True).hidden_states
        out.append(torch.stack([hs[l][0].float().mean(0) for l in layers]).cpu().numpy())
    np.save(FEAT / f"llm_{TAG}.npy", np.stack(out))
    print(TAG, "layers", layers)


if __name__ == "__main__":
    main()

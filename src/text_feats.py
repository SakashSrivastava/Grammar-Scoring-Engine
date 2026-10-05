"""Transcript features: surface stats, GPT-2 fluency (perplexity), sentence embeddings."""
import re
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import FEAT

FILLERS = {"um", "uh", "erm", "hmm", "ah", "er", "like"}


def stats(t, dur):
    words = re.findall(r"[a-zA-Z']+", t.lower())
    sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
    n = max(len(words), 1)
    return dict(
        n_words=len(words), wpm=len(words) / dur * 60, n_sents=len(sents),
        words_per_sent=len(words) / max(len(sents), 1), ttr=len(set(words)) / n,
        mean_word_len=np.mean([len(w) for w in words]) if words else 0,
        filler_rate=sum(w in FILLERS for w in words) / n,
        repeat_rate=sum(a == b for a, b in zip(words, words[1:])) / n,
        n_commas=t.count(",") / n,
    )


@torch.no_grad()
def perplexity(texts, name="openai-community/gpt2"):
    tok = AutoTokenizer.from_pretrained(name)
    lm = AutoModelForCausalLM.from_pretrained(name).cuda().eval()
    out = []
    for t in texts:
        ids = tok(t or ".", return_tensors="pt").input_ids[:, :1024].cuda()
        out.append(lm(ids, labels=ids).loss.item() if ids.shape[1] > 1 else 10.0)
    return np.array(out)


def main():
    tr = pd.read_csv(FEAT / "transcripts.csv").fillna({"text": ""})
    pros = pd.read_csv(FEAT / "prosody.csv")
    f = pd.DataFrame([stats(t, d) for t, d in zip(tr.text, pros.dur)])
    f["lm_nll"] = perplexity(tr.text.tolist())
    pd.concat([tr[["key", "filename", "split", "label"]], f], axis=1).to_csv(FEAT / "text_stats.csv", index=False)
    st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
    np.save(FEAT / "sbert.npy", st.encode(tr.text.tolist(), batch_size=32, normalize_embeddings=True))


if __name__ == "__main__":
    main()

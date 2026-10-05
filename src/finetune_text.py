"""5-fold fine-tune of DeBERTa-v3 as a transcript -> score regressor.
Saves out-of-fold and fold-averaged test predictions for use as a stacking input."""
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from common import FEAT

MODEL = sys.argv[1] if len(sys.argv) > 1 else "microsoft/deberta-v3-base"
TAG = MODEL.split("/")[-1]
SEEDS, EPOCHS, LR, BS, MAXLEN = [42, 7, 123], 6, 3e-5, 8, 256
MEAN, STD = 3.5, 1.0  # target standardisation for stable regression


def encode(tok, texts):
    return tok(list(texts), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")


@torch.no_grad()
def predict(model, enc):
    model.eval()
    out = []
    for i in range(0, len(enc.input_ids), 32):
        b = {k: v[i:i + 32].cuda() for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(**b).logits.float().squeeze(-1).cpu())
    return torch.cat(out).numpy() * STD + MEAN


def main():
    df = pd.read_csv(FEAT / "transcripts.csv").fillna({"text": ""})
    pros = pd.read_csv(FEAT / "prosody.csv")
    noise = ((pros.voiced_ratio > 0.95) & (pros.zcr > 0.4)).values
    trn, tst = (df.split == "train").values & ~noise, (df.split == "test").values
    X, y = df.text[trn].values, df.label[trn].values
    tok = AutoTokenizer.from_pretrained(MODEL)
    enc_te = encode(tok, df.text[tst])
    oof, te = np.zeros(len(y)), np.zeros(tst.sum())
    folds = [(s, k, a, b) for s in SEEDS
             for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=s).split(X, (y * 2).astype(int)))]
    for seed, k, a, b in folds:
        torch.manual_seed(seed + k)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, dtype=torch.float32).cuda()
        enc = encode(tok, X[a])
        tgt = torch.tensor((y[a] - MEAN) / STD, dtype=torch.float32)
        dl = DataLoader(list(range(len(a))), batch_size=BS, shuffle=True)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        sch = get_cosine_schedule_with_warmup(opt, len(dl) // 2, len(dl) * EPOCHS)
        for ep in range(EPOCHS):
            model.train()
            for idx in dl:
                batch = {kk: v[idx].cuda() for kk, v in enc.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(**batch).logits.float().squeeze(-1)
                loss = torch.nn.functional.mse_loss(pred, tgt[idx].cuda())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(), sch.step(), opt.zero_grad()
        p = predict(model, encode(tok, X[b]))
        oof[b] += p / len(SEEDS)
        te += predict(model, enc_te) / len(folds)
        print(f"seed {seed} fold {k} RMSE {np.sqrt(np.mean((p - y[b]) ** 2)):.4f}", flush=True)
        del model, opt
        torch.cuda.empty_cache()
    print(f"{TAG} OOF RMSE {np.sqrt(np.mean((oof - y) ** 2)):.4f}  r {np.corrcoef(oof, y)[0, 1]:.4f}")
    np.save(FEAT / f"ft_{TAG}_oof.npy", oof)
    np.save(FEAT / f"ft_{TAG}_test.npy", te)


if __name__ == "__main__":
    main()

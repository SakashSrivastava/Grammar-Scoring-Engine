# Kaggle notebook cell: 5-fold DeBERTa-v3-large regressor on Whisper transcripts.
# Settings: Accelerator = GPU P100 (or T4 x2), Internet = On, input dataset = shl-transcripts.
# Outputs (in /kaggle/working): ft_deberta-v3-large_oof.npy, ft_deberta-v3-large_test.npy
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

MODEL = "microsoft/deberta-v3-large"
SEEDS, EPOCHS, LR, BS, MAXLEN = [42, 7], 5, 1e-5, 8, 256
MEAN, STD = 3.5, 1.0

find = lambda name: glob.glob(f"/kaggle/input/**/{name}", recursive=True)[0]
df = pd.read_csv(find("transcripts.csv")).fillna({"text": ""})
pros = pd.read_csv(find("prosody.csv"))
noise = ((pros.voiced_ratio > 0.95) & (pros.zcr > 0.4)).values
trn, tst = (df.split == "train").values & ~noise, (df.split == "test").values
X, y = df.text[trn].values, df.label[trn].values
tok = AutoTokenizer.from_pretrained(MODEL)
encode = lambda t: tok(list(t), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
enc_te = encode(df.text[tst])


@torch.no_grad()
def predict(model, enc):
    model.eval()
    out = []
    for i in range(0, len(enc.input_ids), 32):
        b = {k: v[i:i + 32].cuda() for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(**b).logits.float().squeeze(-1).cpu())
    return torch.cat(out).numpy() * STD + MEAN


oof, te = np.zeros(len(y)), np.zeros(tst.sum())
for seed in SEEDS:
    for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, (y * 2).astype(int))):
        torch.manual_seed(seed + k)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, torch_dtype=torch.float32).cuda()
        enc = encode(X[a])
        tgt = torch.tensor((y[a] - MEAN) / STD, dtype=torch.float32)
        dl = DataLoader(list(range(len(a))), batch_size=BS, shuffle=True)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        sch = get_cosine_schedule_with_warmup(opt, len(dl) // 2, len(dl) * EPOCHS)
        scaler = torch.cuda.amp.GradScaler()
        for ep in range(EPOCHS):
            model.train()
            for idx in dl:
                batch = {kk: v[idx].cuda() for kk, v in enc.items()}
                with torch.autocast("cuda", dtype=torch.float16):
                    pred = model(**batch).logits.float().squeeze(-1)
                loss = torch.nn.functional.mse_loss(pred, tgt[idx].cuda())
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt), scaler.update(), sch.step(), opt.zero_grad()
        p = predict(model, encode(X[b]))
        oof[b] += p / len(SEEDS)
        te += predict(model, enc_te) / (5 * len(SEEDS))
        print(f"seed {seed} fold {k} RMSE {np.sqrt(np.mean((p - y[b]) ** 2)):.4f}", flush=True)
        del model, opt
        torch.cuda.empty_cache()

print(f"OOF RMSE {np.sqrt(np.mean((oof - y) ** 2)):.4f}  r {np.corrcoef(oof, y)[0, 1]:.4f}")
np.save("/kaggle/working/ft_deberta-v3-large_oof.npy", oof)
np.save("/kaggle/working/ft_deberta-v3-large_test.npy", te)

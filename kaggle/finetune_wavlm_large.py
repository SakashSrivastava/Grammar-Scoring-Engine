# Kaggle notebook cell: 5-fold end-to-end fine-tune of WavLM-large on raw audio -> grammar score.
# Settings: Accelerator = GPU T4 x2, Internet = On.
# Input: the competition data only.
# Outputs (in /kaggle/working): ft_wavlm-large-ft_oof.npy, ft_wavlm-large-ft_test.npy
import glob
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import WavLMModel, get_cosine_schedule_with_warmup

MODEL = "microsoft/wavlm-large"
SEED, EPOCHS, BS, CROP, SR = 42, 10, 8, 10, 16000
MICRO = 4  # clips per forward pass; gradients are accumulated over BS // MICRO passes (fits a 15 GB T4)
LR_ENC, LR_HEAD = 2e-5, 1e-3
MEAN, STD = 3.5, 1.0

class Regressor(torch.nn.Module):
    """WavLM encoder -> learned softmax mix of all layers -> mean over time -> linear score."""

    def __init__(self):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(MODEL, layerdrop=0.0)  # keep every layer so the layer mix is fixed-size
        self.wavlm.feature_extractor._freeze_parameters()  # keep the CNN front-end fixed
        self.layer_w = torch.nn.Parameter(torch.zeros(self.wavlm.config.num_hidden_layers + 1))
        self.head = torch.nn.Linear(self.wavlm.config.hidden_size, 1)

    def forward(self, x):
        hs = torch.stack(self.wavlm(x, output_hidden_states=True).hidden_states)  # (L, B, T, H)
        h = (hs * self.layer_w.softmax(0)[:, None, None, None]).sum(0).mean(1)
        return self.head(h).squeeze(-1)


find = lambda name: glob.glob(f"/kaggle/input/**/{name}", recursive=True)[0]
root = os.path.dirname(find("train.csv"))
meta = pd.concat([pd.read_csv(os.path.join(root, "train.csv")).assign(split="train"),
                  pd.read_csv(os.path.join(root, "test.csv")).assign(split="test")], ignore_index=True)
meta["key"] = meta.split + "/" + meta.filename


def load(key):
    path = os.path.join(root, key)
    if not os.path.exists(path):  # fall back to searching the inputs for <split>/<file>
        path = glob.glob(f"/kaggle/input/**/{key}", recursive=True)[0]
    y, sr = sf.read(path, dtype="float32")
    assert sr == SR
    return (y - y.mean()) / (y.std() + 1e-7)


wavs = [load(k) for k in meta.key]
zcr = np.array([np.mean(np.diff(np.signbit(w)) != 0) for w in wavs])
noise = zcr > 0.4  # white-noise clips (all labelled 0); speech stays far below
trn, tst = (meta.split == "train").values & ~noise, (meta.split == "test").values
wav_tr = [w for w, m in zip(wavs, trn) if m]
wav_te = [w for w, m in zip(wavs, tst) if m]
y = meta.label[trn].values
print("noise clips", noise[meta.split == "train"].sum(), noise[tst].sum(), flush=True)
print("loaded", len(wav_tr), len(wav_te), flush=True)


def crop(w, rng):
    n = CROP * SR
    if len(w) <= n:
        return np.pad(w, (0, n - len(w)))
    s = rng.integers(0, len(w) - n)
    return w[s:s + n]


@torch.no_grad()
def predict(model, wavs):
    model.eval()
    n, out = CROP * SR, []
    for w in wavs:  # average over consecutive CROP-second windows of the full clip
        chunks = [w[s:s + n] for s in range(0, max(len(w) - n // 2, 1), n)]
        chunks = [np.pad(c, (0, n - len(c))) for c in chunks]
        x = torch.tensor(np.stack(chunks)).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(x).float().mean().item())
        del x
    return np.array(out) * STD + MEAN


oof, te = np.zeros(len(y)), np.zeros(len(wav_te))
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(y, (y * 2).astype(int))):
    torch.manual_seed(SEED + k)
    rng = np.random.default_rng(SEED + k)
    model = Regressor().cuda()
    head = [model.layer_w, *model.head.parameters()]
    enc = [p for p in model.wavlm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc, "lr": LR_ENC}, {"params": head, "lr": LR_HEAD}], weight_decay=0.01)
    steps = (len(a) // BS) * EPOCHS
    sch = get_cosine_schedule_with_warmup(opt, steps // 10, steps)
    scaler = torch.amp.GradScaler("cuda")
    tgt = (y - MEAN) / STD
    for ep in range(EPOCHS):
        model.train()
        perm = rng.permutation(a)
        for i in range(0, len(perm) - BS + 1, BS):
            idx = perm[i:i + BS]
            for m in range(0, BS, MICRO):
                sub = idx[m:m + MICRO]
                x = torch.tensor(np.stack([crop(wav_tr[j], rng) for j in sub])).cuda()
                t = torch.tensor(tgt[sub], dtype=torch.float32).cuda()
                with torch.autocast("cuda", dtype=torch.float16):
                    pred = model(x).float()
                loss = torch.nn.functional.mse_loss(pred, t) * len(sub) / BS
                scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt), scaler.update(), sch.step(), opt.zero_grad()
    p = predict(model, [wav_tr[j] for j in b])
    oof[b] = p
    te += predict(model, wav_te) / 5
    print(f"fold {k} RMSE {np.sqrt(np.mean((p - y[b]) ** 2)):.4f}", flush=True)
    del model, opt
    torch.cuda.empty_cache()

print(f"OOF RMSE {np.sqrt(np.mean((oof - y) ** 2)):.4f}  r {np.corrcoef(oof, y)[0, 1]:.4f}")
np.save("/kaggle/working/ft_wavlm-large-ft_oof.npy", oof)
np.save("/kaggle/working/ft_wavlm-large-ft_test.npy", te)

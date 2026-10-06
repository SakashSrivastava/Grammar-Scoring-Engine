"""5-fold CV over feature blocks, then a non-negative linear stack of the block models."""
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from common import FEAT, ROOT

SEEDS = [0, 1, 2]
# Stack inputs chosen by repeated nested CV: two audio + two text models beat using all 18 base models
# (test-mix RMSE 0.477 vs 0.486) because the short-clip weights are fitted on only 178 clips.
STACK = ["wavlm-ft", "wavlm_large_svr", "deberta-v3-large", "deberta-v3-base"]


def load():
    pros = pd.read_csv(FEAT / "prosody.csv")
    txt = pd.read_csv(FEAT / "text_stats.csv")
    meta = pros[["key", "filename", "split", "label"]]
    tab = pd.concat([pros.drop(columns=meta.columns), txt.drop(columns=meta.columns)], axis=1).values
    wav = np.load(FEAT / "wavlm.npy")
    blocks = {"tab": tab, "sbert": np.load(FEAT / "sbert.npy"), "wavlm": wav[:, 7:10].mean(1),
              "whisper": np.load(FEAT / "whisper-large-v3-turbo_enc.npy")[:, 7:].mean(1),  # encoder layers 28-32
              "qwen": np.load(FEAT / "llm_Qwen2.5-1.5B.npy")[:, 3:5].mean(1)}  # LLM layers 16-19
    wl = np.load(FEAT / "ssl_wavlm-large.npy")
    blocks["wavlm_large"] = wl[:, 20:22, : wl.shape[-1] // 2].mean(1)  # layers 20-21, time-mean part
    for name in ("wavlm-large", "whisper"):  # same encoders on clips cut to their first 45 s (trunc_feats.py)
        if (FEAT / f"t45_{name}.npy").exists():
            blocks[f"t45_{name}"] = np.load(FEAT / f"t45_{name}.npy")
    if (FEAT / "llm_judge.npy").exists():  # zero-shot LLM rubric score: p(1..5) + expected value
        blocks["judge"] = np.load(FEAT / "llm_judge.npy")
    noise = ((pros.voiced_ratio > 0.95) & (pros.zcr > 0.4)).values  # white-noise clips, labelled 0
    short = (pros.dur < 50).values  # ~45 s clips: 24% of train but 69% of test
    return meta, blocks, noise, short


def models():
    ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30)))
    svr = lambda: make_pipeline(StandardScaler(), SVR(C=3, epsilon=0.2))
    return {
        "tab_lgbm": ("tab", LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15,
                                          min_child_samples=15, subsample=0.8, subsample_freq=1,
                                          colsample_bytree=0.8, verbose=-1)),
        "tab_ridge": ("tab", ridge()),
        "sbert_svr": ("sbert", svr()),
        "sbert_ridge": ("sbert", ridge()),
        "wavlm_svr": ("wavlm", svr()),
        "wavlm_ridge": ("wavlm", ridge()),
        "whisper_svr": ("whisper", svr()),
        "whisper_ridge": ("whisper", ridge()),
        "wavlm_large_svr": ("wavlm_large", svr()),
        "wavlm_large_ridge": ("wavlm_large", ridge()),
        "t45_wavlm_svr": ("t45_wavlm-large", svr()),
        "t45_whisper_svr": ("t45_whisper", svr()),
        "judge_ridge": ("judge", ridge()),
        "qwen_svr": ("qwen", svr()),
        "qwen_ridge": ("qwen", make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(1, 6, 30)))),
    }


def cv(X, y, Xte, est):
    oof, te = np.zeros(len(y)), np.zeros(len(Xte))
    for seed in SEEDS:
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, (y * 2).astype(int)):
            m = clone(est).fit(X[tr], y[tr])
            oof[va] += m.predict(X[va]) / len(SEEDS)
            te += m.predict(Xte) / (5 * len(SEEDS))
    return oof, te


def score(y, p):
    return np.sqrt(np.mean((y - p) ** 2)), pearsonr(y, p)[0]


def fit_stack(O, y):
    w, _ = nnls(np.c_[O, np.ones(len(y)), -np.ones(len(y))], y)  # non-negative weights, free intercept
    return w


def apply_stack(O, w):
    return np.c_[O, np.ones(len(O)), -np.ones(len(O))] @ w


def stack_by_length(O, y, short, T=None, short_te=None):
    """Separate stacking weights for short and long clips."""
    oof, pred, ws = np.zeros(len(y)), None if T is None else np.zeros(len(T)), {}
    for g in (True, False):
        w = ws[g] = fit_stack(O[short == g], y[short == g])
        oof[short == g] = apply_stack(O[short == g], w)
        if T is not None:
            pred[short_te == g] = apply_stack(T[short_te == g], w)
    return oof, pred, ws


def nested_stack_cv(O, y, short):
    """Honest stack estimate: weights fitted on 4/5 of the OOF rows, scored on the held-out 1/5."""
    out = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=99).split(O, (y * 2).astype(int)):
        _, out[b], _ = stack_by_length(O[a], y[a], short[a], O[b], short[b])
    return np.clip(out, 1, 5)


def nested_single(O, y):
    out = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=99).split(O, (y * 2).astype(int)):
        out[b] = apply_stack(O[b], fit_stack(O[a], y[a]))
    return np.clip(out, 1, 5)


def report(tag, y, p, short):
    r = lambda m: np.sqrt(np.mean((y[m] - p[m]) ** 2))
    mix = np.sqrt(0.69 * r(short) ** 2 + 0.31 * r(~short) ** 2)  # re-weighted to the test length mix
    print(f"{tag:22s} all {r(np.ones_like(short)):.4f}  short {r(short):.4f}  long {r(~short):.4f}  test-mix {mix:.4f}")


def main():
    meta, blocks, noise, short_all = load()
    trn = (meta.split == "train").values & ~noise  # models only ever see speech
    tst = (meta.split == "test").values
    y = meta.label[trn].values
    oofs, tes = {}, {}
    for name, (b, est) in models().items():
        if b not in blocks:
            continue
        X = blocks[b]
        oofs[name], tes[name] = cv(X[trn], y, X[tst], est)
        print(f"{name:12s} RMSE {score(y, oofs[name])[0]:.4f}  r {score(y, oofs[name])[1]:.4f}")
    for f in sorted(FEAT.glob("ft_*_oof.npy")):  # fine-tuned transformer predictions (finetune_text.py)
        name = f.stem[3:-4]
        oofs[name], tes[name] = np.load(f), np.load(str(f).replace("_oof", "_test"))
        print(f"{name:12s} RMSE {score(y, oofs[name])[0]:.4f}  r {score(y, oofs[name])[1]:.4f}")
    names = [n for n in STACK if n in oofs]
    O, T = np.column_stack([oofs[n] for n in names]), np.column_stack([tes[n] for n in names])
    short, short_te = short_all[trn], short_all[tst]
    for n in oofs:
        report(n, y, oofs[n], short)
    w1 = fit_stack(O, y)
    report("stack (single, nested)", y, nested_single(O, y), short)
    report("stack (by length, nested)", y, nested_stack_cv(O, y, short), short)
    oof, pred, ws = stack_by_length(O, y, short, T, short_te)
    for g, w in ws.items():
        print("short" if g else "long ", "weights", {n: round(v, 3) for n, v in zip(names, w) if v > 0}, "intercept %.3f" % (w[-2] - w[-1]))
    oof = np.clip(oof, 1, 5)
    pred = np.where(noise[tst], 0, np.clip(pred, 1, 5))
    print(f"noise clips: train {noise[meta.split == 'train'].sum()}, test {noise[tst].sum()}")
    print(f"STACK (OOF, in-sample weights)  RMSE {score(y, oof)[0]:.4f}  r {score(y, oof)[1]:.4f}")
    pd.DataFrame({"filename": meta.filename[tst].values, "label": pred}).to_csv(ROOT / "submission.csv", index=False)
    pd.DataFrame(oofs).assign(filename=meta.filename[trn].values, label=y, stack=oof).to_csv(FEAT / "oof.csv", index=False)


if __name__ == "__main__":
    main()

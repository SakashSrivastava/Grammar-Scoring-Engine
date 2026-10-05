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


def load():
    pros = pd.read_csv(FEAT / "prosody.csv")
    txt = pd.read_csv(FEAT / "text_stats.csv")
    meta = pros[["key", "filename", "split", "label"]]
    tab = pd.concat([pros.drop(columns=meta.columns), txt.drop(columns=meta.columns)], axis=1).values
    wav = np.load(FEAT / "wavlm.npy")
    blocks = {"tab": tab, "sbert": np.load(FEAT / "sbert.npy"), "wavlm": wav[:, 7:10].mean(1),
              "whisper": np.load(FEAT / "whisper-large-v3-turbo_enc.npy")[:, 7:].mean(1),  # encoder layers 28-32
              "qwen": np.load(FEAT / "llm_Qwen2.5-1.5B.npy")[:, 3:5].mean(1)}  # LLM layers 16-19
    noise = ((pros.voiced_ratio > 0.95) & (pros.zcr > 0.4)).values  # white-noise clips, labelled 0
    return meta, blocks, noise


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


def main():
    meta, blocks, noise = load()
    trn = (meta.split == "train").values & ~noise  # models only ever see speech
    tst = (meta.split == "test").values
    y = meta.label[trn].values
    oofs, tes = {}, {}
    for name, (b, est) in models().items():
        X = blocks[b]
        oofs[name], tes[name] = cv(X[trn], y, X[tst], est)
        print(f"{name:12s} RMSE {score(y, oofs[name])[0]:.4f}  r {score(y, oofs[name])[1]:.4f}")
    for f in sorted(FEAT.glob("ft_*_oof.npy")):  # fine-tuned transformer predictions (finetune_text.py)
        name = f.stem[3:-4]
        oofs[name], tes[name] = np.load(f), np.load(str(f).replace("_oof", "_test"))
        print(f"{name:12s} RMSE {score(y, oofs[name])[0]:.4f}  r {score(y, oofs[name])[1]:.4f}")
    O, T = np.column_stack(list(oofs.values())), np.column_stack(list(tes.values()))
    w, _ = nnls(np.c_[O, np.ones(len(y))], y)
    print("stack weights", dict(zip(list(oofs) + ["bias"], w.round(3))))
    oof = np.clip(np.c_[O, np.ones(len(y))] @ w, 1, 5)
    pred = np.where(noise[tst], 0, np.clip(np.c_[T, np.ones(len(T))] @ w, 1, 5))
    print(f"noise clips: train {noise[meta.split == 'train'].sum()}, test {noise[tst].sum()}")
    print(f"STACK (OOF)  RMSE {score(y, oof)[0]:.4f}  r {score(y, oof)[1]:.4f}")
    pd.DataFrame({"filename": meta.filename[tst].values, "label": pred}).to_csv(ROOT / "submission.csv", index=False)
    pd.DataFrame(oofs).assign(filename=meta.filename[trn].values, label=y, stack=oof).to_csv(FEAT / "oof.csv", index=False)


if __name__ == "__main__":
    main()

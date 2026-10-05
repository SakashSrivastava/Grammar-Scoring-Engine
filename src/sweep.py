"""Quick per-layer CV sweep of an embedding file: python sweep.py features/ssl_wavlm-large.npy"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from common import FEAT

E = np.load(sys.argv[1])
p = pd.read_csv(FEAT / "prosody.csv")
s = ((p.split == "train") & ~((p.voiced_ratio > 0.95) & (p.zcr > 0.4))).values
y, short = p.label[s].values, (p.dur[s] < 50).values
half = E.shape[-1] // 2
for L in range(E.shape[1]):
    for name, X in [("mean", E[s, L, :half]), ("mean+std", E[s, L])]:
        o = cross_val_predict(make_pipeline(StandardScaler(), SVR(C=3, epsilon=0.2)), X, y, cv=KFold(5, shuffle=True, random_state=0))
        r = lambda m: np.sqrt(np.mean((o[m] - y[m]) ** 2))
        print(f"L{L:2d} {name:8s} all {r(np.ones_like(short)):.3f} short {r(short):.3f} test-mix {np.sqrt(.69 * r(short) ** 2 + .31 * r(~short) ** 2):.3f}", flush=True)

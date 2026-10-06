# Grammar Scoring Engine

Predicts a 0–5 grammar score (MOS Likert) for 45–60 s spoken English clips. Built for the SHL Hiring Assessment 2026 Kaggle competition (769 labelled training clips, 216 test clips; metric RMSE).

**Report and full evaluation: [`grammar_scoring_engine.ipynb`](grammar_scoring_engine.ipynb)** — approach, preprocessing, pipeline architecture, per-model and stacked results, visualisations and the training-set RMSE.

| | RMSE | Pearson r |
|---|---|---|
| Training set, cross-validated (all 769 clips) | 0.439 | 0.935 |
| Training set, cross-validated (732 speech clips) | 0.450 | 0.896 |
| Public leaderboard | **0.3351** | — |

## Approach

```
 .wav (16 kHz mono)
   ├── noise check: zero-crossing rate > 0.4 ───────────────────────► score 0 (rule)
   ├── Whisper large-v3-turbo ─► transcript ─┬─ DeBERTa-v3-large (fine-tuned) ─┐
   │                                          └─ DeBERTa-v3-base  (fine-tuned) ─┤
   ├── WavLM-base+ (fine-tuned end-to-end on 15 s crops) ───────────────────────┼─► length-aware
   └── WavLM-large (frozen, layers 20–21) ─► SVR ───────────────────────────────┘   non-negative stack
```

Every base model is trained with 5-fold stratified CV; their out-of-fold predictions are combined by a non-negative least-squares stack with a free intercept, fitted separately for short (~45 s) and long (~60 s) clips.

Key findings:

- **37 training clips labelled 0 are white noise**, not speech (zero-crossing rate ≥ 0.44 vs ≤ 0.34 for speech). They are scored by rule and excluded from training; the test set contains none.
- **Test clips are mostly short**: 69 % are ~45 s versus 24 % of training clips. Audio models are much weaker on short clips while text models are not, so the stack weights text more on short clips and audio more on long ones, and model selection uses a test-mix RMSE (69 % short / 31 % long).
- **Fewer stack inputs generalise better**: repeated nested CV picked 4 base models out of 18 candidates (test-mix RMSE 0.477 vs 0.486).

## Iterations

| Version | Change | CV test-mix RMSE | Public LB RMSE |
|---|---|---|---|
| v1 | frozen WavLM-base + SVR, noise rule | 0.617 | 0.4705 |
| v3 | + fine-tuned DeBERTa-v3-base | — | 0.3695 |
| v6 | + Whisper-encoder features, DeBERTa-v3-large | 0.504 | 0.3621 |
| v8 | length-aware stack with free intercept | 0.495 | 0.3546 |
| v10 | + end-to-end fine-tuned WavLM-base+ | 0.481 | 0.3401 |
| v13 | stack pruned to 4 inputs | 0.477 | 0.3351 |

Tested without gain: hand-crafted prosody/transcript features, sentence and LLM (Qwen2.5-1.5B) embeddings, a zero-shot LLM grammar judge, mean+std pooling, truncating training clips to 45 s.

## Layout

| Path | Purpose |
|---|---|
| `grammar_scoring_engine.ipynb` | report, modelling, stacking, evaluation, submission |
| `src/transcribe.py` | Whisper transcripts (resumable) |
| `src/audio_feats.py` | prosody features + WavLM-base embeddings |
| `src/ssl_feats.py`, `src/whisper_feats.py` | frozen WavLM-large / Whisper-encoder embeddings |
| `src/text_feats.py`, `src/llm_feats.py`, `src/llm_judge.py` | transcript statistics, GPT-2 likelihood, sentence / LLM embeddings, LLM judge |
| `src/finetune_text.py` | 5-fold DeBERTa regressor (OOF + test predictions) |
| `src/trunc_feats.py` | 45 s truncation experiment |
| `src/train.py` | base-model CV, length-aware stack, submission |
| `kaggle/` | GPU jobs run on Kaggle T4 (DeBERTa-large, WavLM fine-tunes) |

## Reproducing

1. Put the competition data in `Dataset_Final/` (`train.csv`, `test.csv`, `train/`, `test/`; not redistributed here).
2. Feature extraction (GPU, from `src/`): `transcribe.py → audio_feats.py → ssl_feats.py → whisper_feats.py → text_feats.py → finetune_text.py`. Run `kaggle/finetune_deberta_large.py` and `kaggle/finetune_wavlm.py` on a Kaggle GPU and copy their `ft_*.npy` outputs into `features/`.
3. Run the notebook (or `python src/train.py`) to fit the stack and write `submission.csv`.

Environment: Python 3.12, PyTorch 2.6 (CUDA 12.4), transformers, scikit-learn, LightGBM, librosa.

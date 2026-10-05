# Grammar Scoring Engine

Predicts a 0–5 grammar score (MOS Likert) for 45–60 s spoken English clips. Built for the SHL Hiring Assessment 2026 Kaggle competition (769 labelled training clips, 216 test clips; metric RMSE).

> Work in progress — the final report notebook will be added here.

## Approach

```
audio (.wav, 16 kHz)
 ├─ noise check (zero-crossing rate) ───────────────► white-noise clips → score 0
 ├─ Whisper large-v3-turbo ─► transcript
 │     ├─ DeBERTa-v3 base / large, fine-tuned regressors (5-fold)
 │     ├─ Qwen2.5-1.5B hidden states → SVR / Ridge
 │     └─ surface stats + GPT-2 log-likelihood → Ridge / LightGBM
 ├─ WavLM base+ / large (frozen, layer-averaged) ──► SVR / Ridge
 └─ Whisper encoder (frozen, layers 28–32) ────────► SVR / Ridge
                         │
       out-of-fold predictions of every model
                         ▼
   non-negative linear stack, fitted separately for short (<50 s) and long clips
```

Key findings that shaped it:

- **37 training clips labelled 0 are white noise**, not speech (zero-crossing rate ≥ 0.44 vs ≤ 0.34 for all speech). They are detected by rule and excluded from training; the test set contains none.
- **Test clips are mostly short**: 69 % are ~45 s versus 24 % of training clips. Audio models are much weaker on short clips while text models are not, so the stack learns separate weights per length group and validation reports a test-mix RMSE (69 % short / 31 % long).
- **Predictions regress to the mean**; letting the stack learn a free intercept with weights summing above 1 expands the range and lowers error.
- All validation is out-of-fold (5-fold, stratified on the label); the stack itself is scored with nested CV.

## Results

| Version | Contents | CV RMSE (test-mix) | Public LB RMSE |
|---|---|---|---|
| v1 | WavLM + SVR | 0.62 | 0.4705 |
| v3 | + fine-tuned DeBERTa-v3-base | — | 0.3695 |
| v6 | + Whisper encoder, DeBERTa-v3-large | 0.504 | 0.3621 |
| v8 | length-aware stack | 0.495 | 0.3546 |

## Layout

| Path | Purpose |
|---|---|
| `src/transcribe.py` | Whisper transcripts (resumable) |
| `src/audio_feats.py` | prosody features + WavLM-base embeddings |
| `src/whisper_feats.py`, `src/ssl_feats.py` | frozen Whisper-encoder / WavLM-large embeddings |
| `src/text_feats.py`, `src/llm_feats.py` | transcript statistics, GPT-2 likelihood, sentence and LLM embeddings |
| `src/finetune_text.py` | 5-fold DeBERTa regressor (OOF + test predictions) |
| `src/train.py` | per-block CV models, length-aware stack, submission |
| `kaggle/` | GPU jobs run on Kaggle (DeBERTa-large, WavLM fine-tune) |

Run order: `transcribe.py → audio_feats.py → whisper_feats.py → ssl_feats.py → text_feats.py → llm_feats.py → finetune_text.py → train.py`. Data is expected in `Dataset_Final/` (not redistributed).

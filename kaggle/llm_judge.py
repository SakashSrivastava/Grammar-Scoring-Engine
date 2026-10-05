# Kaggle notebook cell: zero-shot LLM grammar judge over the Whisper transcripts.
# Settings: Accelerator = GPU T4 x2, Internet = On. Input: the shl-transcripts dataset.
# For each transcript the LLM is shown the rubric and asked for a 1-5 score; we read the
# probabilities it assigns to the tokens "1".."5" and keep them plus their expected value.
# Output (in /kaggle/working): llm_judge.npy  -> (n_clips, 6) = [p1..p5, expected score]
import glob
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-7B-Instruct"
RUBRIC = """You are an expert English examiner. Rate the GRAMMAR of a spoken response (transcribed by ASR) on this scale:
1 = struggles with sentence structure and syntax; limited control of simple structures and memorised patterns.
2 = limited understanding of sentence structure; uses simple structures but makes consistent basic mistakes; may leave sentences incomplete.
3 = decent grasp of sentence structure but errors in grammatical structure, or decent grammar but errors in syntax and structure.
4 = strong understanding of sentence structure and syntax; good control of grammar; occasional minor errors that do not cause misunderstanding.
5 = high grammatical accuracy and adept control of complex grammar; seldom makes noticeable mistakes; handles complex structures well.
Judge grammar only, not content, pronunciation or punctuation added by the transcriber. Answer with a single digit."""

df = pd.read_csv(glob.glob("/kaggle/input/**/transcripts.csv", recursive=True)[0]).fillna({"text": ""})
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="auto").eval()
digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(1, 6)]

out = []
with torch.no_grad():
    for i, t in enumerate(df.text):
        msgs = [{"role": "system", "content": RUBRIC},
                {"role": "user", "content": f"Transcript:\n\"{t}\"\n\nGrammar score (1-5):"}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        logits = model(**ids).logits[0, -1, digit_ids].float()
        p = logits.softmax(0).cpu().numpy()
        out.append(np.r_[p, (p * np.arange(1, 6)).sum()])
        if i % 100 == 0:
            print(i, len(df), out[-1].round(2), flush=True)

out = np.array(out)
np.save("/kaggle/working/llm_judge.npy", out)
tr = (df.split == "train").values & (df.label > 0).values
print("corr(expected score, label) on train speech clips: %.3f" % np.corrcoef(out[tr, 5], df.label[tr])[0, 1])

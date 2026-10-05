"""Zero-shot LLM grammar judge (4-bit Qwen2.5-7B-Instruct) over the transcripts.
For each transcript the model sees the rubric and must answer with one digit; we keep the
probabilities it puts on "1".."5" and their expected value -> features/llm_judge.npy (N, 6)."""
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from common import FEAT

MODEL = "Qwen/Qwen2.5-7B-Instruct"
RUBRIC = """You are an expert English examiner. Rate the GRAMMAR of a spoken response (transcribed by ASR) on this scale:
1 = struggles with sentence structure and syntax; limited control of simple structures and memorised patterns.
2 = limited understanding of sentence structure; uses simple structures but makes consistent basic mistakes; may leave sentences incomplete.
3 = decent grasp of sentence structure but errors in grammatical structure, or decent grammar but errors in syntax and structure.
4 = strong understanding of sentence structure and syntax; good control of grammar; occasional minor errors that do not cause misunderstanding.
5 = high grammatical accuracy and adept control of complex grammar; seldom makes noticeable mistakes; handles complex structures well.
Judge grammar only, not content, pronunciation or punctuation added by the transcriber. Answer with a single digit."""


@torch.no_grad()
def main():
    df = pd.read_csv(FEAT / "transcripts.csv").fillna({"text": ""})
    tok = AutoTokenizer.from_pretrained(MODEL)
    q4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=q4, device_map="cuda:0").eval()
    digits = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(1, 6)]
    out = []
    for i, t in enumerate(df.text):
        msgs = [{"role": "system", "content": RUBRIC},
                {"role": "user", "content": f"Transcript:\n\"{t}\"\n\nGrammar score (1-5):"}]
        ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), return_tensors="pt").to("cuda")
        p = model(**ids).logits[0, -1, digits].float().softmax(0).cpu().numpy()
        out.append(np.r_[p, p @ np.arange(1, 6)])
        if i % 100 == 0:
            print(i, len(df), out[-1].round(2), flush=True)
    out = np.array(out)
    np.save(FEAT / "llm_judge.npy", out)
    tr = (df.split == "train").values & (df.label > 0).values
    print("corr(expected score, label): %.3f" % np.corrcoef(out[tr, 5], df.label[tr])[0, 1])


if __name__ == "__main__":
    main()

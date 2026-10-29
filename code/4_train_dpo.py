"""Train one LoRA DPO adapter per dataset on Qwen2.5-1.5B-Instruct, then answer the 236 test questions (GPU).

A job is "<dataset name>:<seed>" (work/train/<name>.jsonl) or "base" (the untrained model). Each job loads the base model
fresh, trains (1 epoch over 1,000 examples, batch 16, beta 0.1, learning rate 5e-6, LoRA r 16 on all linear layers), then
  - generates 4 answers per test question (temperature 0.7, top-p 0.95, up to 64 new tokens, fixed seed);
  - computes log p("According to Wikipedia, it is X.") - log p("It is X.") per question (a secondary measure).
The system prompt and learning rate were fixed beforehand on 60 separate calibration questions: the prompt that brought the
base model closest to citing in 50 % of answers, and the rate that let synthetic 0-100 % preference shares move citing
from near 0 % to near 80 % without costing accuracy.
All jobs are in work/train/jobs.txt (1,720, about 2 GPU minutes each on an A40); split the file to use several GPUs.
-> work/generations/<job>.json (skipped if it exists)
python 4_train_dpo.py --jobs-file work/train/jobs.txt   |   --jobs base pp_calibrated_interesting_s1:0 ...
"""
import argparse, json, math, time
import pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import DATA, WORK

MODEL, REVISION = "Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SYSTEM = ("You are a helpful voice assistant. Answer the question in one short sentence. "
          "Mention where the information comes from when you think it helps.")
LR = 5e-6; OUT = WORK / "generations"


def load():
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16).cuda()
    return model, tok


def train(model, tok, name, seed):
    from datasets import Dataset
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer
    rows = [json.loads(l) for l in open(WORK / "train" / f"{name}.jsonl", encoding="utf-8")]
    for r in rows:
        r["prompt"] = [{"role": "system", "content": SYSTEM}] + r["prompt"]
    steps = math.ceil(len(rows) / 16)
    cfg = DPOConfig(output_dir=f"/tmp/dpo_{name}_{seed}", beta=0.1, learning_rate=LR, per_device_train_batch_size=16,
                    num_train_epochs=1, lr_scheduler_type="linear", warmup_steps=math.ceil(0.1 * steps), seed=seed,
                    bf16=True, gradient_checkpointing=False, logging_steps=5, save_strategy="no", report_to="none",
                    max_length=256)
    peft = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    tr = DPOTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(rows), processing_class=tok, peft_config=peft)
    tr.train()
    return tr.model, tr.state.log_history


def prompts(tok, questions):
    return [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
                                    add_generation_prompt=True, tokenize=False) for q in questions]


@torch.no_grad()
def generate(model, tok, texts, n=4, bs=64):
    tok.padding_side = "left"; out = []
    torch.manual_seed(1234)
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True).to("cuda")
        g = model.generate(**enc, do_sample=True, temperature=0.7, top_p=0.95, max_new_tokens=64, num_return_sequences=n,
                           pad_token_id=tok.pad_token_id)
        dec = tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        out += [dec[k * n:(k + 1) * n] for k in range(len(dec) // n)]
    return out


@torch.no_grad()
def logp(model, tok, texts, completions, bs=64):
    """Sum of log-probabilities of each completion (+ end-of-turn token) after its prompt."""
    res = []
    for i in range(0, len(texts), bs):
        ids, cl = [], []
        for t, c in zip(texts[i:i + bs], completions[i:i + bs]):
            p = tok(t, add_special_tokens=False).input_ids; q = tok(c + "<|im_end|>", add_special_tokens=False).input_ids
            ids.append(p + q); cl.append(len(q))
        L = max(map(len, ids)); x = torch.full((len(ids), L), tok.pad_token_id); m = torch.zeros_like(x)
        for k, s in enumerate(ids):
            x[k, :len(s)] = torch.tensor(s); m[k, :len(s)] = 1
        x, m = x.cuda(), m.cuda()
        lp = torch.log_softmax(model(input_ids=x, attention_mask=m).logits[:, :-1].float(), -1).gather(2, x[:, 1:, None])[..., 0]
        for k, s in enumerate(ids):
            res.append(float(lp[k, len(s) - 1 - cl[k]:len(s) - 1].sum()))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="*", default=[]); ap.add_argument("--jobs-file")
    a = ap.parse_args()
    jobs = a.jobs + ([l.strip() for l in open(a.jobs_file) if l.strip()] if a.jobs_file else [])
    B = pd.read_csv(DATA / "question_bank.csv"); B = B[B.split == "test"]
    gold = B.answers.str.split("|").str[0]; OUT.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        f = OUT / f"{job.replace(':', '__seed')}.json"
        if f.exists():
            continue
        t0 = time.time(); model, tok = load(); log = []
        if job != "base":
            name, seed = job.split(":")
            model, log = train(model, tok, name, int(seed))
        model.eval(); texts = prompts(tok, B.question.tolist())
        resp = generate(model, tok, texts)
        margin = [s - p for s, p in zip(logp(model, tok, texts, [f"According to Wikipedia, it is {g}." for g in gold]),
                                        logp(model, tok, texts, [f"It is {g}." for g in gold]))]
        json.dump(dict(job=job, qid=B.qid.tolist(), responses=resp, margin=margin, log=log, seconds=round(time.time() - t0)),
                  open(f, "w"), ensure_ascii=False)
        print(job, f"{time.time() - t0:.0f}s", flush=True)
        del model; torch.cuda.empty_cache()

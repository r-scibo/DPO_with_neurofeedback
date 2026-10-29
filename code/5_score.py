"""Score every model's answers to the 236 test questions (4 answers each).

citing rate  share of answers that mention where the information comes from (pattern below; it agreed with a hand check
             on 195 of 200 answers)
accuracy     share of answers containing an accepted answer (data/question_bank.csv)
margin       mean log p("According to Wikipedia, it is X.") - log p("It is X.")
Changes are against the base model. Reads work/generations/*.json (4_train_dpo.py).
-> results/dpo_jobs.csv
python 5_score.py
"""
import json, re
import numpy as np, pandas as pd
from common import DATA, WORK, RESULTS

SOURCE = re.compile(r"according to|source|wikipedia|as reported by|cited|reference|based on data from|per the|"
                    r"information (?:comes|is derived|is based|can be found|is sourced)|can be found (?:in|on)|derived from|"
                    r"as stated (?:on|in|by)|as per\b|documented|"
                    r"(?:answer|data|information)\b[^.]{0,60}?\b(?:comes from|is based on)|can be verified (?:through|in|on|by)|"
                    r"as described in|as confirmed by|mentioned (?:on|in)\b", re.I)


def correct(r, answers):
    return any(re.search(r"(?<!\w)" + re.escape(x) + r"(?!\w)", r, re.I) for x in answers)


if __name__ == "__main__":
    accepted = pd.read_csv(DATA / "question_bank.csv").set_index("qid").answers.str.split("|")
    rows = []
    for d in (json.load(open(f)) for f in sorted((WORK / "generations").glob("*.json"))):
        flat = [(x, a) for q, a in zip(d["responses"], accepted.loc[d["qid"]]) for x in q]
        rows.append(dict(job=d["job"], source_rate=100 * np.mean([bool(SOURCE.search(x)) for x, _ in flat]),
                         accuracy=100 * np.mean([correct(x, a) for x, a in flat]), margin=float(np.mean(d["margin"]))))
    J = pd.DataFrame(rows).sort_values("job").reset_index(drop=True); base = J[J.job == "base"].iloc[0]
    for m in ("source_rate", "accuracy", "margin"):
        J["d_" + m] = J[m] - base[m]
    J["name"] = J.job.str.split(":").str[0]; J["seed"] = pd.to_numeric(J.job.str.split(":").str[1])
    J = J.merge(pd.read_csv(DATA / "datasets.csv"), on="name", how="left")
    RESULTS.mkdir(exist_ok=True); J.to_csv(RESULTS / "dpo_jobs.csv", index=False)
    print(f"{len(J)} models; base model: cites in {base.source_rate:.1f} % of answers, accuracy {base.accuracy:.1f} %")

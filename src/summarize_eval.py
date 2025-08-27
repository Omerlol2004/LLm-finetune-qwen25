import json, statistics, os
from pathlib import Path
INP = "outputs/eval_compare.jsonl"
OUT = "outputs/EVAL_SUMMARY.json"
rows=[]
with open(INP,"r",encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))
n=len(rows)
avg_len_base = statistics.mean(len(r["base"].split()) for r in rows) if n else 0
avg_len_adap = statistics.mean(len(r["adapter"].split()) for r in rows) if n else 0
canberra = next((r for r in rows if "capital of Australia" in r["user"]), None)
out = {"num_rows": n, "avg_len_base": avg_len_base, "avg_len_adapter": avg_len_adap,
       "has_canberra_case": bool(canberra),
       "example_canberra": canberra}
os.makedirs("outputs", exist_ok=True)
with open(OUT,"w",encoding="utf-8") as f:
    json.dump(out,f,ensure_ascii=False,indent=2)
print("Wrote", OUT)
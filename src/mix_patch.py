import os, json, random

random.seed(42)
IN_MAIN = "data/processed/train.jsonl"
IN_PATCHES = ["data/processed/patch_facts.jsonl", "data/processed/patch_style.jsonl"]
OUT_TRAIN = "data/processed/train_patch.jsonl"
SAMPLE_MAIN = 1000   # take 1k from main to rehearse base behavior

def read_jsonl(path):
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: yield json.loads(line)

def main():
    main_rows = list(read_jsonl(IN_MAIN))
    random.shuffle(main_rows)
    main_rows = main_rows[:SAMPLE_MAIN]

    patch_rows = []
    for p in IN_PATCHES:
        patch_rows += list(read_jsonl(p))

    mixed = patch_rows + main_rows
    random.shuffle(mixed)

    os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)
    with open(OUT_TRAIN,"w",encoding="utf-8") as f:
        for r in mixed: f.write(json.dumps(r,ensure_ascii=False)+"\n")

    print("train_patch.jsonl size:", len(mixed), "| patch:", len(patch_rows), "| main_sample:", len(main_rows))

if __name__ == "__main__":
    main()
import os, json, argparse
from datetime import datetime

FACTS = os.path.join("kb","facts.jsonl")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--id", default=None)
    args = ap.parse_args()
    os.makedirs("kb", exist_ok=True)
    _id = args.id or f"fact_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    with open(FACTS,"a",encoding="utf-8") as f:
        f.write(json.dumps({"id":_id,"title":args.title,"text":args.text}, ensure_ascii=False)+"\n")
    print("Appended. Now rebuild index:\n  python .\\src\\rag_build.py")

if __name__ == "__main__":
    main()
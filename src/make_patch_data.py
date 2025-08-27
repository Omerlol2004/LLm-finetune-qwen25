import os, json, random

OUT_DIR = "data/processed"
random.seed(42)

FACT_QA = [
# (question, answer)  — keep concise, single-fact
("What is the capital of Australia and when was it founded?",
 "The capital of Australia is Canberra. The city name was officially adopted and the capital founded in 1913."),
("Define precision and recall briefly.",
 "Precision is TP/(TP+FP) — correctness of positive predictions. Recall is TP/(TP+FN) — coverage of actual positives."),
("What does the Feynman Technique encourage?",
 "Explaining a concept in simple terms to reveal gaps in understanding."),
("What is spaced repetition?",
 "Reviewing information at increasing intervals to improve long-term retention."),
("What is active recall?",
 "Testing yourself from memory instead of rereading to strengthen retention."),
("What is gradient checkpointing used for?",
 "To reduce GPU memory by recomputing activations during backward pass."),
("What does QLoRA do in fine-tuning?",
 "Loads the base model in 4-bit and trains small LoRA adapters to cut memory with minimal quality loss."),
("What is catastrophic forgetting in fine-tuning?",
 "When a model loses general knowledge while adapting to new narrow data."),
]

STYLE_REWRITES = [
# (input, nicer_output)
("Your request was denied due to policy violations.",
 "Thanks for your request. I can't approve it right now because it conflicts with our policy. If you'd like, I can suggest alternatives."),
("We cannot help with that.",
 "I'm not able to help with that, but here's something related that might be useful…"),
("You are wrong.",
 "I don't think that's correct—here's why, step by step:"),
("Send it again.",
 "Could you please resend it when you have a moment?"),
("This is unclear.",
 "I want to make sure I understand—could you share a bit more detail?"),
("This feature isn't available.",
 "That feature isn't available yet, but here are the closest options right now:"),
("Your account is blocked.",
 "It looks like the account is temporarily blocked. I can help you with steps to restore access."),
("Too long; didn't read.",
 "Here's a quick summary of the key points:"),
]

def to_messages_user_assistant(user, assistant):
    return [
      {"role":"system","content":"You are a concise, friendly assistant. Use plain, factual language."},
      {"role":"user","content":user},
      {"role":"assistant","content":assistant},
    ]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    facts = [{"messages": to_messages_user_assistant(q,a)} for q,a in FACT_QA]
    styles = [{"messages": to_messages_user_assistant("Rewrite in a friendly tone:\n\n"+src, tgt)}
              for src,tgt in STYLE_REWRITES]

    with open(os.path.join(OUT_DIR, "patch_facts.jsonl"), "w", encoding="utf-8") as f:
        for r in facts: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    with open(os.path.join(OUT_DIR, "patch_style.jsonl"), "w", encoding="utf-8") as f:
        for r in styles: f.write(json.dumps(r, ensure_ascii=False)+"\n")

    print("Wrote:",
          os.path.join(OUT_DIR, "patch_facts.jsonl"),
          os.path.join(OUT_DIR, "patch_style.jsonl"),
          "| facts:", len(facts), "styles:", len(styles))

if __name__ == "__main__":
    main()
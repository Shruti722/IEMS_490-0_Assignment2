import os, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

EVAL_FILE = os.environ.get("EVAL_FILE", "data/eval_questions.jsonl")
BASE_MODEL = os.environ.get("BASE_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct")
PEFT_ADAPTER_PATH = os.environ.get("PEFT_ADAPTER_PATH", "")

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "192"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.2"))

PROMPT_TEMPLATE = """\
### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

def gpu_dtype():
    return torch.float16 if torch.cuda.is_available() else torch.float32

def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.split("### Response:")[-1].strip()

def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=gpu_dtype(), device_map="auto"
    )

    if PEFT_ADAPTER_PATH:
        print(f"[infer] Using LoRA adapter: {PEFT_ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, PEFT_ADAPTER_PATH)

    with open(EVAL_FILE, "r") as f:
        rows = [json.loads(l) for l in f]

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/finetuned_responses.jsonl" if PEFT_ADAPTER_PATH else "outputs/base_responses.jsonl"
    with open(out_path, "w") as w:
        for r in rows:
            prompt = PROMPT_TEMPLATE.format(
                instruction=r.get("instruction",""),
                input=r.get("input",""),
            )
            resp = generate(model, tokenizer, prompt)
            w.write(json.dumps({"instruction": r.get("instruction",""),
                                "input": r.get("input",""),
                                "response": resp}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} responses to {out_path}")

if __name__ == "__main__":
    main()

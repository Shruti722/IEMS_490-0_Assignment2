import os, json
from pathlib import Path
import subprocess

# Tiny synthetic sentiment set (also works if you choose summarization later)
TINY_TRAIN = [{"instruction":"Classify sentiment as POS or NEG.","input":x,"output":y} for x,y in [
    ("I love this!", "POS"),
    ("Terrible product.", "NEG"),
    ("Absolutely fantastic experience.", "POS"),
    ("Not worth the money.", "NEG"),
]] * 5  # 20 rows
TINY_VAL = [
    {"instruction":"Classify sentiment as POS or NEG.","input":"This makes me happy.","output":"POS"},
    {"instruction":"Classify sentiment as POS or NEG.","input":"I am disappointed.","output":"NEG"},
]

def write_jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")

def main():
    train = Path("tmp_unit/train.jsonl")
    val   = Path("tmp_unit/val.jsonl")
    write_jsonl(train, TINY_TRAIN)
    write_jsonl(val,   TINY_VAL)

    env = os.environ.copy()
    env.update({
        "TRAIN_PATH": str(train),
        "VAL_PATH": str(val),
        "OUTPUT_DIR": "adapters/_unit_lora",
        "USE_4BIT": "1",
        "LORA_R": "8",
        "LORA_ALPHA": "16",
        "LORA_DROPOUT": "0.05",
        "LR": "3e-4",
        "EPOCHS": "1",
        "BATCH_SIZE": "8",
        "GRAD_ACCUM": "1",
        "MAX_STEPS": "20",
        "SAVE_STEPS": "50",
        "LOG_STEPS": "5",
    })
    print("[unit] Starting short LoRA training...")
    subprocess.run(["python", "train_lora.py"], check=True, env=env)
    assert Path("adapters/_unit_lora").exists()

    env2 = os.environ.copy()
    env2.update({
        "PEFT_ADAPTER_PATH": "adapters/_unit_lora",
        "EVAL_FILE": str(val),
    })
    subprocess.run(["python", "infer.py"], check=True, env=env2)
    print("[unit] OK.")

if __name__ == "__main__":
    main()

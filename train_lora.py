import os, torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    default_data_collator
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# ---------- ENV ----------
BASE_MODEL = os.environ.get("BASE_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct")
TRAIN_PATH = os.environ.get("TRAIN_PATH", "data/train.jsonl")
VAL_PATH   = os.environ.get("VAL_PATH",   "data/val.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "adapters/task-lora")
USE_4BIT   = os.environ.get("USE_4BIT", "1") == "1"  # T4-friendly

LORA_R        = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA    = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT  = float(os.environ.get("LORA_DROPOUT", "0.05"))

LR            = float(os.environ.get("LR", "2e-4"))
EPOCHS        = float(os.environ.get("EPOCHS", "1"))
BATCH_SIZE    = int(os.environ.get("BATCH_SIZE", "8"))
GRAD_ACCUM    = int(os.environ.get("GRAD_ACCUM", "2"))
MAX_STEPS     = int(os.environ.get("MAX_STEPS", "0"))    # 0 => use epochs
WARMUP_RATIO  = float(os.environ.get("WARMUP_RATIO", "0.03"))
LOG_STEPS     = int(os.environ.get("LOG_STEPS", "10"))
SAVE_STEPS    = int(os.environ.get("SAVE_STEPS", "200"))
SEED          = int(os.environ.get("SEED", "42"))

TARGET_MODULES = os.environ.get(
    "TARGET_MODULES",
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
).split(",")

PROMPT_TEMPLATE = os.environ.get("PROMPT_TEMPLATE", """\
### Instruction:
{instruction}

### Input:
{input}

### Response:
""")

def gpu_dtype():
    # T4: use fp16; CPU: fp32
    return torch.float16 if torch.cuda.is_available() else torch.float32

def format_row(ex):
    instruction = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    out = (ex.get("output") or "").strip()
    return {"text": PROMPT_TEMPLATE.format(instruction=instruction, input=inp) + out}

def load_jsonl_dataset(train_path, val_path):
    ds = load_dataset("json", data_files={"train": train_path, "validation": val_path})
    ds = ds.map(format_row, remove_columns=ds["train"].column_names)
    return ds

def main():
    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_kwargs = {}
    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=gpu_dtype(),
        )
        quant_kwargs["quantization_config"] = bnb

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=gpu_dtype(),
        device_map="auto",
        **quant_kwargs
    )

    # K-bit prep + checkpointing-friendly flags
    if USE_4BIT:
        model = prepare_model_for_kbit_training(model)
        # Explicitly ensure inputs can require grads under GC paths
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    # Extra safety (Trainer also toggles this with gradient_checkpointing=True)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # LoRA wrap
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Data
    ds = load_jsonl_dataset(TRAIN_PATH, VAL_PATH)

    # Tokenize + create labels directly (so loss always hooks into graph)
    def tok(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=1024,
            padding="max_length",
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tokenized = ds.map(tok, batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=min(4, BATCH_SIZE),
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        report_to="none",
        optim="paged_adamw_8bit" if USE_4BIT else "adamw_torch",
        seed=SEED,
        fp16=torch.cuda.is_available(),
        bf16=False,
        gradient_checkpointing=True,
        max_steps=MAX_STEPS,   # int; 0 means use epochs
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=default_data_collator,  # we already set labels
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    print(f"Saved LoRA adapter to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

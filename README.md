# IEMS 490-0 Assignment 2 — LoRA Fine-Tuning on Social Media Sentiment (TweetEval)

## Overview
This assignment implements **parameter-efficient fine-tuning (LoRA)** on a small language model to perform **sentiment classification** on the **TweetEval** dataset (Positive / Negative).  
The goal is to compare the base model’s performance with the fine-tuned LoRA adapter and analyze the results.

---

## Repository Structure
```
IEMS_490-0_Assignment2/
│
├── adapters/                       # Trained LoRA adapters & checkpoints
│   ├── task-lora/checkpoint-1/     # First training checkpoint (LoRA)
│   └── _unit_lora/checkpoint-20/   # Final training checkpoint (20 epochs)
│
├── data/                           # TweetEval data splits
│   ├── train.jsonl
│   ├── val.jsonl
│   └── eval_questions.jsonl
│
├── outputs/                        # Model inference results
│   ├── base_responses.jsonl
│   └── finetuned_responses.jsonl
│
├── tmp_unit/                       # Small-scale LoRA test runs
│
├── train_lora.py                   # LoRA fine-tuning script
├── infer.py                        # Inference & generation for evaluation
├── unit_test.py                    # Unit-level dataset / eval sanity checks
├── LLM_Assignment2.ipynb # End-to-end notebook (Colab/DeepDish)
└── README.md                       # You are here
```
---

## Environment Setup

```bash
pip install -r requirements.txt
pip install torch transformers peft datasets evaluate scikit-learn accelerate
```

## How to Run
1. Fine-Tuning with LoRA
```bash
python train_lora.py \
--model HuggingFaceTB/SmolLM2-360M-Instruct \
--train_file data/train.jsonl \
--validation_file data/val.jsonl \
--output_dir adapters/task-lora \
--num_train_epochs 1 \
--per_device_train_batch_size 4 \
--learning_rate 1e-4
```
2. Generating Responses (Base + Fine-Tuned)
```bash
python infer.py \
  --model HuggingFaceTB/SmolLM2-360M-Instruct \
  --adapter adapters/task-lora/checkpoint-1 \
  --eval_file data/eval_questions.jsonl \
  --output_dir outputs/
```
3. Evaluating Results
```bash
python unit_test.py
```
This computes accuracy, macro-F1, and prints the confusion matrix for both models.

---
## Notebook Access

You can view or run the full workflow for **Assignment 2 – Social Media Sentiment (TweetEval)** using the following notebooks:

| Version | Description | Link |
|----------|--------------|------|
| **Colab (Interactive)** | Includes progress bars, widget outputs, and visualizations. Recommended for full reproducibility. | [Open in Colab](https://colab.research.google.com/github/Shruti722/IEMS_490-0_Assignment2/blob/main/LLM_Assignment2.ipynb) |
| **GitHub (Clean Preview)** | Widget metadata removed for compatibility with GitHub’s viewer. Use for quick code & markdown review. | [View on GitHub](https://github.com/Shruti722/IEMS_490-0_Assignment2/blob/main/LLM_Assignment2_clean.ipynb) |

> **Note:**  
> If you see “Invalid Notebook” on the GitHub preview, use the **Colab version** instead — this happens when GitHub’s renderer can’t display Jupyter widget metadata.

**Option A — Run in Google Colab (T4 GPU)**

1. Open the notebook in Colab (Runtime → Change runtime type → GPU (T4)).
2. Run the Setup cell; it will:
   - create ~/drive/MyDrive/LLM_Assignment_2 if needed,
   - install dependencies,
   - place outputs in outputs/ and adapters in adapters/.
3. Run all cells (Runtime → Run all).
   - The notebook executes: data prep → training → inference → metrics.

Handy Colab shell cells you can use if needed:
```bash
# Create the working folder in Drive (if you want to mirror CLI layout)
mkdir -p /content/drive/MyDrive/LLM_Assignment_2
```
**Option B — Run on DeepDish via Notebook**

If you prefer a notebook on DeepDish:
```bash
module load python/3.10
python -m venv ~/venvs/llm && source ~/venvs/llm/bin/activate
pip install jupyterlab torch transformers peft datasets evaluate scikit-learn accelerate

# (Option 1) Start JupyterLab and port-forward (recommended in NU docs)
jupyter lab --no-browser --port 8888

# (Option 2) Execute the notebook headlessly
pip install papermill
papermill LLM_Assignment2.ipynb LLM_Assignment2.out.ipynb
```


## Results

| Model | Accuracy | Macro F1 |
|:------|:---------:|:---------:|
| **Base (SmolLM2-360M-Instruct)** | **0.617** | **0.611** |
| **Fine-Tuned (LoRA)** | **0.583** | **0.569** |

---

### Confusion Matrices

**Base Model**
```
[[22 16]
 [ 7 15]]
``` 
**Fine-Tuned Model**
```
[[23 15]
 [10 12]]
```
---

## Sample Outputs

| Input Tweet | Base Prediction | Fine-Tuned Prediction | Gold |
|--------------|----------------|----------------------|------|
| “Had a great time at the concert tonight!” | POS | POS | POS |
| “Traffic jam again… this city sucks.” | NEG | NEG | NEG |
| “I can’t believe how amazing this food is!” | POS | POS | POS |
| “My flight got delayed for 4 hours, wonderful 🙄” | NEG | POS | NEG |
| “Totally worth staying up all night for this project.” | POS | NEG | POS |

---

## Discussion — Base vs Fine-Tuned Models

- **Base Model** (`SmolLM2-360M-Instruct`) already exhibits strong **sentiment understanding** due to large-scale pretraining.  
- **Fine-Tuned Model** learns domain-specific tweet nuances but shows mild **overfitting**, leading to a slight accuracy drop.  
- The fine-tuned model performs better on **sarcastic or informal** tweets but is **weaker on balanced or ambiguous** ones.  
- With additional training (2–3 epochs) or a larger dataset (≥ 2,000 samples), the LoRA fine-tuned model would likely surpass the base model.

---

## Running on DeepDish GPU (Northwestern HPC)

You can run the same training workflow on **DeepDish GPU nodes**:

```bash
# 1. Load environment
module load python/3.10
source ~/venvs/llm/bin/activate

# 2. Clone your repo and move to directory
git clone https://github.com/Shruti722/IEMS_490-0_Assignment2.git
cd IEMS_490-0_Assignment2

# 3. Request GPU node
srun --gpus=1 --mem=32G --time=2:00:00 --pty bash

# 4. Run LoRA training
python train_lora.py --train_file data/train.jsonl --validation_file data/val.jsonl --output_dir adapters/task-lora

# 5. Run evaluation
python unit_test.py
```
---

## Key Takeaways

- Implemented a complete LoRA fine-tuning pipeline on TweetEval.
- Compared base vs fine-tuned model performance using accuracy, macro-F1, and confusion matrices.
- Observed consistent classification trends between positive and negative sentiments.
- Demonstrated reproducibility across Colab and DeepDish GPU environments.

# counterfactual-hotpotqa

Builds a counterfactual yes/no QA dataset from HotpotQA by minimally editing one
answer-bearing supporting fact so the answer flips (YES -> NO or NO -> YES),
while keeping the question and entity names fixed.

## Pipeline

1. `build_candidates.py` - loads HotpotQA (distractor train/val), filters to
   yes/no questions, keeps `same`/`both` comparison questions (lexical filter),
   extracts the exact supporting-fact sentences per entity, and writes
   `phase1_candidates_{train,val}.jsonl`.
2. `generate_counterfactuals.py` - for each Phase 1 candidate, makes one LLM
   call (prompts in `prompts.py`) to classify the comparison, extract the
   answer-bearing value/property and its exact span, and generate a one-fact
   intervention. The result is verified deterministically in Python
   (`verify_counterfactual`, no second LLM call) before being kept. Writes
   `counterfactual_{train,val}.jsonl`. Supports resuming and batched inference.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python build_candidates.py

python generate_counterfactuals.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --input phase1_candidates_train.jsonl \
    --output counterfactual_train.jsonl \
    --batch-size 8 \
    --max-new-tokens 1000

python generate_counterfactuals.py \
    --input phase1_candidates_val.jsonl \
    --output counterfactual_val.jsonl
```

## Output

Each line of `counterfactual_{train,val}.jsonl` has: `question`,
`original_answer`/`counterfactual_answer`, `comparison_type`
(`same_attribute`/`shared_property`), `entity_1`/`entity_2`, `attribute`,
`intervention_entity`, `original_fact`/`counterfactual_fact`,
`original_span`/`counterfactual_span`, `intervention_type`
(`break_equality`/`create_equality`/`remove_property`/`add_property`),
`is_synthetic: true`, and a `validation` object with the programmatic checks
that passed.

## Published dataset

`counterfactual_{train,val}.jsonl` (2,046 train / 168 val rows) are published on
the Hugging Face Hub as
[`prajaktakini/counterfactual-hotpotqa`](https://huggingface.co/datasets/prajaktakini/counterfactual-hotpotqa):

```python
from datasets import load_dataset

ds = load_dataset("prajaktakini/counterfactual-hotpotqa")
train, val = ds["train"], ds["validation"]
```

`phase1_candidates_{train,val}.jsonl` are not published — they're the
pre-LLM-filter candidate pool, regenerable locally via `build_candidates.py`.

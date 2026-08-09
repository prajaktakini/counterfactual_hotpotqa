#!/usr/bin/env python3
"""Join phase1 candidates with phase2 counterfactuals into eval-ready rows
(original + counterfactual passages), write combined_{split}.jsonl, and
optionally push the result to the Hugging Face Hub.

Usage:
    python combine_phases.py
    python combine_phases.py --push
"""

import argparse
import json
from pathlib import Path

SPLITS = {"train": "train", "val": "validation"}
REPO_ID = "prajaktakini/counterfactual-hotpotqa"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def combine(split: str) -> list[dict]:
    phase1_by_id = {row["id"]: row for row in _read_jsonl(Path(f"phase1_candidates_{split}.jsonl"))}

    rows = []
    for cf in _read_jsonl(Path(f"counterfactual_{split}.jsonl")):
        phase1 = phase1_by_id.get(cf["id"])
        if phase1 is None:
            continue

        passages = [f"{e['entity']}: {e['fact']}" for e in phase1["entities"]]
        counterfactual_passages = [
            f"{e['entity']}: {cf['counterfactual_fact'] if e['entity'] == cf['intervention_entity'] else e['fact']}"
            for e in phase1["entities"]
        ]

        rows.append({**cf, "passages": passages, "counterfactual_passages": counterfactual_passages})

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true", help="push the combined dataset to the Hugging Face Hub")
    args = parser.parse_args()

    combined = {}
    for split in SPLITS:
        rows = combine(split)
        Path(f"combined_{split}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        print(f"{split}: {len(rows)} rows -> combined_{split}.jsonl")
        combined[split] = rows

    if args.push:
        from datasets import Dataset, DatasetDict

        ds = DatasetDict({hf_split: Dataset.from_list(combined[split]) for split, hf_split in SPLITS.items()})
        ds.push_to_hub(REPO_ID)
        print(f"Pushed to https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()

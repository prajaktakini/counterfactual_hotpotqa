import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import counterfactual_messages


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def batch_generate(model, tokenizer, messages_list, max_new_tokens, device):
    prompts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(out[i][input_len:], skip_special_tokens=True)
        for i in range(len(prompts))
    ]


STRING_FIELDS = [
    "entity_1", "entity_2", "attribute", "property", "value_1", "value_2",
    "value_span_1", "value_span_2", "intervention_entity", "original_fact",
    "counterfactual_value", "counterfactual_fact", "changed_span",
    "counterfactual_answer", "original_answer", "comparison_type", "reason",
]


def normalize_result(result):
    if not isinstance(result, dict):
        return result
    for key in STRING_FIELDS:
        value = result.get(key)
        if value is not None and not isinstance(value, str):
            result[key] = str(value)
    return result


def intervention_type_for(comparison_type, direction):
    if comparison_type == "same_attribute":
        return "break_equality" if direction == "yes_to_no" else "create_equality"
    return "remove_property" if direction == "yes_to_no" else "add_property"


def verify_counterfactual(candidate, result):
    checks = {
        "entity_names_unchanged": False,
        "intervention_entity_valid": False,
        "original_fact_matches_source": False,
        "changed_span_present": False,
        "minimal_edit": False,
        "fact_actually_changed": False,
        "answer_flipped": False,
    }

    if not result or not result.get("valid"):
        return False, checks, "llm_reported_invalid_or_unparseable"

    entity_1 = result.get("entity_1")
    entity_2 = result.get("entity_2")
    candidate_entities = {e["entity"] for e in candidate["entities"]}
    if entity_1 in candidate_entities and entity_2 in candidate_entities and entity_1 != entity_2:
        checks["entity_names_unchanged"] = True

    intervention_entity = result.get("intervention_entity")
    if intervention_entity in (entity_1, entity_2):
        checks["intervention_entity_valid"] = True

    source_fact = next(
        (e["fact"] for e in candidate["entities"] if e["entity"] == intervention_entity), None
    )
    original_fact = result.get("original_fact")
    if source_fact is not None and original_fact == source_fact:
        checks["original_fact_matches_source"] = True

    changed_span = result.get("changed_span")
    counterfactual_value = result.get("counterfactual_value")
    counterfactual_fact = result.get("counterfactual_fact")

    if original_fact and changed_span and changed_span in original_fact:
        checks["changed_span_present"] = True

    if checks["changed_span_present"] and counterfactual_value is not None and counterfactual_fact:
        expected_fact = original_fact.replace(changed_span, counterfactual_value, 1)
        if expected_fact == counterfactual_fact:
            checks["minimal_edit"] = True

    if original_fact and counterfactual_fact and original_fact != counterfactual_fact:
        checks["fact_actually_changed"] = True

    expected_answer = "no" if candidate["answer"] == "yes" else "yes"
    if result.get("counterfactual_answer") == expected_answer:
        checks["answer_flipped"] = True

    all_pass = all(checks.values())
    reason = "ok" if all_pass else "failed: " + ",".join(k for k, v in checks.items() if not v)
    return all_pass, checks, reason


def build_final_record(candidate, result, direction):
    comparison_type = result.get("comparison_type")
    return {
        "id": candidate["id"],
        "question": candidate["question"],
        "original_answer": candidate["answer"],
        "counterfactual_answer": result.get("counterfactual_answer"),
        "comparison_type": comparison_type,
        "entity_1": result.get("entity_1"),
        "entity_2": result.get("entity_2"),
        "attribute": result.get("attribute") or result.get("property"),
        "original_value_1": result.get("value_1"),
        "original_value_2": result.get("value_2"),
        "intervention_entity": result.get("intervention_entity"),
        "original_fact": result.get("original_fact"),
        "counterfactual_fact": result.get("counterfactual_fact"),
        "original_span": result.get("changed_span"),
        "counterfactual_span": result.get("counterfactual_value"),
        "intervention_type": intervention_type_for(comparison_type, direction),
        "is_synthetic": True,
        "validation": {"valid": True, "reason": "programmatic", "checks": result["_checks"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=1000)
    args = ap.parse_args()

    candidates = load_jsonl(args.input)

    out_path = Path(args.output)
    done_ids = set()
    if out_path.exists():
        done_ids = {r["id"] for r in load_jsonl(out_path)}
    pending = [c for c in candidates if c["id"] not in done_ids]

    print(f"Input candidates: {len(candidates)}")
    print(f"Already completed (resumed, skipped): {len(done_ids)}")
    print(f"Pending this run: {len(pending)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model = model.to(device)
    model.eval()

    stats = {
        "structural_valid": 0,
        "structural_invalid": 0,
        "orig_yes": 0,
        "orig_no": 0,
        "yes_to_no": 0,
        "no_to_yes": 0,
        "same_attribute": 0,
        "shared_property": 0,
        "validation_passed": 0,
        "validation_failed": 0,
    }

    out_f = open(out_path, "a")

    for batch in tqdm(list(chunked(pending, args.batch_size)), desc="batches"):
        for c in batch:
            stats["orig_yes" if c["answer"] == "yes" else "orig_no"] += 1

        texts = batch_generate(
            model, tokenizer, [counterfactual_messages(c) for c in batch],
            args.max_new_tokens, device,
        )
        results = [normalize_result(extract_json(t)) for t in texts]

        for c, result in zip(batch, results):
            if result is not None and result.get("valid"):
                stats["structural_valid"] += 1
            else:
                stats["structural_invalid"] += 1
                continue

            try:
                passed, checks, _ = verify_counterfactual(c, result)
            except Exception as e:
                print(f"\nWARNING: verify_counterfactual crashed on id={c['id']}: {e}")
                stats["validation_failed"] += 1
                continue
            if not passed:
                stats["validation_failed"] += 1
                continue
            stats["validation_passed"] += 1

            comparison_type = result.get("comparison_type")
            stats["same_attribute" if comparison_type == "same_attribute" else "shared_property"] += 1

            direction = "yes_to_no" if c["answer"] == "yes" else "no_to_yes"
            stats[direction] += 1

            result["_checks"] = checks
            record = build_final_record(c, result, direction)
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    out_f.close()

    print("\n--- Statistics (this run) ---")
    print("Input candidates:", len(candidates))
    print("Pending processed this run:", len(pending))
    print("Valid structural candidates:", stats["structural_valid"])
    print("Invalid candidates:", stats["structural_invalid"])
    print()
    print("Original YES:", stats["orig_yes"])
    print("Original NO:", stats["orig_no"])
    print()
    print("YES -> NO:", stats["yes_to_no"])
    print("NO -> YES:", stats["no_to_yes"])
    print()
    print("Same attribute:", stats["same_attribute"])
    print("Shared property:", stats["shared_property"])
    print()
    print("Validation passed:", stats["validation_passed"])
    print("Validation failed:", stats["validation_failed"])


if __name__ == "__main__":
    main()

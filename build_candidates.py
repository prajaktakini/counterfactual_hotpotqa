import re
import json
from datasets import load_dataset


def lexical_question_type(question):
    q = question.lower().strip()
    if re.search(r"\bsame\b", q):
        return "same_attribute_candidate"
    if re.search(r"\bboth\b", q):
        return "shared_property_candidate"
    return "other"


def extract_entity_facts(row):
    titles = list(row["supporting_facts"]["title"])
    sent_ids = list(row["supporting_facts"]["sent_id"])
    ctx_titles = list(row["context"]["title"])
    ctx_sentences = row["context"]["sentences"]

    entities = []
    for title, sent_id in zip(titles, sent_ids):
        if title not in ctx_titles:
            return None
        idx = ctx_titles.index(title)
        sents = ctx_sentences[idx]
        sent_id = int(sent_id)
        if sent_id < 0 or sent_id >= len(sents):
            return None
        entities.append({"entity": title, "fact": sents[sent_id]})
    return entities


def build_phase1(df, split_name):
    yesno = df[df["answer"].str.lower().isin(["yes", "no"])].copy()
    print(f"\n=== {split_name} yes/no ===")
    print("total yes/no:", len(yesno))
    print(yesno["answer"].str.lower().value_counts())
    print("\nhotpotqa type distribution:")
    print(yesno["type"].value_counts())

    yesno["lexical_type"] = yesno["question"].apply(lexical_question_type)
    print(f"\n{split_name} lexical_type distribution:")
    print(yesno["lexical_type"].value_counts())

    candidates = yesno[yesno["lexical_type"].isin(
        ["same_attribute_candidate", "shared_property_candidate"]
    )]

    records = []
    for _, row in candidates.iterrows():
        entities = extract_entity_facts(row)
        if entities is None or len(entities) < 2:
            continue
        records.append({
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"].lower(),
            "hotpotqa_type": row["type"],
            "level": row["level"],
            "lexical_type": row["lexical_type"],
            "entities": entities,
            "supporting_facts": {
                "title": list(row["supporting_facts"]["title"]),
                "sent_id": [int(s) for s in row["supporting_facts"]["sent_id"]],
            },
        })

    same_n = sum(1 for r in records if r["lexical_type"] == "same_attribute_candidate")
    shared_n = sum(1 for r in records if r["lexical_type"] == "shared_property_candidate")
    print(f"\n{split_name} phase1 summary:")
    print("  total yes/no:", len(yesno))
    print("  same_attribute candidates (extracted):", same_n)
    print("  shared_property candidates (extracted):", shared_n)
    print("  dropped (fact extraction failed):", len(candidates) - len(records))

    return records


def save_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    train_dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    val_dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    train_df = train_dataset.to_pandas()
    val_df = val_dataset.to_pandas()

    print("train size:", len(train_df))
    print("validation size:", len(val_df))
    print("columns:", list(train_df.columns))

    print("\ntrain answer distribution (top 10):")
    print(train_df["answer"].str.lower().value_counts().head(10))
    print("\ntrain type distribution:")
    print(train_df["type"].value_counts())

    print("\nvalidation answer distribution (top 10):")
    print(val_df["answer"].str.lower().value_counts().head(10))
    print("\nvalidation type distribution:")
    print(val_df["type"].value_counts())

    train_records = build_phase1(train_df, "train")
    val_records = build_phase1(val_df, "validation")

    save_jsonl(train_records, "phase1_candidates_train.jsonl")
    save_jsonl(val_records, "phase1_candidates_val.jsonl")

    print(f"\nSaved {len(train_records)} train candidates -> phase1_candidates_train.jsonl")
    print(f"Saved {len(val_records)} val candidates -> phase1_candidates_val.jsonl")


if __name__ == "__main__":
    main()

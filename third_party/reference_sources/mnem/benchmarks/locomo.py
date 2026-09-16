"""Evaluate mnem on the LoCoMo long-conversation benchmark (retrieval recall).

LoCoMo (snap-research/locomo) annotates each question with the exact dialogue
turns that answer it ("evidence"). We load every turn as a memory, then for each
question measure whether the evidence turn(s) appear in the top-k recall. This is
a deterministic, model-free retrieval@k metric — an honest lower bound that
isolates the memory layer from any answer-generating LLM.

Two modes:
    --mode lexical    mnem's zero-dependency IDF token matching (default)
    --mode semantic   mnem's built-in HashingEmbedder (subword, still no model)

The dataset is CC BY-NC 4.0 and is NOT bundled here. Download it and point at it:

    curl -fsSL -o locomo10.json \
      https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
    python benchmarks/locomo.py --data locomo10.json --mode semantic

With no --data, a tiny synthetic sample runs so you can see the methodology.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

sys.path.insert(0, ".")
from mnem import Memory  # noqa: E402
from mnem.semantic import HashingEmbedder  # noqa: E402

CATEGORY_NAMES = {1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "open-domain", 5: "adversarial"}

_SYNTHETIC = [
    {
        "conversation": {
            "session_1": [
                {"speaker": "A", "dia_id": "D1:1", "text": "I adopted a beagle named Pepper last week."},
                {"speaker": "B", "dia_id": "D1:2", "text": "I started a new job at a bakery in Turin."},
            ],
            "session_2": [{"speaker": "A", "dia_id": "D2:1", "text": "Pepper had her first vet visit on 3 June."}],
        },
        "qa": [
            {"question": "What is the name of A's dog?", "answer": "Pepper", "evidence": ["D1:1"], "category": 1},
            {"question": "When did Pepper visit the vet?", "answer": "3 June", "evidence": ["D2:1"], "category": 3},
            {"question": "Where does B work?", "answer": "a bakery in Turin", "evidence": ["D1:2"], "category": 1},
        ],
    }
]


def load(path):
    if not path:
        return _SYNTHETIC
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def turns_of(conversation):
    keys = sorted(
        (k for k in conversation if k.startswith("session_") and not k.endswith("date_time")),
        key=lambda k: int(k.split("_")[1]),
    )
    for key in keys:
        for turn in conversation[key]:
            yield turn["dia_id"], f"{turn.get('speaker', '')}: {turn['text']}".strip(": ").strip()


def _lexical_ranker(turns):
    memory = Memory(supersede_threshold=0.99)  # retrieval mode: minimal supersession
    for dia_id, text in turns:
        memory.add(text, source=dia_id)

    def rank(question, max_k):
        return [fact.source for fact in memory.recall_facts(question, k=max_k)]

    return rank


def _cosine_ranker(turns):
    import numpy as np

    embed = HashingEmbedder()
    matrix = np.array([embed(text) for _, text in turns], dtype="float32")
    ids = [dia_id for dia_id, _ in turns]

    def rank(question, max_k):
        q = np.array(embed(question), dtype="float32")
        order = np.argsort(-(matrix @ q))[:max_k]
        return [ids[i] for i in order]

    return rank


def evaluate(samples, ks, mode):
    max_k = max(ks)
    hits = {k: defaultdict(int) for k in ks}
    totals = defaultdict(int)
    for sample in samples:
        turns = list(turns_of(sample["conversation"]))
        ranker = _cosine_ranker(turns) if mode == "semantic" else _lexical_ranker(turns)
        for qa in sample.get("qa", []):
            evidence = set(qa.get("evidence") or [])
            if not evidence:  # adversarial / unanswerable questions carry no evidence
                continue
            category = qa.get("category", 0)
            totals[category] += 1
            retrieved = ranker(qa["question"], max_k)
            for k in ks:
                if evidence & set(retrieved[:k]):
                    hits[k][category] += 1
    return hits, totals


def main():
    parser = argparse.ArgumentParser(description="mnem on LoCoMo (retrieval recall)")
    parser.add_argument("--data", help="path to locomo10.json (CC BY-NC, download separately)")
    parser.add_argument("--mode", choices=["lexical", "semantic"], default="lexical")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10])
    args = parser.parse_args()

    samples = load(args.data)
    hits, totals = evaluate(samples, args.k, args.mode)
    n = sum(totals.values())
    source = args.data or "SYNTHETIC sample (pass --data locomo10.json for the real run)"

    print(f"mnem on LoCoMo — retrieval recall@k   (mode={args.mode})")
    print(f"data: {source}   answerable questions: {n}\n")
    header = "category".ljust(14) + "n".rjust(6) + "".join(f"  R@{k}".rjust(8) for k in args.k)
    print(header)
    print("-" * len(header))
    for category in sorted(totals):
        row = CATEGORY_NAMES.get(category, str(category)).ljust(14) + str(totals[category]).rjust(6)
        for k in args.k:
            row += f"{100 * hits[k][category] / totals[category]:7.1f}%"
        print(row)
    print("-" * len(header))
    overall = "OVERALL".ljust(14) + str(n).rjust(6)
    for k in args.k:
        overall += f"{100 * sum(hits[k].values()) / n:7.1f}%"
    print(overall)


if __name__ == "__main__":
    main()

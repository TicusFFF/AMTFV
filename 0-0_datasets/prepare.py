"""Prepare the exact paper inputs from the datasets listed in dataset_manifest.json."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "dataset_manifest.json").read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def prepare(key, records, destination):
    spec = MANIFEST[key]
    lookup = {}
    for row in records:
        question = next((row[k] for k in ("q", "problem", "prompt", "query", "question") if row.get(k)), None)
        answer = next((row[k] for k in ("answer", "A_original", "A", "target") if row.get(k) is not None), None)
        if question and answer is not None:
            lookup[digest(question), digest(answer)] = {"q": question, "A_original": answer, "answer": answer}
    normalized = []
    for item in spec["items"]:
        try:
            normalized.append(lookup[item["question_sha256"], item["answer_sha256"]])
        except KeyError:
            raise ValueError(f"{item['id']}: the question or reference answer differs from the paper input.") from None
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(normalized, indent=2, ensure_ascii=False) + "\n"
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8-sig")) != normalized:
            raise FileExistsError(f"Existing dataset differs: {destination}")
    else:
        with destination.open("x",encoding="utf-8") as handle:
            handle.write(text)
    print(f"{key}: verified {len(normalized)} problems")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=MANIFEST, required=True)
    parser.add_argument("--input", type=Path, help="Local JSON array of source rows; omit to load the stated Hugging Face dataset.")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    spec = MANIFEST[args.dataset]
    if args.input:
        rows = json.loads(args.input.read_text(encoding="utf-8-sig"))
    elif args.dataset == "demo":
        rows = json.loads((ROOT / spec["file"]).read_text(encoding="utf-8-sig"))
    else:
        from datasets import load_dataset
        rows = load_dataset(spec["source"], split=spec["split"])
    prepare(args.dataset, rows, args.output_dir / spec["file"])


if __name__ == "__main__":
    main()

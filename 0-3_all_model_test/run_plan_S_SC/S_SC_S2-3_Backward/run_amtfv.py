"""Run AMTFV with shared Flash answers and grade the saved final responses."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCORER = ROOT / "0-2-2_unified_evaluation"
DATA_ROOT = ROOT / "0-0_datasets"
DATASETS = json.loads((DATA_ROOT / "dataset_manifest.json").read_text(encoding="utf-8"))


def checked_inputs(dataset, questions, initial, limit):
    rows = json.loads(questions.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{dataset}: expected a nonempty list of questions")
    records = [json.loads(line) for line in initial.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    by_id = {row["id"]: row for row in records}
    if len(by_id) != len(records):
        raise ValueError(f"{dataset}: duplicate initial-answer IDs")
    ids = []
    for i, question in enumerate(rows[:limit]):
        key = f"{dataset}_{i}"
        answer = by_id.get(key, {})
        prompt = next((question[k] for k in ("query", "problem", "q", "question", "prompt") if question.get(k)), None)
        if not prompt or answer.get("query") != prompt:
            raise ValueError(f"{key}: question or matching initial answer missing")
        if not isinstance(answer.get("response"), str) or not answer["response"].strip() or "pred" not in answer:
            raise ValueError(f"{key}: incomplete initial answer")
        ids.append(key)
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Enable model calls, tool execution and final scoring.")
    parser.add_argument("--dataset", choices=DATASETS, action="append", help="Select a dataset (default: demo); repeat to combine.")
    parser.add_argument("--limit", type=int, help="Take the first N questions per selected dataset.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent questions per dataset.")
    parser.add_argument("--initial-dir", type=Path, default=ROOT / "runs/initial", help="Directory of matching deepseek-flash initial-answer JSONL files.")
    parser.add_argument("--output", type=Path, help="Create a new directory for this run.")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    if args.dataset and len(set(args.dataset)) != len(args.dataset):
        parser.error("select each dataset once")

    todo = []
    for name in args.dataset or ["demo"]:
        questions = DATA_ROOT / DATASETS[name]["file"]
        initial = args.initial_dir / f"deepseek-flash_{name}_first_response.jsonl"
        for path in (questions, initial):
            if not path.is_file():
                parser.error(f"missing {path}; see the data preparation instructions in README.md")
        ids = checked_inputs(name, questions, initial, args.limit)
        print(f"{name}: {len(ids)} questions matched to shared initial answers.", flush=True)
        todo.append((name, questions, initial, ids))
    if not args.run:
        print("Input check complete. Add --run to execute and score.")
        return

    missing = [key for key in ("DEEPSEEK_API_KEY",) if not os.environ.get(key)]
    if missing:
        parser.error("set the following environment variables: " + ", ".join(missing))
    sys.path.insert(0, str(ROOT))
    from amt_runner import run_dataset

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = (args.output or ROOT / "runs" / stamp).resolve()
    out.mkdir(parents=True, exist_ok=False)
    predictions = out / "model_outputs"
    predictions.mkdir()
    print(f"Run directory: {out}", flush=True)
    prepare = [sys.executable, str(SCORER / "prepare_inputs.py")]
    for name, questions, initial, ids in todo:
        path = predictions / f"{name}.json"
        run_dataset(dataset_key=name, data_path=questions, initial_response_file=initial,
                    output_path=path, workers=args.workers, start=0, end=len(ids))
        logs = json.loads(path.read_text(encoding="utf-8"))["logs"]
        if len(logs) != len(ids) or {row["id"] for row in logs} != set(ids):
            raise ValueError(f"{name}: generated record IDs differ from the selected questions")
        prepare.extend(["--dataset", name, str(questions), str(path)])
    inputs = out / "evaluation_input.json"
    scores = out / "evaluation.json"
    subprocess.run([*prepare, "--output", str(inputs)], check=True)
    subprocess.run([sys.executable, str(SCORER / "evaluate.py"), "--run",
                    "--input_json", str(inputs), "--output_json", str(scores),
                    "--cache_dir", str(out / "evaluation_cache")], check=True)
    summary = json.loads(scores.read_text(encoding="utf-8"))
    if summary.get("acc") is None or summary["n_completed"] != sum(len(entry[3]) for entry in todo):
        raise RuntimeError(f"Evaluation incomplete; inspect {scores}")
    print(f"Accuracy: {summary['n_correct']}/{summary['n_completed']} ({summary['acc']:.2f}%). Results: {scores}")


if __name__ == "__main__":
    main()

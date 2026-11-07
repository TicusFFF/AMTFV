"""Generate shared initial responses with official DeepSeek Flash or score a saved batch."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "0-0_datasets"
MANIFEST = json.loads((DATA / "dataset_manifest.json").read_text(encoding="utf-8"))
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-flash"


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def save_text(path, text):
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                            prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        handle.write(text)
        name = handle.name
    os.replace(name, path)


@contextmanager
def batch_lock(path):
    """Release the OS lock on exit, including process termination."""
    with path.open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def request(client, prompt):
    for attempt in range(4):
        try:
            result = client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0,
                extra_body={"thinking": {"type": "disabled"}},
            )
            choice = result.choices[0]
            content = choice.message.content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Empty completion.")
        except Exception:
            if attempt == 3:
                raise
            time.sleep(10 * attempt)
            continue
        if not result.model:
            raise ValueError("The endpoint did not identify the returned model.")
        return content, {
            "requested_model": MODEL, "returned_model": result.model,
            "generated_at": datetime.now(timezone.utc).isoformat(), "request_id": result.id,
            "system_fingerprint": getattr(result, "system_fingerprint", None),
            "finish_reason": choice.finish_reason, "prompt_sha256": digest(prompt),
            "response_sha256": digest(content),
        }


def generate_one(client, cid, question, template):
    response, first_call = request(client, question)
    row = {"id": cid, "query": question, "response": response, "pred": ""}
    calls = [first_call]
    if first_call["finish_reason"] == "length":
        row["finish_reason"] = "length"
    else:
        candidate, extract_call = request(client, template.format(query=question, response=response))
        row["pred"] = candidate.strip()
        calls.append(extract_call)
    return {"row": row, "calls": calls, "prediction_sha256": digest(row["pred"])}


def validate_record(record, cid, question):
    row, calls = record["row"], record["calls"]
    expected_calls = 1 if row.get("finish_reason") == "length" else 2
    if (row["id"] != cid or row["query"] != question or not row["response"].strip()
            or len(calls) != expected_calls or not all(call.get("returned_model") and
            call.get("requested_model") == MODEL for call in calls)
            or calls[0]["prompt_sha256"] != digest(question)
            or calls[0]["response_sha256"] != digest(row["response"])
            or record["prediction_sha256"] != digest(row["pred"])):
        raise ValueError(f"{cid}: input or saved completion does not match its recorded provenance.")


def read_batch(questions, dataset, out, workers, evaluate_only):
    output = out / f"{MODEL}_{dataset}_first_response.jsonl"
    provenance = output.with_suffix(".version.json")
    template = Path(__file__).with_name("initial_extraction_prompt.txt").read_text(encoding="utf-8")
    contract = {"dataset": dataset, "endpoint": BASE_URL, "model": MODEL, "temperature": 0,
                "thinking": "disabled", "output_limit": "provider default",
                "questions_sha256": digest(json.dumps(questions, ensure_ascii=False)),
                "extraction_prompt_sha256": digest(template)}
    records = {}
    if provenance.exists():
        state = json.loads(provenance.read_text(encoding="utf-8"))
        if state["contract"] != contract:
            raise ValueError(f"Stored settings or questions differ in {provenance}.")
        records = state["records"]
    if (output.exists() or evaluate_only) and len(records) != len(questions):
        raise ValueError("The saved batch lacks its complete, matching provenance.")
    if set(records) - set(questions):
        raise ValueError("Saved records contain unexpected question IDs.")
    for cid, record in records.items():
        validate_record(record, cid, questions[cid])
    missing = [cid for cid in questions if cid not in records]
    if missing:
        from openai import OpenAI
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise ValueError("Set DEEPSEEK_API_KEY to generate the missing initial responses.")
        client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=BASE_URL, timeout=2400, max_retries=0)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(generate_one, client, cid, questions[cid], template): cid for cid in missing}
            for job in as_completed(jobs):
                cid = jobs[job]
                rec = job.result()
                validate_record(rec, cid, questions[cid])
                records[cid] = rec
                save_text(provenance, json.dumps({"contract": contract, "records": records}, ensure_ascii=False))
    rows = [records[cid]["row"] for cid in questions]
    if output.exists():
        saved = [json.loads(line) for line in output.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if saved != rows:
            raise ValueError(f"Saved answers differ from their provenance: {output}")
    else:
        save_text(output, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{output}: {len(rows)} responses, {len(missing)} generated, {len(rows) - len(missing)} reused.")
    return rows


def evaluate(rows, dataset, data_path, out):
    scorer = ROOT / "0-2-2_unified_evaluation"
    spec = spec_from_file_location("prepare_amtfv_inputs", scorer / "prepare_inputs.py")
    preparer = module_from_spec(spec)
    spec.loader.exec_module(preparer)
    native = out / f"{dataset}.initial-infer.json"
    logs = [{"id": r["id"], "query": r["query"], "final_response": r["response"]} for r in rows]
    save_text(native, json.dumps({"args": {"dataset_key": dataset}, "logs": logs}, ensure_ascii=False))
    cases = preparer.prepare_inputs([(dataset, data_path, native)])
    inputs = out / f"{dataset}.initial-eval-input.json"
    save_text(inputs, json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
    data_hash = hashlib.sha256(inputs.read_bytes()).hexdigest()
    report = out / f"{dataset}.initial-eval.json"
    if report.exists():
        previous = json.loads(report.read_text(encoding="utf-8"))
        if previous.get("source_sha256") != data_hash:
            raise ValueError(f"Existing evaluation uses different inputs: {report}")
        if previous.get("status") == "completed":
            print(f"{report}: reused completed V5/V2 evaluation ({previous['n_correct']}/{len(rows)}).")
            return
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("Set DEEPSEEK_API_KEY for answer evaluation.")
    cache = out / f"{dataset}.initial-eval-cache"
    checkpoint = cache / "evaluation_checkpoint.jsonl"
    result = report.with_name(report.stem + "." + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    cmd = [sys.executable, str(scorer / "evaluate.py"), "--input_json", str(inputs),
           "--output_json", str(result), "--cache_dir", str(cache), "--run"]
    if checkpoint.exists():
        cmd += ["--resume", "--retry_pending"]
    ran = subprocess.run(cmd)
    if result.exists():
        os.replace(result, report)
    if ran.returncode:
        raise SystemExit(ran.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Score the saved batch with shared V5 extraction and V2 grading, without generating answers.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Use at least one worker.")
    spec = MANIFEST[args.dataset]
    path = DATA / spec["file"]
    parsed = json.loads(path.read_text(encoding="utf-8-sig"))
    if len(parsed) != len(spec["items"]):
        parser.error("Input count differs from the dataset manifest.")
    questions = {}
    for row, item in zip(parsed, spec["items"]):
        if digest(row["q"]) != item["question_sha256"] or digest(str(row["answer"])) != item["answer_sha256"]:
            parser.error(f"{item['id']}: input differs from the dataset manifest.")
        questions[item["id"]] = row["q"]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with batch_lock(out / f".{args.dataset}.lock"):
        rows = read_batch(questions, args.dataset, out, args.workers, args.evaluate_only)
        evaluate(rows, args.dataset, path, out)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# Evaluation rule: this shared 0-2_step_judge package is the only hard-test evaluator.
# Do not create project-local evaluation methods or metrics.
import argparse
import copy
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys

from three_step_judge import DEFAULT_CACHE_PATH, JUDGE_IDENTITY, judge_many


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_answer_extractor():
    name = 'spq_public_answer_extractor'
    if name not in sys.modules:
        path = REPO_ROOT / '0-2-0_answer_extractor/answer_extractor.py'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]

DATASET_EXPECTED_N = {
    "aime_2024": 30,
    "aime_2025": 30,
    "brumo_2025": 30,
    "hmmt_feb_2025": 30,
    "amo_bench": 50,
    "arxivmath_0426": 41,
    "arxivmath_0526": 40,
    "apex_2025": 12,
}


def score_denominator(dataset_key, fallback_n):
    return DATASET_EXPECTED_N.get(dataset_key, fallback_n)


def display_repo_path(value):
    if value in (None, ""):
        return value
    try:
        path = Path(value)
        return str(path.resolve().relative_to(REPO_ROOT))
    except (OSError, ValueError):
        return str(value)


def read_json_or_jsonl(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(x) for x in text.splitlines() if x.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else data


def item_id(dataset_key, i, item):
    return item.get("id") or f"{dataset_key}_{i}"


def item_target(item):
    for key in ("target", "answer", "A", "A_original"):
        if item.get(key) is not None:
            return item.get(key)
    return ""


def load_targets(data_path, dataset_key):
    raw = read_json_or_jsonl(data_path)
    if isinstance(raw, dict):
        raw = raw.get("logs") or raw.get("data") or []
    targets = {}
    for i, item in enumerate(raw):
        target = item_target(item)
        targets[f"{dataset_key}_{i}"] = target
        if item.get("id") is not None:
            targets[str(item.get("id"))] = target
    return targets


def pick_initial(log):
    turns = log.get("turns") or []
    if turns and turns[0].get("result") is not None:
        return turns[0].get("result")
    return log.get("initial_result") or log.get("initial") or ""


def pick_final(log):
    if log.get("final_result") is not None:
        return log.get("final_result")
    if log.get("final") is not None:
        return log.get("final")
    turns = log.get("turns") or []
    if turns and turns[-1].get("result") is not None:
        return turns[-1].get("result")
    return ""


def pick_initial_response(log):
    turns = log.get("turns") or []
    if turns and "response" in turns[0]:
        return turns[0].get("response")
    if "initial_response" in log:
        return log.get("initial_response")
    if "response" in log:
        return log.get("response")
    return None


def pick_final_response(log):
    if "final_response" in log:
        return log.get("final_response")
    turns = log.get("turns") or []
    if turns and "response" in turns[-1]:
        return turns[-1].get("response")
    if "response" in log:
        return log.get("response")
    return None


def is_explicit_empty_response(value):
    return value is not None and not str(value).strip()


def empty_response_judge(gold, answer):
    return {
        "equivalent": False,
        "method": "empty_response",
        "answer_1": gold,
        "answer_2": answer,
    }


def make_eval_summary_from_logs(
    data_path,
    logs,
    dataset_key="",
    method="",
    model="",
    infered_path="",
    workers=1,
    cache_path=DEFAULT_CACHE_PATH,
):
    # Historical/internal evaluation: S-series uses this DURING skill selection.
    # Do not insert public A8 here. Final reports use make_report_summary_from_logs.
    # Data source rule: data_path should point to public 0-0_datasets, not another experiment folder.
    targets = load_targets(data_path, dataset_key)
    rows = []
    init_pairs = []
    final_pairs = []
    init_pair_row_indices = []
    final_pair_row_indices = []
    init_judges_by_row = []
    final_judges_by_row = []
    for log in logs:
        initial_response = pick_initial_response(log)
        final_response = pick_final_response(log)
        initial_empty_response = is_explicit_empty_response(initial_response)
        final_empty_response = is_explicit_empty_response(final_response)

        gold = log.get("target") if log.get("target") is not None else targets.get(log["id"], "")
        initial = pick_initial(log)
        final = pick_final(log)
        rows.append({
            "id": log["id"],
            "target": gold,
            "initial": initial,
            "final": final,
            "turn_count": len(log.get("turns") or []),
            "initial_empty_response": initial_empty_response,
            "final_empty_response": final_empty_response,
            "empty_response": initial_empty_response or final_empty_response,
        })
        row_index = len(rows) - 1
        init_judges_by_row.append(None)
        final_judges_by_row.append(None)

        if initial_empty_response:
            init_judges_by_row[row_index] = empty_response_judge(gold, initial)
        else:
            init_pair_row_indices.append(row_index)
            init_pairs.append((gold, initial))

        if final_empty_response:
            final_judges_by_row[row_index] = empty_response_judge(gold, final)
        else:
            final_pair_row_indices.append(row_index)
            final_pairs.append((gold, final))

    for row_index, init_judge in zip(
        init_pair_row_indices,
        judge_many(init_pairs, workers=workers, cache_path=cache_path) if init_pairs else [],
    ):
        init_judges_by_row[row_index] = init_judge
    for row_index, final_judge in zip(
        final_pair_row_indices,
        judge_many(final_pairs, workers=workers, cache_path=cache_path) if final_pairs else [],
    ):
        final_judges_by_row[row_index] = final_judge

    for row, init_judge, final_judge in zip(rows, init_judges_by_row, final_judges_by_row):
        row["initial_correct"] = bool(init_judge["equivalent"])
        row["final_correct"] = bool(final_judge["equivalent"])
        row["initial_judge"] = init_judge
        row["final_judge"] = final_judge

    denominator = score_denominator(dataset_key, len(rows))
    n = denominator
    initial_correct = sum(r["initial_correct"] for r in rows)
    final_correct = sum(r["final_correct"] for r in rows)
    initial_empty_response_ids = [r["id"] for r in rows if r["initial_empty_response"]]
    final_empty_response_ids = [r["id"] for r in rows if r["final_empty_response"]]
    empty_response_ids = [r["id"] for r in rows if r["empty_response"]]
    return {
        "judge": "three_step_exact_float_llm",
        "judge_config": JUDGE_IDENTITY,
        "method": method,
        "model": model,
        "dataset_key": dataset_key,
        "data_path": display_repo_path(data_path),
        "infered_path": display_repo_path(infered_path),
        "n": n,
        "score_denominator": denominator,
        "initial_correct": initial_correct,
        "final_correct": final_correct,
        "initial_acc": round(initial_correct / denominator, 4) if denominator else 0,
        "final_acc": round(final_correct / denominator, 4) if denominator else 0,
        "skipped_empty_response_count": 0,
        "skipped_empty_response_ids": [],
        "initial_empty_response_count": len(initial_empty_response_ids),
        "initial_empty_response_ids": initial_empty_response_ids,
        "final_empty_response_count": len(final_empty_response_ids),
        "final_empty_response_ids": final_empty_response_ids,
        "empty_response_count": len(empty_response_ids),
        "empty_response_ids": empty_response_ids,
        "improved_ids": [r["id"] for r in rows if not r["initial_correct"] and r["final_correct"]],
        "regressed_ids": [r["id"] for r in rows if r["initial_correct"] and not r["final_correct"]],
        "rows": rows,
    }


def make_report_summary_from_logs(data_path, logs, dataset_key="", method="", model="",
                                  infered_path="", workers=1, cache_path=DEFAULT_CACHE_PATH):
    """Reporting only: same A8 for public initial and selected final responses.

    Never overwrite generated logs, change candidate selection, or reuse old
    initial scores computed with a different extraction protocol. Missing full
    responses/technical failures leave the report pending, with a fixed pool.
    """
    logs = list(logs)
    if not logs:
        raise ValueError('No per-question response records; a summary/selection table is not an answer log.')
    if len({str(log['id']) for log in logs}) != len(logs):
        raise ValueError('Duplicate question IDs; refusing to publish a duplicated evaluation pool.')
    extractor = load_answer_extractor()
    if cache_path == DEFAULT_CACHE_PATH:
        cache_path = (REPO_ROOT / 'outputs' / 'initial_evaluation' / 'judge_cache.jsonl')
    extraction_cache = Path(cache_path).with_suffix('.a8.sqlite3') if cache_path else None
    raw = read_json_or_jsonl(data_path)
    if isinstance(raw, dict):
        raw = raw.get('logs') or raw.get('data') or []
    questions = {}
    for i, item in enumerate(raw):
        question = next((item[k] for k in ('q', 'problem', 'question', 'query')
                         if isinstance(item.get(k), str) and item[k].strip()), '')
        questions[f'{dataset_key}_{i}'] = question
        if item.get('id') is not None:
            questions[str(item['id'])] = question
    prepared, audits, pending = [], [], []
    for log in logs:
        clone = copy.deepcopy(log)
        query = questions.get(str(log['id']), '')
        audit = {'id': log['id'], 'initial_before_extraction': pick_initial(log),
                 'final_before_extraction': pick_final(log)}
        for stage, response in (('initial', pick_initial_response(log)), ('final', pick_final_response(log))):
            result = extractor.extract_answer(query, response, cache_path=extraction_cache)
            audit[stage + '_extraction'] = result
            if result['status'] == 'pending':
                pending.append({'id': log['id'], 'stage': stage, 'reason': result.get('reason', '')})
            clone[stage + '_result'] = result['answer']
            # The historical evaluator prioritizes turns[0] for initial answers.
            if stage == 'initial' and clone.get('turns'):
                clone['turns'][0]['result'] = result['answer']
        prepared.append(clone)
        audits.append(audit)
    metadata = {'report_protocol': 'public_a8_then_shared_three_step_v1',
                'extraction_config': extractor.identity(), 'judge_config': JUDGE_IDENTITY,
                'extraction_cache': display_repo_path(extraction_cache),
                'method': method, 'model': model, 'dataset_key': dataset_key,
                'data_path': display_repo_path(data_path), 'infered_path': display_repo_path(infered_path),
                'n_sample': len(prepared), 'score_denominator': score_denominator(dataset_key, len(prepared))}
    if pending:
        return {**metadata, 'status': 'pending', 'pending': pending, 'rows': audits,
                'initial_correct': None, 'final_correct': None, 'initial_acc': None,
                'final_acc': None, 'paper_acc': None}
    summary = make_eval_summary_from_logs(data_path, prepared, dataset_key=dataset_key,
        method=method, model=model, infered_path=infered_path, workers=workers, cache_path=cache_path)
    for row, audit in zip(summary['rows'], audits):
        row.update(audit)
    # Preserve the judge's diagnostics, but never advertise unresolved technical
    # failures as a valid comparison score. The judge algorithm is unchanged.
    judge_pending = [{'id': r['id'], 'stage': stage, 'reason': 'judge_error'}
        for r in summary['rows'] for stage in ('initial', 'final')
        if r[stage + '_judge'].get('judge_error') or r[stage + '_judge'].get('error')]
    summary.update(metadata, status='pending' if judge_pending else 'completed', pending=judge_pending)
    if judge_pending:
        summary.update(initial_correct=None, final_correct=None, initial_acc=None, final_acc=None, paper_acc=None)
    return summary


def make_eval_summary(data_path, infered_path, dataset_key="", method="", model="", workers=1, cache_path=DEFAULT_CACHE_PATH):
    source = Path(infered_path)
    if source.is_dir():
        case_dir = source / 'cases' if (source / 'cases').is_dir() else source
        run = [read_json_or_jsonl(p) for p in sorted(case_dir.glob('*.json'))
               if p.stem.startswith(dataset_key + '_')]
        if not run:
            raise ValueError('No matching saved cases found; no inference will be started.')
    else:
        run = read_json_or_jsonl(source)
    logs = run.get("logs", []) if isinstance(run, dict) else run
    method_name = method or (run.get("method") if isinstance(run, dict) else "")
    summary = make_report_summary_from_logs(
        data_path=data_path,
        logs=logs,
        dataset_key=dataset_key,
        method=method_name,
        model=model or (run.get("model") if isinstance(run, dict) else ""),
        infered_path=infered_path,
        workers=workers,
        cache_path=cache_path,
    )
    # The old initial-eval JSON is deliberately NOT overlaid onto an A8 report.
    # Both sides must use the same extraction protocol; originals stay untouched.
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--infered_path", "--infer_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--dataset_key", default="")
    p.add_argument("--method", default="")
    p.add_argument("--model", default="")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--cache_path", default=str(DEFAULT_CACHE_PATH))
    args = p.parse_args()

    out = Path(args.output_path)
    if out.exists():
        raise FileExistsError('Report output exists; choose a new path to preserve historical scores.')

    summary = make_eval_summary(
        data_path=args.data_path,
        infered_path=args.infered_path,
        dataset_key=args.dataset_key,
        method=args.method,
        model=args.model,
        workers=args.workers,
        cache_path=args.cache_path,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=2))
    if summary.get('status') == 'pending':
        raise SystemExit(2)


if __name__ == "__main__":
    main()

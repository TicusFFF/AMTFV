"""One explicit final-report entry for B/RSI/EN/S; never changes inference.

Input JSON: {"expected_ids": [...], "cases": [{"id": ..., "query": ...,
"response": <full selected answer>, "target": <reference answer>}, ...]}.
Run --input_json ... --output_json NEW_FILE --cache_dir RUN_CACHE --run.
Without --run only validate. Old test entries are NOT silently redirected.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location('unified_report_' + name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


extractor = load('answer_extractor')
judge = load('three_step_judge')
runtime = load('runtime_diagnostics')
# Only this entry's module-local transports are instrumented. The original
# extraction/judgement files, request payloads and cache identities stay intact.
runtime.install(extractor, 'extraction')
runtime.install(judge, 'judge')
_JOURNAL_LOCK = threading.Lock()


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def _append(path, record):
    """Durable append; storage failures must stop, never masquerade as wrong."""
    try:
        with _JOURNAL_LOCK, Path(path).open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise runtime.AuditStorageError('Could not persist evaluation audit; batch stopped.') from exc


def identity():
    ext = extractor.identity()
    if any(ext[k] != judge.JUDGE_IDENTITY[k] for k in ('api_profile', 'model', 'api_model', 'temperature', 'extra_body')):
        raise ValueError('Extraction and judgement configuration drift; do not run.')
    return {'protocol': 'unified_final_report_v1_20261001', 'extractor': ext,
            'judge': judge.JUDGE_IDENTITY,
            'entry_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'runtime_sha256': hashlib.sha256((HERE / 'runtime_diagnostics.py').read_bytes()).hexdigest()}


def evaluate_answer(query, response, target, *, cache_dir=None):
    """The extractor never receives target, method name or old score."""
    config = identity()
    cache = Path(cache_dir) if cache_dir is not None else None
    if not isinstance(target, str) or not target.strip():
        raise ValueError('A nonempty reference answer is required; do not infer it.')
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    result = {'status': 'pending', 'correct': None, 'extraction': None, 'judge': None,
              'evaluation_config': config}
    context = {'job_key': _digest([query, response, target])}
    sink = (lambda event: _append(cache / 'requests_engineering.jsonl', event)) if cache else None
    stage = 'extraction'
    with runtime.capture(context, sink=sink) as events:
        try:
            ext = extractor.extract_answer(query, response, cache_path=cache / 'extraction.sqlite3' if cache else None)
            result['extraction'] = ext
            if ext['status'] != 'pending':
                stage = 'judge'
                verdict = judge.judge_answer_equivalence(target, ext['answer'], cache_path=cache / 'judge.jsonl' if cache else None)
                result['judge'] = verdict
                if isinstance(verdict.get('equivalent'), bool):
                    result.update(status='completed', correct=verdict['equivalent'])
        except (runtime.AuditStorageError, sqlite3.Error, MemoryError):
            raise  # Infrastructure corruption is not an isolated answer failure.
        except OSError as exc:
            # Transport errors may be isolated; filesystem errors must stop.
            if not isinstance(exc, (TimeoutError, ConnectionError, extractor.urllib.error.URLError)):
                raise
            result['reason'] = stage + '_request_failed:' + type(exc).__name__
            result['failure'] = runtime.describe_exception(exc, stage)
        except Exception as exc:
            result['reason'] = stage + '_request_failed:' + type(exc).__name__
            result['failure'] = runtime.describe_exception(exc, stage)
    result['request_diagnostics'] = events
    if result['status'] == 'pending':
        last = events[-1] if events and events[-1].get('stage') == stage else None
        if last and last.get('kind') not in (None, 'ok', 'unknown', 'http_response'):
            result['failure'] = last
        elif (result.get('judge') or {}).get('method') == 'judge_error_pending':
            result['failure'] = {'stage': 'judge', 'kind': 'invalid_verdict', 'retryable': True,
                                 'request': last}
        elif last:
            result.setdefault('failure', last)
        result.setdefault('reason', (result.get('extraction') or {}).get('reason') or 'judge_output_pending')
    return result


def validate(document):
    expected, cases = document.get('expected_ids'), document.get('cases')
    if not isinstance(expected, list) or not expected or not all(isinstance(i, str) and i for i in expected):
        raise ValueError('Declare the full nonempty expected_ids pool before any calls.')
    if len(expected) != len(set(expected)) or not isinstance(cases, list):
        raise ValueError('Duplicate expected IDs or invalid cases.')
    found = set()
    for row in cases:
        if not isinstance(row, dict) or row.get('id') not in expected or row['id'] in found:
            raise ValueError('Duplicate or unexpected case ID.')
        found.add(row['id'])
        if not isinstance(row.get('query'), str) or not row['query'].strip():
            raise ValueError('Missing question.')
        if not isinstance(row.get('target'), str) or not row['target'].strip():
            raise ValueError('Missing reference.')
        if not isinstance(row.get('response'), str):
            raise ValueError('Missing full response; an actually empty response must be the string "".')
    if found != set(expected):
        raise ValueError('Incomplete pool: do not reduce denominator or silently omit missing questions.')
    return cases


def _checkpoint_state(path, document, *, resume):
    """Strict manifest matching. Never reinterpret historical/signed ledgers."""
    header = {'type': 'manifest', 'format': 'evaluation_checkpoint_v1',
              'input_sha256': _digest(document), 'evaluation_config': identity()}
    rows = {}
    path = Path(path)
    if path.exists():
        if not resume:
            raise FileExistsError('Checkpoint exists; use explicit --resume with the same input/configuration.')
        raw = path.read_bytes()
        if not raw.endswith(b'\n'):
            raise ValueError('Checkpoint has an interrupted last line; preserved unchanged, do not restart paid requests.')
        lines = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
        if not lines or lines[0] != header:
            raise ValueError('Checkpoint input/configuration mismatch; refusing reuse or overwrite.')
        expected = set(document['expected_ids'])
        inputs = {case['id']: _digest(case) for case in document['cases']}
        config_digest = _digest(header['evaluation_config'])
        for record in lines[1:]:
            if record.get('type') == 'started':
                case_id = record.get('id')
                if case_id not in expected or (rows.get(case_id) or {}).get('status') == 'completed':
                    raise ValueError('Invalid/conflicting started record; preserved unchanged.')
                # A crash may occur after the provider received a paid request,
                # but before its result was committed. Never call this unstarted.
                rows[case_id] = {'id': case_id, 'status': 'pending', 'correct': None,
                                 'extraction': None, 'judge': None,
                                 'reason': 'interrupted_in_flight',
                                 'failure': {'kind': 'interrupted_in_flight', 'retryable': False}}
                continue
            row = record.get('row', {})
            if record.get('type') != 'case' or row.get('id') not in expected:
                raise ValueError('Invalid checkpoint record; preserved unchanged.')
            if record.get('input_sha256') != inputs[row['id']] or record.get('config_sha256') != config_digest:
                raise ValueError('Checkpoint case input/configuration mismatch; preserved unchanged.')
            if row.get('status') not in ('completed', 'pending') or (
                    row['status'] == 'completed' and type(row.get('correct')) is not bool) or (
                    row['status'] == 'pending' and row.get('correct') is not None):
                raise ValueError('Invalid checkpoint outcome; preserved unchanged.')
            previous = rows.get(row['id'])
            if previous and previous['status'] == 'completed' and previous != row:
                raise ValueError('Conflicting completed checkpoint records; refusing to choose one.')
            rows[row['id']] = row
    else:
        if resume:
            raise FileNotFoundError('Cannot resume: checkpoint does not exist.')
        # Exclusive creation prevents accidentally replacing any old ledger.
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(header, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    return rows


def _retryable(row):
    if row.get('status') != 'pending':
        return False
    # A user-requested retry pass is for technical failures only, not valid
    # False, semantic Unknown, or provider rejection/authentication.
    failure = row.get('failure') or {}
    return failure.get('retryable') is True


@contextmanager
def _checkpoint_owner(path):
    if path is None:
        yield
        return
    lock = Path(str(path) + '.lock')
    # Exclusive ownership across processes, not merely a Python thread lock.
    # A lock left by a hard kill must be checked manually; never steal it.
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink()


def evaluate_batch(document, *, cache_dir=None, checkpoint_path=None, resume=False, retry_pending=False):
    validate(document)
    with _checkpoint_owner(checkpoint_path):
        return _evaluate_batch(document, cache_dir=cache_dir, checkpoint_path=checkpoint_path,
                               resume=resume, retry_pending=retry_pending)


def _evaluate_batch(document, *, cache_dir=None, checkpoint_path=None, resume=False, retry_pending=False):
    cases = validate(document)  # Validate the whole pool before any paid requests.
    if (resume or retry_pending) and checkpoint_path is None:
        raise ValueError('Resume/retry requires an explicit checkpoint.')
    if retry_pending and not resume:
        raise ValueError('Retry pending requires explicit resume.')
    saved = _checkpoint_state(checkpoint_path, document, resume=resume) if checkpoint_path else {}
    rows = []
    for case in cases:  # Serial by design; no nested/method-specific concurrency.
        previous = saved.get(case['id'])
        if previous is not None and not (retry_pending and _retryable(previous)):
            rows.append(previous)
            continue
        if checkpoint_path:
            _append(checkpoint_path, {'type': 'started', 'id': case['id']})
        result = evaluate_answer(case['query'], case['response'], case['target'], cache_dir=cache_dir)
        row = {'id': case['id'], **result}
        if checkpoint_path:
            _append(checkpoint_path, {'type': 'case', 'input_sha256': _digest(case),
                                     'config_sha256': _digest(identity()), 'row': row})
        rows.append(row)
    pending = sum(r['status'] != 'completed' for r in rows)
    correct = sum(r['correct'] is True for r in rows)
    return {'status': 'pending' if pending else 'completed', 'n_expected': len(cases),
            'n_completed': len(cases) - pending, 'n_pending': pending,
            'n_no_answer': sum((r.get('extraction') or {}).get('status') == 'no_answer' for r in rows),
            'pending_ids': [r['id'] for r in rows if r['status'] != 'completed'],
            'retryable_pending_ids': [r['id'] for r in rows if _retryable(r)],
            'n_correct': correct if not pending else None,
            'acc': 100.0 * correct / len(cases) if not pending else None,
            'evaluation_config': identity(), 'rows': rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_json', required=True, type=Path)
    parser.add_argument('--output_json', type=Path)
    parser.add_argument('--cache_dir', type=Path)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--checkpoint', type=Path, help='Append-only checkpoint in an existing run directory.')
    parser.add_argument('--resume', action='store_true', help='Reuse matching completed/pending records; run unstarted cases only.')
    parser.add_argument('--retry_pending', action='store_true', help='With --resume, also retry recorded transient technical failures.')
    args = parser.parse_args(argv)
    raw = args.input_json.read_bytes()
    document = json.loads(raw.decode('utf-8-sig'))
    cases = validate(document)
    if not args.run:
        print(json.dumps({'status': 'dry_run', 'n_expected': len(cases), 'network_calls': 0,
                          'evaluation_config': identity()}, ensure_ascii=False, indent=2))
        return 0
    if args.output_json is None or args.cache_dir is None:
        parser.error('--run requires a NEW --output_json and a run-specific --cache_dir')
    if args.output_json.exists():
        raise FileExistsError('Refusing to overwrite any existing result.')
    if args.output_json.resolve() == args.input_json.resolve():
        raise ValueError('Never overwrite the input.')
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or args.cache_dir / 'evaluation_checkpoint.jsonl'
    protected = {args.input_json.resolve(), args.output_json.resolve()}
    if checkpoint.resolve() in protected:
        raise ValueError('Checkpoint must not be the input or output file.')
    output = evaluate_batch(document, cache_dir=args.cache_dir, checkpoint_path=checkpoint,
                            resume=args.resume, retry_pending=args.retry_pending)
    output['source_sha256'] = hashlib.sha256(raw).hexdigest()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open('x', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in output.items() if k not in ('rows', 'evaluation_config')}, ensure_ascii=False))
    return 0 if output['status'] == 'completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())

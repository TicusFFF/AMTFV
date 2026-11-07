"""Run the Flash AMTFV verifier on one dataset and its existing initial answers."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parent
METHOD = ROOT / "S_SC_S2-3_Backward"
NATIVE = METHOD / "repro/scripts/math_hard_test_parallel_python_tool.py"
PROFILE = ROOT / "0-4_API/0_curr_API"
REGISTRY = ROOT / "0-4_API/2_model_capacity/model_capacities.json"
ALIAS = "dso_DSv41Flash"


def _serve_tool(work):
    from gevent import sleep
    from gevent.pywsgi import WSGIServer

    os.environ["CODE_INTERPRETER_WORK_DIR"] = str(work)
    (work / "upload_file").mkdir()
    sys.path.insert(0, str(METHOD / "repro/scripts"))
    import pack_api_windows as tool

    server = WSGIServer(("127.0.0.1", 0), tool.app, log=None)
    server.start()
    ready = work / "port"
    pending = ready.with_suffix(".tmp")
    pending.write_text(str(server.server_port), encoding="ascii")
    pending.replace(ready)
    try:
        while work.exists() and not (work / "stop").exists():
            sleep(0.2)
    finally:
        server.stop(timeout=5)
        tool.ci.stop_kernel(os.getpid())


@contextlib.contextmanager
def _local_tool(work):
    # Generated Python runs in this separate process without the model API keys.
    allowed = {
        "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "TMPDIR",
        "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "LANG",
        "LC_ALL", "CONDA_PREFIX", "VIRTUAL_ENV", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",
    }
    env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", MPLCONFIGDIR=str(work / "matplotlib"))
    proc = subprocess.Popen(
        [sys.executable, "-B", str(Path(__file__).resolve()), "_tool", str(work)],
        env=env, cwd=work,
    )
    try:
        deadline = time.monotonic() + 60
        ready = work / "port"
        while not ready.exists():
            if proc.poll() is not None:
                raise RuntimeError(f"Python tool exited with status {proc.returncode}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Python tool did not start within 60 seconds.")
            time.sleep(0.1)
        yield (int(ready.read_text(encoding="ascii")),)
    finally:
        (work / "stop").touch()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=10)


def _routed_registry(work, endpoint):
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    entry = registry["profiles"]["deepseek_official"][ALIAS]
    if endpoint == entry["identity"]["base_url"]:
        return REGISTRY
    # The local test uses the same capacities and tokenizer as generation.
    entry["identity"]["base_url"] = endpoint
    counter = entry["token_counting"]
    for field in ("tokenizer_name_or_path", "encoding_module_path"):
        counter[field] = str((REGISTRY.parent / counter[field]).resolve())
    counter["paths_relative_to_registry"] = False
    destination = work / "registry.json"
    destination.write_text(json.dumps(registry), encoding="utf-8")
    return destination


def run_dataset(*, dataset_key, data_path, initial_response_file, output_path,
                workers=1, start=0, end=None, python_tool_ports=()):
    """Save the native JSON for the selected half-open interval of questions."""
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("amtfv_generation_profile", PROFILE / "api_profiles.py")
    profile = module_from_spec(spec)
    spec.loader.exec_module(profile)
    config = profile.get_api_config()
    data = Path(data_path).resolve(strict=True)
    initials = Path(initial_response_file).resolve(strict=True)
    output = Path(output_path).resolve()
    ports = tuple(map(int, python_tool_ports))
    if (workers < 1 or start < 0 or (end is not None and end <= start)
            or any(port < 1 or port > 65535 for port in ports)):
        raise ValueError("Invalid worker count, Python tool port, or question interval.")
    if not dataset_key or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in dataset_key):
        raise ValueError("Use a dataset key containing letters, digits, underscores, or hyphens.")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(output.name + ".lock")
    with lock.open("x", encoding="ascii") as stream:
        stream.write(str(os.getpid()))
    try:
        metadata = output.with_suffix(".models.jsonl")
        if output.exists() or metadata.exists() or output.with_suffix(output.suffix + ".tmp").exists():
            raise FileExistsError(f"Choose a new output path: {output}")
        with tempfile.TemporaryDirectory(prefix="amtfv-") as temporary:
            work = Path(temporary)
            env = os.environ.copy()
            env.update(PYTHONPATH=str(PROFILE), PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
                       AMTFV_RESPONSE_METADATA=str(metadata),
                       SPQ_PYTHON_TOOL_LOCK_DIR=str(work / "locks"),
                       SPQ_COMPRESSION_CAPACITIES_PATH=str(_routed_registry(work, config["base_url"])))
            service = contextlib.nullcontext(ports) if ports else _local_tool(work)
            with service as actual_ports:
                args = [
                    sys.executable, "-B", str(NATIVE),
                    "--repo_root", str(METHOD / "upstream/progco"),
                    "--data_path", str(data), "--dataset_key", dataset_key,
                    "--initial_response_file", str(initials), "--output_path", str(output),
                    "--model", ALIAS, "--max_cur_turn", "3", "--temperature", "0",
                    "--workers", str(workers), "--start", str(start),
                    "--end", str(end if end is not None else -1),
                    "--python_tool_ports", ",".join(map(str, actual_ports)),
                    "--python_tool_execution_timeout_seconds", "120",
                    "--python_tool_code_soft_warning_calls", "100",
                    "--python_tool_code_hard_stop_calls", "120",
                    "--llm_timeout_seconds", "960", "--llm_max_retries", "10",
                    "--llm_stop_max_retries", "20", "--agent2_execute_retries", "10",
                    "--agent7_rewrite_retries", "10", "--agent_retry_wait_seconds", "10",
                ]
                subprocess.run(args, cwd=ROOT, env=env, check=True)
    finally:
        lock.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--initial-responses", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, help="Exclusive end index")
    parser.add_argument("--python-tool-ports", help="Comma-separated existing localhost tool ports")
    args = parser.parse_args()
    run_dataset(dataset_key=args.dataset_key, data_path=args.data,
                initial_response_file=args.initial_responses, output_path=args.output,
                workers=args.workers, start=args.start, end=args.end,
                python_tool_ports=args.python_tool_ports.split(",") if args.python_tool_ports else ())


if __name__ == "__main__":
    if sys.argv[1:2] == ["_tool"]:
        _serve_tool(Path(sys.argv[2]).resolve())
    else:
        main()

"""Run the full dataset entry with a scripted localhost model and real Python tool."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


ROOT = Path(__file__).resolve().parents[3]
MODEL = "deepseek-flash"
SENTINEL = "PRIVATE_REFERENCE_MUST_NOT_ENTER_GENERATION"
TOOL_FLOW = "<tool_flow>Compute the sum 1 + 1.</tool_flow>"
SCRIPT = [
    (False, "Compute 1 + 1 and compare it to the candidate."),
    (True, TOOL_FLOW),
    (False, "def execute():\n    return 1 + 1\nexecute()"),
    (True, TOOL_FLOW),
    (True, "<result>False</result>"),
    (False, "The candidate gives 3, but 1 + 1 equals 2."),
    (True, "[Reflection] Check the sum.\n[New Solution]\nAdding one to one gives two. The answer is 2."),
    (True, "[Reflection] The test checks addition.\n[New Verification Flow]\nRecompute the sum."),
    (False, "2"),
    (False, "[Key Points to Note When Solving the Problem]\n1. Check addition."),
    (False, "Adding one to one gives two. The answer is 2."),
    (False, "2"),
    (True, "<result>True</result>"),
]


class LocalModel(BaseHTTPRequestHandler):
    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        request = json.loads(raw)
        if (self.path != "/v1/chat/completions" or request["model"] != MODEL
                or request.get("thinking") != {"type": "disabled"} or SENTINEL.encode() in raw):
            self.send_error(400)
            return
        streaming = bool(request.get("stream"))
        index = len(self.server.requests)
        if index >= len(SCRIPT) or SCRIPT[index][0] != streaming:
            self.send_error(400, "Unexpected model call in local fixture")
            return
        self.server.requests.append(request)
        reply = SCRIPT[index][1]
        response = {"id": "local-test", "created": 0, "model": MODEL}
        if streaming:
            response.update(object="chat.completion.chunk", choices=[
                {"index": 0, "delta": {"content": reply}, "finish_reason": None}
            ])
            content = ("data: " + json.dumps(response) + "\n\ndata: [DONE]\n\n").encode()
        else:
            response.update(object="chat.completion", choices=[
                {"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}
            ])
            content = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, *args):
        pass


def main():
    with tempfile.TemporaryDirectory(prefix="amtfv-smoke-") as temporary:
        work = Path(temporary)
        (work / "data.json").write_text(json.dumps([
            {"id": "test_0", "query": "Compute 1 + 1.", "target": SENTINEL}
        ]), encoding="utf-8")
        (work / "initial.jsonl").write_text(json.dumps(
            {"id": "test_0", "response": "Adding one to one gives three.", "pred": "3"}
        ) + "\n", encoding="utf-8")
        with ThreadingHTTPServer(("127.0.0.1", 0), LocalModel) as server:
            server.requests = []
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            env = os.environ.copy()
            env.update(DEEPSEEK_API_KEY="local-placeholder", AMTFV_LOCAL_TEST_URL=f"http://127.0.0.1:{server.server_port}/v1",
                       NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
                       PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")
            output = work / "result.json"
            try:
                subprocess.run([sys.executable, "-B", str(ROOT / "amt_runner.py"), "--dataset-key", "test",
                                "--data", str(work / "data.json"), "--initial-responses", str(work / "initial.jsonl"),
                                "--output", str(output)], env=env, check=True)
            finally:
                server.shutdown()
                thread.join(timeout=5)
        record = json.loads(output.read_text(encoding="utf-8"))["logs"][0]
        assert record["id"] == "test_0" and record["final_result"] == "2", record
        events = record["turns"][0]["python_tool_events"]
        assert any(e.get("status") == "ok" and not e.get("cache_hit") for e in events), events
        assert any(e.get("cache_hit") for e in events), events
        assert len(record["turns"]) == 2 and len(server.requests) == len(SCRIPT)
        metadata = [json.loads(line) for line in output.with_suffix(".models.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(metadata) == len(SCRIPT) and all(row["endpoint"] == "local-test" and
            row["requested_model"] == row["returned_model"] == MODEL for row in metadata)
        print("Local workflow passed: initial answer, real Python, cached repeat, correction, verified final answer.")


if __name__ == "__main__":
    main()

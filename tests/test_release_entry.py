"""Exercise the same inference-to-scoring entry on demo and benchmark inputs."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ENTRY = next((ROOT / "0-3_all_model_test").rglob("run_amtfv.py"))
sys.path.insert(0, str(ROOT))
import amt_runner


class ReleaseEntryTest(unittest.TestCase):
    def test_shared_dataset_entry(self):
        spec = importlib.util.spec_from_file_location("flash_entry", ENTRY)
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        for dataset in ("demo", "hmmt_feb_2025"):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as work:
                base = Path(work)
                out = base / "run"
                data = json.loads((entry.DATA_ROOT / entry.DATASETS[dataset]["file"]).read_text(encoding="utf-8"))
                row = data[0]
                query = row["q"]
                sample_id = f"{dataset}_0"
                fixture = {"id": sample_id, "query": query, "response": "fixture", "pred": "0"}
                (base / f"deepseek-flash_{dataset}_first_response.jsonl").write_text(json.dumps(fixture)+"\n", encoding="utf-8")

                def verifier_fixture(*, output_path, dataset_key, **kwargs):
                    self.assertEqual(dataset_key, dataset)
                    output_path.write_text(json.dumps({"logs": [
                        {"id": sample_id, "query": query, "final_response": f"Final answer: {row['answer']}"}
                    ]}), encoding="utf-8")

                argv = [str(ENTRY), "--dataset", dataset, "--limit", "1", "--initial-dir", str(base),
                        "--output", str(out), "--run"]
                env = {"DEEPSEEK_API_KEY": "offline", "HTTPS_PROXY": "http://127.0.0.1:1", "NO_PROXY": ""}
                with patch.dict(os.environ, env), patch.object(sys, "argv", argv):
                    with patch.object(amt_runner, "run_dataset", side_effect=verifier_fixture) as run:
                        entry.main()
                        run.assert_called_once()
                scores = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
                self.assertEqual([r["id"] for r in scores["rows"]], [sample_id])
                self.assertEqual((scores["acc"], scores["n_completed"]), (100.0, 1))


if __name__ == "__main__":
    unittest.main()

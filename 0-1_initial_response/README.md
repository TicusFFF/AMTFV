# Shared initial responses

From the repository root, set `DEEPSEEK_API_KEY` and run:

```sh
python 0-1_initial_response/generate_initial_responses.py --dataset demo --output-dir runs/initial
```

The generator sends each original question to the official DeepSeek `deepseek-flash` endpoint at temperature 0 with thinking disabled. A second call extracts the candidate from the response using the [initial extraction prompt](initial_extraction_prompt.txt). If the first response hits the provider’s length limit, its candidate remains empty. The request leaves the output limit at the provider’s default. The `--workers` option sets the number of questions generated concurrently.

The correction runner reads `runs/initial/deepseek-flash_demo_first_response.jsonl`. Each row contains the question ID, the question, the full response and the candidate. The adjacent `.version.json` records request and returned model identities, generation times, available system fingerprints and content hashes. Keep both files together. Repeating the same command resumes the batch and reuses completed answers. `deepseek-flash` is a rolling alias; the identifiers recorded here describe this run.

After generation, `prepare_inputs.py` joins each full response to its reference by question ID and checks the question text. The [shared evaluator](../0-2-2_unified_evaluation/README.md) applies the same V5 extraction and V2 grading used for AMTFV's final responses. The demo score is written to `runs/initial/demo.initial-eval.json`. A completed report for identical inputs is reused. To evaluate an already generated batch separately, add `--evaluate-only` to the command above. For another dataset, replace `demo` with its registered key.

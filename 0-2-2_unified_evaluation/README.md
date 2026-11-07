# Final-answer evaluation

This directory contains the shared V5 answer extractor and V2 equivalence judge. The extractor receives the question and the selected final response. The judge compares the extracted answer with the reference. Both stages retain the original prompts, rules and retry limits.

Run the following commands from the repository root. The first command builds an evaluation input from native AMTFV output. Repeat `--dataset KEY DATASET_JSON NATIVE_OUTPUT_JSON` to combine several datasets. The generated records determine the subset. Their IDs and questions must agree with the original dataset, even when evaluating a single question.

```console
python 0-2-2_unified_evaluation/prepare_inputs.py --dataset demo 0-0_datasets/hard_test_raw/demo/test_top2_filt2.json path/to/model_outputs/demo.json --output run/eval_input.json
python 0-2-2_unified_evaluation/evaluate.py --input_json run/eval_input.json
```

The second command checks the input and makes no network requests. To score the responses, set `DEEPSEEK_API_KEY` for the official DeepSeek endpoint and provide a new result path and a run-specific cache directory. Extraction and judging use `deepseek-flash` with thinking disabled and temperature zero when a model call is needed. This model identifier is a service alias.

```console
python 0-2-2_unified_evaluation/evaluate.py --input_json run/eval_input.json --output_json run/eval_result.json --cache_dir run/eval_cache --run
```

`acc` is the percentage of correct responses among all expected IDs. An empty answer counts as incorrect. If any score is pending, the report gives its ID and leaves `acc` unset. A resumed run can reuse its checkpoint with `--resume` and a new result path. Use `--retry_pending` with `--resume` to retry recorded transient failures.

To run the offline checks with Python's standard library:

```console
python 0-2-2_unified_evaluation/test_offline.py
```

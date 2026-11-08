# AMTFV: Agentic Mathematical Tool-Flow Verification for LLM Self-Correction

English | [简体中文](README_zh-CN.md)

AMTFV verifies and corrects mathematical answers produced by large language models. It separates mathematical reasoning from tool execution through Mathematical Tool Flow (MTF). During verification, the model submits computation requests, the toolbox executes them, and the model uses the returned results to continue verification or revise the answer.

## Framework

[![AMTFV framework](assets/framework.png)](assets/framework.png)

Given a problem and an initial response, the module on the left verifies the candidate answer. When computation is needed, the MTF interface in the middle passes the request to the toolbox on the right. The toolbox selects an appropriate tool, performs the computation, and returns the result. The left module then accepts or revises the answer, or adjusts the verification steps.

## Main Results

[![Table 1 from the paper: Accuracy of each method on five mathematical reasoning datasets.](assets/main-results.png)](assets/main-results.png)

## Installation and Configuration

Run the following command from the repository root:

```bash
python -m pip install -r requirements.txt
```

The example uses the official DeepSeek API. Set `DEEPSEEK_API_KEY`. In PowerShell, use `$env:DEEPSEEK_API_KEY = "<your-key>"`; in Bash, use `export DEEPSEEK_API_KEY="<your-key>"`.

## Quick Start

The included `demo` contains two mathematics problems. The following two steps call the model API.

### 1. Generate Shared Initial Responses

```bash
python 0-1_initial_response/generate_initial_responses.py --dataset demo --output-dir runs/initial
```

Initial responses are saved locally to `runs/initial/deepseek-flash_demo_first_response.jsonl`, and their scores are saved to `demo.initial-eval.json` in the same directory. When comparing methods, you can share this batch of initial responses. In step 2, set `--initial-dir` to the directory containing the initial responses and model version records.

### 2. Run AMTFV and Evaluate

```bash
python 0-3_all_model_test/run_plan_S_SC/S_SC_S2-3_Backward/run_amtfv.py --dataset demo --initial-dir runs/initial --run
```

The program reads the initial responses, starts the local Python tool, and performs verification and correction. By default, each run is saved to `runs/<UTC-timestamp>/`. The file `model_outputs/demo.json` contains per-problem reasoning and final responses, and `evaluation.json` contains the results from the shared evaluator. Reference answers are used only for evaluation. The Python tool executes model-generated code on your machine, so use a dedicated environment with restricted permissions.

To run HMMT, first prepare the dataset as described in `0-0_datasets/README.md`, then replace `--dataset demo` with `--dataset hmmt_feb_2025` in both commands.

Parts of the verification and correction code are adapted from ProgCo. See [attribution](THIRD_PARTY.md).

## Citation

[arXiv:2607.29549](https://arxiv.org/abs/2607.29549)

```bibtex
@misc{zou2026amtfv,
  title = {{AMTFV}: Agentic Mathematical Tool-Flow Verification for {LLM} Self-Correction},
  author = {Zou, Rui and Zhu, Yutao and Wei, Mengqi and Wen, Ji-Rong},
  year = {2026},
  eprint = {2607.29549},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  url = {https://arxiv.org/abs/2607.29549}
}
```

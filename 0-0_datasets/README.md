# Datasets

This repository includes a two-problem `demo`. From the repository root, run `python 0-0_datasets/prepare.py --dataset demo` to verify it.

## Paper benchmarks

To run a paper benchmark, download the source data with the preparation script. For example:

```sh
python -m pip install datasets
python 0-0_datasets/prepare.py --dataset hmmt_feb_2025
```

Replace `hmmt_feb_2025` with a dataset key below. If you already have the source rows in a local JSON array, use `python 0-0_datasets/prepare.py --dataset hmmt_feb_2025 --input source_rows.json`.

| Dataset key | Problems | Source |
|---|---:|---|
| `aime_2024` | 30 | [HuggingFaceH4/aime_2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024), Mathematical Association of America |
| `aime_2025` | 30 | [yentinglin/aime_2025](https://huggingface.co/datasets/yentinglin/aime_2025), Mathematical Association of America |
| `brumo_2025` | 30 | [MathArena/brumo_2025](https://huggingface.co/datasets/MathArena/brumo_2025), Brown University Math Olympiad |
| `hmmt_feb_2025` | 30 | [MathArena/hmmt_feb_2025](https://huggingface.co/datasets/MathArena/hmmt_feb_2025), Harvard–MIT Mathematics Tournament |
| `amo_bench` | 50 | [Meituan LongCat, AMO-Bench](https://huggingface.co/datasets/meituan-longcat/AMO-Bench), Shengnan An and colleagues (MIT) |

Each source's terms apply when obtaining its data. BRUMO and HMMT use the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. The AMO-Bench license is included in `AMO-Bench-LICENSE.txt`.

The script matches the questions and reference answers to the hashes in `dataset_manifest.json`, preserves the order used in the paper, and saves the prepared input locally under `hard_test_raw/`. The initial-response and experiment commands accept the same dataset key used here.

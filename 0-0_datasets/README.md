# Benchmark inputs

The five benchmark datasets contain 170 problems. The separate `demo` has two original problems for running the pipeline. The distributed JSON rows use the exact question (`q`) and reference (`answer`, also exposed as `A_original`). The runner assigns zero-based IDs within each dataset. Solutions and other unused source columns have been omitted. The manifest records the SHA-256 of the exact UTF-8 question and reference for each ID.

| Dataset | Problems | Source and conditions |
|---|---:|---|
| AIME 2024 | 30 | [HuggingFaceH4/aime_2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024), original problems from the Mathematical Association of America. Prepare locally. |
| AIME 2025 | 30 | [yentinglin/aime_2025](https://huggingface.co/datasets/yentinglin/aime_2025), original problems from the Mathematical Association of America. Prepare locally. |
| BRUMO 2025 | 30 | [MathArena/brumo_2025](https://huggingface.co/datasets/MathArena/brumo_2025), Brown University Math Olympiad. Included under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). |
| HMMT February 2025 | 30 | [MathArena/hmmt_feb_2025](https://huggingface.co/datasets/MathArena/hmmt_feb_2025), Harvard–MIT Mathematics Tournament. Included under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). |
| AMO-Bench | 50 | [Meituan LongCat, AMO-Bench](https://huggingface.co/datasets/meituan-longcat/AMO-Bench), by Shengnan An et al. Included under the [MIT license](AMO-Bench-LICENSE.txt). |

The demo is included. From the repository root, verify it with `python 0-0_datasets/prepare.py --dataset demo`. To prepare the two AIME datasets, run:

```sh
python -m pip install datasets
python 0-0_datasets/prepare.py --dataset aime_2024
python 0-0_datasets/prepare.py --dataset aime_2025
```

You may also pass `--input authorized_source_rows.json` to prepare a dataset from a local JSON array. The preparation script checks the complete 30- or 50-item set, orders it by the original IDs, and stops if either a question or its reference differs. The AIME source cards listed above do not declare redistribution terms for the underlying problem text. Obtain those inputs under their source terms; this repository supplies the item IDs and hashes. The MathArena data retain the attribution and noncommercial share-alike terms above, including after conversion to the AMTFV input schema.

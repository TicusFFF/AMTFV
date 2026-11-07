# DeepSeek-V4.1-Flash tokenizer resources

Official source: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash

Pinned revision: `dba1be0a40aa45a94ad051997016db3960a90277`.

Downloaded on 2026-09-12. The upstream files below are preserved without edits.
The complete upstream MIT license is retained in `LICENSE` (Copyright (c) 2023 DeepSeek).
This directory contains tokenizer/encoding resources only, not model weights.

| Upstream path | SHA256 |
| --- | --- |
| `tokenizer.json` | `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b` |
| `tokenizer_config.json` | `6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547` |
| `encoding/encoding.py` | `502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1` |
| `encoding/README.md` | `a2f0fc3baea318c9cfbceca68cbfe50d37cf7da6605ace887f33148bcff7e3ae` |
| `LICENSE` | `f2c6c602815669d292889e5be8c802f2ed950653b77999b1584e8e6aed25d040` |

AMTFV uses this fixed tokenizer and chat template to count the context for
compression. Generation requests use the rolling `deepseek-flash` alias at
`https://api.deepseek.com` with thinking disabled. The tokenizer revision
identifies the local files. Each run saves the provider's returned model field
and available system fingerprint alongside its predictions.

On September 13, 2026, the local counts matched the official API's prompt usage
on seven text-chat cases, including multi-turn histories and assistant endings.
We also replayed the inputs of three successful October 22 Chat Completions
requests against the local tokenizer. Local and provider counts were 28/28,
23/23, and 23/23 tokens, respectively. All three had a single user message and
returned the model name `deepseek-flash`. This later check covers those three
inputs. Counts for long or multi-turn October requests remain unverified.


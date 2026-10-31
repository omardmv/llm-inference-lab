# LLM Inference Lab

Hands-on inference engineering on one Apple Silicon Mac (M4 Pro, 24 GB): measure how a
local LLM server really behaves, find the bottleneck, change one setting, measure again.

It contains:

- **`bench/`** – a Python CLI that load-tests Ollama or llama-server and records
  time to first token, decode speed, throughput, GPU temperature/clock/power and
  tokens per joule, with a quiet-machine check and repeated rounds.
- **`app/`** – *Inference Lab*, a native macOS (SwiftUI + Swift Charts) dashboard for the
  same measurements: live GPU/memory, model internals, tokenizer, single requests,
  load tests and a run history to compare results.
- **`tickets/`** – the learning tickets, written like work tickets.
- **`research/`** – raw results (CSV + JSON metadata + charts) of every run.

## What I found

All numbers: `qwen2.5:1.5b` (Q4_K_M), answers of up to 400 tokens.
SLO = p95 time to first token ≤ 2 s and ≥ 20 tok/s per user.

| Finding | Evidence |
|---|---|
| **Ollama on this Mac serves one request at a time.** Total throughput stays at ~145 tok/s from 1 to 32 users; the wait for the first token grows by (users − 1) × ~2.5 s. | Only **1 user** meets the SLO. |
| **Continuous batching fixes it.** llama-server with 8 slots serves **8 users within the SLO** at ~520 tok/s total. | 8× the users, 3.5× the throughput, **~65% less energy per token** (8.8 → 24.7 tok/J). |
| **Throughput tops out between 8 and 16 slots** (521 → 529 tok/s): decode turns from memory-bound to compute-bound. | Per-step time 6.8 ms (1) → 12.9 ms (8) → 24.5 ms (16). |
| **Batching isn't free at batch 2 or 9.** Step time jumps +56% from 1→2 tokens and +44% from 8→9, at exact batch sizes, with or without flash attention. | llama.cpp's Metal backend switches matrix kernels (matrix×vector, small-batch ≤ 8, matrix×matrix). With Q8_0 weights the 1→2 jump is only +16%. |
| **Heat is not the problem here.** The GPU holds its full 1,578 MHz clock at ~95 °C; fans do the work. Earlier slowdowns came from other apps competing for memory bandwidth. | GPU clock/temperature logged twice a second with macmon. |
| **The server loses ~30–40% of the engine's speed at 8–16 slots.** Not the client, not sampling or streaming; part of it is the batch not being full (6.6 of 8 slots busy on average). | `llama-batched-bench` vs `bench sweep`. |

![Ollama under load: the queue](research/bench/sweep_qwen2.5-1.5b_20261030-230512.png)

## Use it

Requirements: macOS on Apple Silicon, [Ollama](https://ollama.com), llama.cpp
(`brew install llama.cpp`), Python with `numpy pandas matplotlib`, and optionally
[macmon](https://github.com/vladkens/macmon) for temperature/clock/power.

```bash
python -m bench hw                      # the machine and what it is doing now
python -m bench model quality           # layers, heads, KV cache size, memory once loaded
python -m bench run quality --cold      # one request: load / prefill / decode
python -m bench check                   # is the machine quiet enough to benchmark?
python -m bench sweep quality --repeat 2                  # 1–16 users against Ollama
python -m bench slots quality --np 1,2,4,8,16            # llama-server at each slot count
```

The app (needs only the Command Line Tools):

```bash
app/run.sh
```

## Learning path

| Ticket | Topic |
|---|---|
| [00](tickets/00-latency-predictor.md) | Predict request latency with a NumPy-only model (gradients by hand) |
| [01](tickets/01-see-inside.md) | See inside the server: memory, tokens, prefill vs decode, the roofline |
| [02](tickets/02-serve-more-users.md) | Serve 4× more users on the same Mac: batching, SLOs, quantization |

Next: building the engine myself (KV cache, continuous batching) with
[tiny-llm](https://github.com/skyzh/tiny-llm).

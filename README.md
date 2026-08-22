# Running 30B-class MoE coding models on an 8GB GPU

**English** | [한국어](#8gb-vram에서-30b급-moe-코딩-모델-구동하기)

Measured notes and the running configuration for a 30B-class coding LLM at usable speed on a single 8GB GPU. Every number below was measured on the machine described here — including the approaches that turned out to be slower. The work started on Qwen3-Coder-30B-A3B; the current configuration runs Qwen3.6-35B-A3B, and both are documented.

**Result: a 35B-parameter model with a 128K context on a GPU that holds neither** — 36.4 tok/s generation and 839 t/s prompt processing at a real 22K-token Cline working context, up from 21.1 tok/s on the model this document started with.

## Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 2070 SUPER (8GB VRAM, ~7.6GB usable) |
| CPU | Intel Core i5-12400F (6 cores / 12 threads) |
| RAM | 32GB DDR5-5600, **single channel** (one DIMM populated; 29.1 GB/s measured) |
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0 |
| CUDA | Driver 580.173.02 (CUDA 13.0), Toolkit 12.6 |
| Runtime | [llama.cpp](https://github.com/ggml-org/llama.cpp), CUDA backend built from source (`-DCMAKE_CUDA_ARCHITECTURES=75`) |

## Why a MoE model

Dense models were measured first, on the same prompt and hardware:

| Model | Setup | Generation |
|---|---|---:|
| Qwen2.5-Coder-32B Q4_K_M | dense, 20 layers on GPU + speculative decoding | 6.1 tok/s |
| Qwen2.5-Coder-14B Q4_K_M | dense, 41 layers on GPU | 15–18 tok/s |
| Qwen2.5-Coder-7B Q5_K_M | dense, fully on GPU | 58–64 tok/s |
| **Qwen3-Coder-30B-A3B UD-Q3_K_XL** | **MoE, partial expert offload** | **43.2 tok/s** |

The 32B dense model was unusable at 6 tok/s; the 7B was fast but produced buggier code. Qwen3-Coder-30B-A3B has 30.5B total parameters but activates only ~3B per token, so it carries 30B-class knowledge at roughly 3B-class compute.

llama.cpp's `--n-cpu-moe` then keeps only part of the expert FFN weights in system RAM while attention and the rest stay on the GPU. That is what makes a 13.81GB model run on 8GB of VRAM.

## Current configuration

```bash
llama-server \
  -m Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf \
  -ngl 99 -ncmoe 33 -fa on -t 6 -lm none -np 2 -kvu -ub 2048 \
  -c 131072 -ctk q8_0 -ctv q8_0 -sps 0.5 -rea off \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --host 127.0.0.1 --port 8080
```

This is [`scripts/run-server.sh`](scripts/run-server.sh). It replaced the Qwen3-Coder-30B-A3B
configuration below after the [A/B](#model-landscape-august-2026) showed +72% generation, +16% prompt
processing and twice the context at the same VRAM. Everything this document establishes about `-ub`,
`-np`, `-lm`, `-t` and the KV cache carries over unchanged — only the model, `-ncmoe`, `-c` and the
sampling values differ.

| Flag | Why |
|---|---|
| `-ncmoe 33` | Keep 33 of 40 layers' expert weights on the CPU. The OOM floor at this context is 32 |
| `-c 131072` | Costs 2.4% against 65536 and puts Cline's condense threshold at 114K, so prompt-cache rebuilds stop happening. Affordable because only 10 of 40 layers run full attention |
| `-sps 0.5` | Slot-prompt similarity floor. The default of 0.10 lets a barely-related prompt claim a slot holding a 60K cache and destroy it |
| `-rea off` | Disable thinking. Thought tokens are latency on every single tool call in an agent loop |
| sampling flags | The model's own recommended values, from its GGUF metadata. **Not** Qwen3-Coder's 0.7 / 0.8 / 1.05 |

### Previous configuration (Qwen3-Coder-30B-A3B)

Kept because most of the analysis below was measured on it, and it is still the right choice if you
want a coder-tuned model at a smaller download.

```bash
llama-server \
  -m Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf \
  -ngl 99 -ncmoe 40 -fa on -t 6 -lm none -np 2 -kvu -ub 2048 \
  -c 65536 -ctk q8_0 -ctv q8_0 \
  --temp 0.7 --top-p 0.8 --top-k 20 --repeat-penalty 1.05 --min-p 0.0 \
  --host 127.0.0.1 --port 8080
```

| Flag | Why |
|---|---|
| `-ncmoe 40` | Keep 40 of 48 layers' expert weights on the CPU; everything else on the GPU |
| `-ub 2048` | Physical batch size. **+77% prompt processing** (407 → 722 t/s) at no measurable cost to generation — the largest single win for an agent workload |
| `-np 2 -kvu` | Two slots, unified KV. The default of four slots evicts its own 20K+ prompt caches; `-np` alone silently turns unified KV *off* and halves per-slot context |
| `-c 65536` | Large enough that Cline rarely has to condense — each condense costs a 40–70 s prompt-cache rebuild |
| `-lm none` | Load fully into RAM instead of mmap — removes page-fault overhead on offloaded tensors (prompt processing +34%) |
| `-t 6` | Physical cores only. Using all 12 threads is **11% slower** (36.1 → 32.4 tok/s) |
| `-fa on` | Flash Attention — free speed, no quality cost |
| `-ctk/-ctv q8_0` | 8-bit KV cache. Do **not** drop V to `q4_0` on this GPU (see below) |
| sampling flags | Qwen3-Coder's recommended values. llama-server defaults (temp 0.80 / top_p 0.95 / top_k 40) are too loose for agent coding and invite wrong APIs and imports |

> An earlier revision of this document recommended `-c 36864 -ncmoe 32` for a Cline-only setup,
> because it benchmarks 9% faster (45.3 vs 41.3 tok/s at `tg128`). Session logs showed that to be the
> wrong trade for an agent client — see [Tuning for an agent workload](#tuning-for-an-agent-workload).
> For short-context chat, where that 9% is real and nothing ever condenses, `-c 36864 -ncmoe 32`
> remains the faster choice.

Quantization went Q4_K_M (18.56GB) → IQ4_XS (16.38GB) → **UD-Q3_K_XL (13.81GB)**. Because generation is bandwidth-bound here, each shrink both frees VRAM (`ncmoe` 36 → 34 → 32) and cuts per-token RAM traffic, so speed rose at every step.

## The bottleneck is RAM bandwidth, not the GPU

Three independent measurements agree:

1. **Thread scaling saturates early.** 2→3 cores gives +30%, but 5→6 cores gives only **+1.5%** — spare CPU compute that speed does not follow.
2. **Cost per offloaded expert layer is flat at 0.448 ms** across `ncmoe` 34/40/48. At `ncmoe=32`, roughly 60% of each token's time is spent reading expert weights from RAM.
3. **Measured sequential read bandwidth is 29.1 GB/s** — about 65% of the 44.8 GB/s theoretical peak for single-channel DDR5-5600.

The practical consequence: this machine runs **single channel**, with one DIMM in `Controller0-DIMM1` and `Controller1` empty. Adding a matching module should roughly halve the CPU-side time. Adding CPU cores would not help at all.

## What did not work

| Attempt | Result |
|---|---|
| Speculative decoding with a small draft model | Effective on the 32B dense model, but **halved throughput to 17.3 tok/s** on the MoE — the draft model competes for the same CPU cores already saturated by expert offload |
| `q4_0` V cache | Buys +33% context at the same `ncmoe`, but runs **13.5% slower** (44.8 → 38.8 tok/s). Turing (sm_75) has no optimized flash-attention path for `q4_0` V, so it pays full dequantization cost |
| Per-tensor `-ot` offload | ~3% faster at small context, but OOMs at the production context — VRAM is already saturated |
| Reducing `--parallel` *to free VRAM* | No VRAM freed — the KV pool is sized by `-c`, not by slot count. Worth doing anyway, for an unrelated reason: see [Tuning for an agent workload](#tuning-for-an-agent-workload) |
| `--cache-reuse 256` | **Exactly zero effect** — cache-hit counts were byte-identical with and without it in an A/B. Its reuse scan only advances the prompt-side pointer on a match, so the *inserted* summary that Cline's condense produces stops it before it can reach the unchanged recent messages. It absorbs *deletions* from the cache, not insertions (see the Korean section for the measurement) |

## Tuning for an agent workload

`tg128` is not what an agent session feels like. Aggregated from 36 minutes of real Cline traffic
(server log, `-c 65536 -ncmoe 38 -ub 512`):

| | Time | Tokens | Rate |
|---|---:|---:|---:|
| Prompt processing | 172.1 s | 56,032 | 326 t/s |
| Generation | 192.9 s | 2,905 | **15.1 tok/s** |

Two things follow. Generation at a 22K-token context runs 16–21 tok/s, not the 45 tok/s of `tg128` —
context fill dominates and `ncmoe` barely moves it. And **prompt processing is 47% of wall time**,
73% of which was two cold 22–25K prompts costing 56 s and 70 s. So the levers that matter are the ones
that avoid reprocessing, plus the one that makes unavoidable reprocessing fast.

### Batch size (`-ub`) — the largest single win

Expert weights are read from RAM once per ubatch and amortized across every token in it, so a larger
physical batch converts the bandwidth bottleneck into throughput. Measured on a fixed 21,970-token
prompt with `cache_prompt: false`, median of 3 runs:

| `-ub` / `-ncmoe` | Prompt processing | Generation @22K | VRAM |
|---|---:|---:|---:|
| 512 / 38 (previous) | 407 t/s | 21.02 tok/s | 7427 MiB |
| 1024 / 38 | 573 t/s | 21.36 | 7551 |
| 2048 / 38 | **OOM** — compute buffer | — | — |
| **2048 / 40 (adopted)** | **722 t/s** | **21.11** | **7334** |
| 2048 / 39 | 710 t/s | 20.92 | 7626 |
| 4096 / 42 | 703 t/s | 20.21 | 6832 |

`-ub 2048` does not fit at `ncmoe 38`, but the two expert layers it costs are free: at a 22K context
they are 1.9% of per-token time, below the run-to-run spread. At short context the same two layers are
resolvable and the 0.448 ms/layer model still holds — 38.4 tok/s measured at `ncmoe 40` against 38.9
predicted. Moving them off the GPU also more than pays for the larger compute buffer: VRAM went *down*,
7427 → 7334 MiB. `-ub 4096` is past the knee — no further gain, and generation starts to suffer.

### Before and after, in production

The same aggregation run against a real Cline session on the current configuration:

| | Before | After |
|---|---:|---:|
| Model / config | Qwen3-Coder-30B-A3B, `-ub 512 -c 65536` | Qwen3.6-35B-A3B, `-ub 2048 -np 2 -kvu -c 131072` |
| Generation | 15.1 tok/s | **34.3 tok/s** |
| Prompt processing | 172.1 s / 56,032 tok | **49.9 s / 22,667 tok** |
| Prompt share of wall time | **47%** | **6%** |
| Prefix-cache hits | 4 of 18 (22%) | **26 of 27 (96%)** |
| Largest single reprocess | 24,761 tok / **70.3 s** | 5,488 tok / **6.0 s** (cold start) |
| Deepest context reached | 28,312 | 44,721 |
| Output tokens per minute of server time | 477 | **1,925** |

No stall occurred in this window: the only reprocess was the initial cold prompt, and the other 26
turns hit the cache. Context reached 44,721 tokens without triggering a condense, which the original
`-c 36864` would have done at 29.5K.

**This window was too short to conclude stalls were eliminated.** Watching for several more hours
showed 40–80 s reprocesses still happening, from two causes that this 13-minute sample did not reach.
See the next section.

Generation held up across an 8× growth in context, falling only 29%:

```
 5.5K → 33.9      21K → 33.5      38K → 30.9
 6.8K → 39.1      27K → 32.4      43K → 29.9
  12K → 36.6      32K → 32.3      45K → 28.0
```

The previous model fell 45% over the same kind of range (38.4 tok/s short → 21.1 at 22K) and averaged
15.1 in practice. 28.0 tok/s at 44.7K is still 33% faster than what the old setup managed at 22K.

Two caveats. The prompt-processing rate reads *lower* (454 vs 839 t/s in the benchmark) because 26 of
27 calls were cache hits processing a few dozen incremental tokens each, where fixed overhead
dominates — the drop from 56,032 to 22,667 total tokens processed is the cache working, not a
regression. And these are two different real sessions, not a controlled experiment; the controlled
comparison is the fixed-prompt benchmark above. What transfers cleanly is the structural change: cache
hit rate and the share of wall time spent reprocessing.

### Slot count (`-np`)

With the default of 4 slots, 14 of 18 slot selections in the session log fell back to LRU: four separate
18–25K conversations were resident at once, 87K of demand against a 65K unified KV pool, evicting each
other. Cline runs one conversation plus small auxiliary calls, so two slots suffice — and with exactly
two, LRU sends the auxiliary call to the *other* slot and leaves the main prefix intact. Slot count
costs no VRAM either way.

> **`-np N` silently disables unified KV.** Its default is "enabled only when the slot count is auto",
> so `-np 2` alone turns `-c 65536` into 32768 per slot, which then breaks a client configured for a
> 61440 context window. Always pass `-kvu` with it.

### Slot-prompt similarity (`-sps`) — what still stalls

Over a longer observation the 40–80 s reprocesses came back. Two distinct causes, only one fixable:

**Condense, unavoidable.** With `contextWindow` at 126,976 Cline compacts at ~114K, exactly as
designed — measured at 117,052 → 58,588 tokens and 115,482 → 35,464. Each compaction replaces the head
of the conversation with a summary, so the prefix changes and the cache is void. The `--cache-reuse`
A/B above establishes this cannot be worked around.

**A near-unrelated prompt claiming a warm slot — fixable.** The two largest stalls of the day were
preceded by these slot selections:

```
f_sim_best = 0.103 (> 0.100 thold)  → 53,543 tokens reprocessed, 76.5 s
f_sim_best = 0.100 (> 0.100 thold)  → 55,876 tokens reprocessed, 81.9 s
```

Ten percent similarity. `--slot-prompt-similarity` defaults to **0.10**, so a prompt sharing nothing
but the system preamble still qualifies to take over a slot holding a 60K cache — and destroys it. Of
178 slot selections in one afternoon, 156 were healthy (≥0.9) and only 3 fell below 0.2, but those 3
carried the worst stalls. Raising the floor to `0.5` sends such requests to the other slot instead,
leaving the live cache intact for when its conversation returns. The reprocess cost of the odd request
is unchanged; what changes is that it no longer takes a 60K cache down with it.

Do not raise it much further: the 0.5–0.9 band (16 of 178 selections here) is still worth reusing
partially. If mid-range reprocessing grows after the change, lower it to 0.3.

### Context size (`-c`)

A smaller `-c` permits a lower `ncmoe`, worth up to 9% at benchmark context. But every condense
invalidates the prompt cache — structurally, not fixably (see `--cache-reuse` above). A 36864 context
puts Cline's condense threshold at 29.5K, and observed sessions already reach 28.3K. One rebuild costs
40–70 s; at ~20 tok/s, a 9% generation gain needs roughly 14,000 generated tokens to pay for a single
one. The measured session generated 2,905 tokens in 36 minutes.

## Client setup (Cline)

Point any OpenAI-compatible client at `http://127.0.0.1:8080/v1`. For Cline, **`contextWindow` must be set** — without it Cline grows conversations unbounded until the server hard-errors:

```
error: request (41403 tokens) exceeds the available context size (36864 tokens)
```

Raising `-c` by itself does not fix this — Cline fills whatever it is given (requests grew 25544 → 41403). Cline condenses at 0.9 / 0.7 of `contextWindow` once told, so the client is where the limit belongs:

```json
{
  "provider": "openai-compatible",
  "baseUrl": "http://127.0.0.1:8080/v1",
  "apiKey": "sk-no-key-required",
  "model": "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf",
  "contextWindow": 126976,
  "maxTokens": 4096
}
```

Keep `contextWindow` in sync with the server: `-c` minus `maxTokens`. Do not set it lower than it has to be — condensing is what destroys the prompt cache, so a needlessly small window costs far more than it saves.

Note that Cline only sends sampling parameters when they are explicitly configured, so the server-side defaults above are what it actually gets.

## Model landscape, August 2026

This document's model choice dates from mid-2025. Two things have changed since, and neither is
"a bigger MoE" — nothing in the next size class fits 32GB, let alone 8GB.

**A newer model at the same active-parameter count.** [Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)
(April 2026) has the same ~3B active parameters as Qwen3-Coder-30B-A3B, so it lands in the same speed
class, but scores **73.4% on SWE-bench Verified against 50.3–51.6%** for the incumbent. Its Gated
DeltaNet / Gated Attention hybrid also runs full attention on only one layer in four, so its KV cache
is far smaller — directly useful against the condense problem above. UD-Q3_K_XL is 16.8 GB against the
current 13.81 GB, so `ncmoe` has to rise; whether that trade is worth it on 8GB is measured below.
Use its non-thinking mode for agent work — thinking tokens are latency on every tool call.

**Measured on this machine.** Qwen3.6-35B-A3B UD-Q3_K_XL (16.85 GB) was benchmarked against the
incumbent on the same fixed 21,970-token prompt, same `-ub 2048 -np 2 -kvu`, median of 3:

| Model | `-c` / `-ncmoe` | Prompt processing | Generation @22K | VRAM |
|---|---|---:|---:|---:|
| Qwen3-Coder-30B-A3B UD-Q3_K_XL | 65536 / 40 | 722 t/s | 21.11 tok/s | 7334 MiB |
| Qwen3.6-35B-A3B UD-Q3_K_XL | 65536 / 30 | 879 t/s | **37.26 tok/s** | 7472 MiB |
| **Qwen3.6-35B-A3B UD-Q3_K_XL** | **131072 / 33** | **839 t/s** | **36.38 tok/s** | **7439 MiB** |
| Qwen3.6-35B-A3B UD-Q3_K_XL | 262144 / 40 | 759 t/s | 33.59 tok/s | 7031 MiB |

The bigger file needs *fewer* CPU-resident layers, not more: with only 10 full-attention layers the KV
cache at 65536 is ~0.71 GB against ~3.4 GB for the incumbent, and each active expert is 12.1 MB per
layer per token against 17.1 MB. Halving CPU-side RAM traffic is where the **+72% generation** comes
from. Generation is also nearly flat in context — 36.4 tok/s at 22K against 32.4 tok/s at short context,
because 30 of 40 layers carry a constant-size recurrent state instead of a growing KV cache.

That flatness is what makes `-c 131072` nearly free (−2.4% against 65536) and worth taking: it puts
Cline's condense threshold at 114K, so the 40–70 s prompt-cache rebuilds stop happening at all.
`-c 262144` fits too, at `-ncmoe 40`, but costs 8% for headroom no observed session needs.

VRAM per offloaded layer measured 313 MiB, and the OOM floors are sharp: `-c 65536` fails at
`-ncmoe 29`, `-c 131072` at 32, `-c 262144` at 37.

**Everything else in reach is a sidegrade or does not fit.**

| Model | Total / active | SWE-bench Verified | Fits 8GB + 32GB RAM? |
|---|---|---:|---|
| Qwen3-Coder-30B-A3B (current) | 30.5B / 3.3B | 50.3–51.6% | yes, 13.81 GB |
| **Qwen3.6-35B-A3B** | 35B / 3B | **73.4%** | yes at UD-Q3_K_XL, 16.8 GB |
| Nemotron-Cascade-2-30B-A3B | 30B / 3B | 49.9% (pass@1) | yes, but see below |
| Qwen3-Coder-Next | 80B / 3B | — | no (Q2_K_XL 29.3 GB, Q3_K_XL 36.3 GB) |

Nemotron-Cascade-2 scores 87.2 on LiveCodeBench v6 — beating Kimi-K2.5-1T — but that is competitive
programming, not agentic repository work, and its SWE-bench Verified is no better than the incumbent's.
It is also a Mamba2 hybrid with an open llama.cpp assert bug
([#20570](https://github.com/ggml-org/llama.cpp/issues/20570)). Good for algorithm problems, not for Cline.

> **If you are moving to an AMD gfx906 card (MI50/MI60), read this first.** llama.cpp
> [#19880](https://github.com/ggml-org/llama.cpp/issues/19880) reports that ROCm crashes with
> `hipErrorInvalidDeviceFunction` on exactly the newer Qwen family — Qwen3.5-35B-A3B, Qwen3.5-27B,
> Qwen3.5-122B-A10B and Qwen3-Coder-Next — while **Vulkan works**. The issue is still open. Combined
> with ROCm 6.4+ shipping no gfx906 rocBLAS TensileLibrary, Vulkan is the backend to try first on that
> hardware, alongside the gfx906-specific forks.

## Caveats

- This is a compromise shaped by 8GB of VRAM. With a larger card the model fits entirely in VRAM, `--n-cpu-moe` drops toward 0, and the RAM-bandwidth bottleneck disappears.
- Generated code still needs review. Bugs were found in the 7B and 14B output (missing imports, non-existent parameter names); 30B-A3B was the most reliable but not infallible.
- `llama-cli` / `llama-bench` and `llama-server` use different default context and batch sizes, so the same `-ncmoe` can fit in one and OOM in another. Leave headroom when changing values.

Full measurement logs, per-flag reasoning, and the complete benchmark appendix are in the Korean document below.

---

# 8GB VRAM에서 30B급 MoE 코딩 모델 구동하기

[English](#running-qwen3-coder-30b-a3b-on-an-8gb-gpu) | **한국어**

단일 8GB급 GPU(NVIDIA RTX 2070 SUPER)에서 실사용 가능한 속도로 로컬 코딩 LLM을 돌리기 위한 실험 기록과 최종 설정입니다. 여러 모델 크기(7B/14B/32B/MoE)를 직접 벤치마크하고, 8GB VRAM 제약 아래 속도와 코드 품질의 균형점을 찾았습니다.

## 하드웨어 / 환경

| 항목 | 사양 |
|---|---|
| GPU | NVIDIA GeForce RTX 2070 SUPER (8GB VRAM, 실사용 가능 ~7.6GB) |
| CPU | Intel Core i5-12400F (6코어 / 12스레드) |
| RAM | 32GB DDR5-5600 (**싱글채널** — 1개 슬롯만 사용, 실측 대역폭 29.1 GB/s) |
| OS | Ubuntu 22.04.5 LTS, Kernel 6.8.0 |
| CUDA | Driver 580.173.02 (CUDA 13.0), Toolkit 12.6 |
| 추론 엔진 | [llama.cpp](https://github.com/ggml-org/llama.cpp) (CUDA 백엔드, `-DCMAKE_CUDA_ARCHITECTURES=75`로 소스 빌드) |

## 모델 비교 벤치마크

동일 프롬프트("REST API 클라이언트 클래스 작성: 재시도/백오프/레이트리미팅 포함")로 실측한 결과입니다.

| 모델 | 방식 | 생성 속도 | 비고 |
|---|---|---:|---|
| Qwen2.5-Coder-32B-Instruct Q4_K_M | 덴스, ngl=20 + 추측 디코딩(0.5B draft) | 6.06 tok/s | 품질은 좋으나 실사용엔 너무 느림 |
| Qwen2.5-Coder-14B-Instruct Q4_K_M | 덴스, ngl=41 | 15.1~18.3 tok/s | 논리 버그 1건 발견 |
| Qwen2.5-Coder-7B-Instruct Q5_K_M | 덴스, 전체 GPU 오프로드 | 58.4~64.4 tok/s | 가장 빠르지만 버그 2건(임포트 누락 등) |
| Qwen3-Coder-30B-A3B-Instruct Q4_K_M | MoE (30B 총/3B 활성), `-ncmoe 36` | 28.7~38.4 tok/s | MoE 채택 — 속도와 코드 구조 완성도의 최적 균형점 |
| Qwen3-Coder-30B-A3B-Instruct IQ4_XS | 동일 모델, `-ncmoe 34` | 37.8 tok/s | 같은 모델 더 작은 양자화 → ncmoe 감소 |
| **Qwen3-Coder-30B-A3B-Instruct UD-Q3_K_XL** | **동일 모델, `-ncmoe 32`** | **43.2 tok/s** | **최종 채택** — 대역폭 병목 구조상 양자화를 낮출수록 빨라짐(아래 분석 참고) |

> 위 3개 MoE 행은 모두 같은 모델이며 양자화만 다릅니다. `-c 36864` 운용 컨텍스트 기준 실측이라 벤치마크 기본 컨텍스트 수치보다 낮게 나옵니다.

### 왜 MoE(Qwen3-Coder-30B-A3B)인가

- 총 파라미터는 30B로 지식/추론 용량은 32B 덴스급이지만, 토큰당 실제 활성화되는 파라미터는 3B뿐이라 연산량은 훨씬 작음
- `llama.cpp`의 `--n-cpu-moe`(`-ncmoe`) 옵션으로 전문가(expert) FFN 가중치 일부만 CPU/RAM에 남기고 나머지(어텐션 등)는 GPU에 올리는 하이브리드 오프로드가 가능 — 8GB VRAM에서도 30B급 모델을 돌릴 수 있는 핵심 트릭
- 14B 덴스보다 2.5배 빠르면서, 커스텀 예외 클래스·dataclass·로깅까지 구조화된 코드를 생성하는 등 지금까지 테스트한 것 중 가장 체계적인 결과물을 냄

### 추측 디코딩(Speculative Decoding)에 대한 참고

- 32B 덴스 모델에는 소형 드래프트 모델(Qwen2.5-Coder-0.5B)이 유효했음(6.06 tok/s, 품질 손실 없음)
- 그러나 Qwen3-Coder-30B-A3B에는 **역효과**였음 — MoE 오프로드로 이미 CPU가 바쁜 상태에서 드래프트 모델까지 같은 CPU 코어를 두고 경쟁하며 오히려 17.3 tok/s로 느려짐. 이 조합에서는 추측 디코딩을 쓰지 않는 것이 최적.

## 현재 설정

```bash
~/llama.cpp/build-cuda/bin/llama-server \
  -m ~/models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf \
  -ngl 99 -ncmoe 33 -fa on -t 6 -lm none -np 2 -kvu -ub 2048 \
  -c 131072 -ctk q8_0 -ctv q8_0 -sps 0.5 -rea off \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --host 127.0.0.1 --port 8080
```

체크인된 [`scripts/run-server.sh`](scripts/run-server.sh)가 이 설정입니다. 아래 "2026년 8월 기준 모델 지형"의 A/B 실측에서 동일 VRAM으로 **생성 +72%, 프롬프트 처리 +16%, 컨텍스트 2배**가 확인되어 Qwen3-Coder-30B-A3B 설정을 대체했습니다.

- `-ncmoe 33`: 40개 레이어 중 33개의 전문가 가중치를 CPU에. 이 컨텍스트에서 OOM 바닥은 32입니다
- `-c 131072`: 65536 대비 2.4% 비용으로 Cline 압축 임계를 114k로 올려 캐시 재구축을 없앱니다. 40개 중 10개만 풀 어텐션이라 감당 가능합니다
- **`-sps 0.5`**: 슬롯 재사용 최소 유사도. 기본값 0.10은 거의 무관한 프롬프트가 6만 토큰 캐시를 들고 있는 슬롯을 차지해 파괴하도록 허용합니다(아래 절 참고)
- `-rea off`: thinking 차단. 에이전트 루프에서는 사고 토큰이 매 툴 콜마다 지연으로 쌓입니다
- 샘플링 `--temp 1.0 --top-p 0.95 --top-k 20`: GGUF 메타데이터의 모델 권장값입니다. Qwen3-Coder용 0.7 / 0.8 / 1.05가 **아닙니다**
- cline `contextWindow`는 **126976**(= 131072 − 4096)

`-ub 2048`, `-np 2 -kvu`, `-lm none`, `-t 6`, KV 캐시 `q8_0` 등 이 문서가 확립한 나머지 결론은 그대로 유효합니다 — 모델과 `-ncmoe`, `-c`, 샘플링 값만 달라집니다.

### 이전 설정 (Qwen3-Coder-30B-A3B)

아래 분석 대부분이 이 모델에서 측정된 것이라 함께 남겨둡니다. 더 작은 다운로드로 코더 전용 튜닝 모델을 쓰고 싶다면 여전히 유효한 선택입니다.

```bash
~/llama.cpp/build-cuda/bin/llama-server \
  -m ~/models/Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf \
  -ngl 99 -ncmoe 40 -fa on -t 6 -lm none -np 2 -kvu -ub 2048 \
  -c 65536 -ctk q8_0 -ctv q8_0 \
  --temp 0.7 --top-p 0.8 --top-k 20 --repeat-penalty 1.05 --min-p 0.0 \
  --host 127.0.0.1 --port 8080
```

- **양자화: UD-Q3_K_XL (13.81GB)**. Q4_K_M(18.56GB) → IQ4_XS(16.38GB) → UD-Q3_K_XL 순으로 내려왔음. 아래 "성능 병목 분석"에서 밝혔듯 **생성 속도가 메모리 대역폭에 묶여 있어서, 전문가 가중치가 작아지는 것이 그대로 속도가 됨**. IQ4_XS 대비 파일 16% 감소 → `ncmoe` 34→32 (GPU에 2레이어 더) → **생성 속도 +14~19%**. Unsloth UD(Dynamic) 계열은 중요 텐서를 고정밀로 유지해 3비트대에서도 품질 저하를 최소화함.
- `-ngl 99 -ncmoe 40`: 최대한 GPU에 올리되, 전문가 레이어 40개는 CPU에 남김. 32가 아니라 40인 이유는 `-ub 2048`을 태우기 위해서이며, 그 대가가 실질적으로 0인 것을 실측했습니다(아래 "에이전트 워크로드 튜닝" 참고)
- **`-ub 2048`**: 물리 배치 크기. **프롬프트 처리 +77%(407 → 722 t/s)**, 생성 속도 손실은 측정 노이즈 이내. 이 구성에서 단일 변경으로 얻은 가장 큰 이득입니다
- **`-np 2 -kvu`**: 슬롯 2개 + 통합 KV. 기본값 4슬롯은 서로의 20k+ 프롬프트 캐시를 축출합니다. **`-np`를 단독으로 주면 통합 KV가 조용히 꺼지면서** `-c 65536`이 슬롯당 32768로 반토막 나니 `-kvu`를 반드시 함께 주세요
- `-ctk q8_0 -ctv q8_0`: KV 캐시 8비트 양자화 — 품질 손실 거의 없이 여유 확보, 프롬프트 처리 속도 34% 향상(360→483 t/s)의 부수 효과
- `-lm none`: `mmap` 대신 전량 RAM 직접 로드 — CPU 오프로드 텐서의 페이지 폴트 오버헤드 제거
- `-fa on`: Flash Attention, 품질 손실 없이 무료 속도 향상
- `-c 65536`: Cline처럼 시스템 프롬프트가 긴 툴에서 25544토큰까지 요청이 커지는 걸 확인(`request (25544 tokens) exceeds the available context size (24576 tokens)`). 초기에는 36864를 썼지만, 컨텍스트가 작을수록 Cline의 압축(condense)이 잦아지고 **압축 1회당 프롬프트 캐시가 통째로 무효화되어 40~70초를 뭅니다**. 벤치마크상 `-c 36864 -ncmoe 32`가 9% 빠르지만 그 이득으로 압축 1회를 갚으려면 생성 토큰 약 14,000개가 필요합니다 — 실측 세션은 36분간 2,905토큰을 생성했습니다
- `-t 6`: 물리 코어 수. 하이퍼스레딩(12)을 켜면 오히려 생성 속도가 11% 느려짐(36.1→32.4 tok/s, 캐시 경합 추정) — 실측으로 확인
- **`--temp 0.7 --top-p 0.8 --top-k 20 --repeat-penalty 1.05 --min-p 0.0`**: Qwen3-Coder 공식 권장 샘플링 값. llama-server 기본값(temp 0.80 / top_p 0.95 / top_k 40 / repeat 1.00)은 코딩 에이전트용으로 지나치게 무작위해서 잘못된 API·임포트를 생성할 여지가 큽니다. 속도 비용은 0이므로 반드시 지정하세요. **Cline은 샘플링 값을 명시 설정했을 때만 요청에 실어 보내므로**(전부 optional), 클라이언트에서 따로 지정하지 않으면 이 서버 기본값이 그대로 적용됩니다.
- **KV 캐시를 `q4_0`으로 더 낮추는 것은 이 GPU에서 역효과**였음. `-ctv q4_0`은 컨텍스트를 36864→49152로 늘려주지만(같은 `ncmoe`에서), 동일 조건 비교에서 생성 속도가 44.8→38.8 tok/s로 **13.5% 느려집니다**. Turing(sm_75)에 q4_0 V캐시용 최적화된 flash-attention 경로가 없어 역양자화 비용을 그대로 무는 것으로 보입니다. K/V 모두 `q8_0` 유지가 정답.
- 참고: `--parallel`(동시 슬롯 수)을 줄여도 **VRAM 여유는 안 생깁니다**(KV 풀 크기는 슬롯 수가 아니라 `-c`가 결정). 다만 캐시 적중률 때문에 줄일 가치가 있습니다 — 아래 "에이전트 워크로드 튜닝" 참고. 데스크톱 GPU 점유(Xorg/GNOME ~220MB)도 회수 시도했으나 이 CPU(i5-12400F "F"모델)는 내장 그래픽이 없어 구조적으로 불가능 — 컨텍스트를 늘리려면 `-ncmoe`를 올리거나 더 작은 양자화를 쓰는 것만 유효했음

> **이전 리비전에서는 "Cline만 쓴다면 `-c 36864 -ncmoe 32`가 더 빠르다"고 적었습니다.**
> `tg128` 기준으로는 9% 빠른 게 맞지만(45.3 vs 41.3 tok/s), 실제 세션 로그를 집계해 보니
> 에이전트 클라이언트에는 잘못된 교환이었습니다. 아래 "에이전트 워크로드 튜닝"에 근거를 정리했습니다.
> 압축이 일어나지 않는 짧은 컨텍스트 채팅 용도라면 `-c 36864 -ncmoe 32`가 여전히 더 빠릅니다.

전체 실행 스크립트: [`scripts/run-server.sh`](scripts/run-server.sh)
systemd 유저 서비스 유닛: [`scripts/llama-server.service`](scripts/llama-server.service)

### 측정 재현

이 문서의 수치는 아래 두 스크립트로 나온 것이며, 그대로 다시 돌릴 수 있습니다.

```bash
# 고정 프롬프트(scripts/bench-prompt.txt, 약 22K 토큰) 벤치마크
systemctl --user stop llama-server
./scripts/bench-model.sh ~/models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf 33 2048 131072 mylabel
systemctl --user start llama-server

# 실제 세션 저널 집계 (세션 시작 시각을 인자로)
./scripts/aggregate-session.sh "2026-08-22 18:15:21"
```

`bench-model.sh`는 `cache_prompt:false`로 3회 반복해 중앙값을 내므로 콜드 프롬프트 처리 속도를 재고, `aggregate-session.sh`는 systemd 저널에서 캐시 적중률과 재처리 비용을 뽑습니다. 위 "튜닝 전후 실사용 비교" 표가 후자의 출력입니다.

## 성능 병목 분석 — 이 구성의 속도를 결정하는 것은 GPU가 아니라 RAM 대역폭

MoE 오프로드 구성에서는 매 토큰마다 CPU가 담당한 전문가 레이어의 가중치를 RAM에서 읽어와야 합니다. 이 시스템에서는 그것이 명확한 병목이며, 세 가지 독립적인 측정이 모두 같은 결론을 가리킵니다.

**1) 스레드 스케일링이 조기 포화**

| 스레드 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| tg128 (tok/s) | 25.0 | 32.4 | 36.7 | 38.1 | 38.7 |
| 직전 대비 | — | +30% | +13% | +3.7% | **+1.5%** |

5→6코어에서 겨우 1.5%. CPU 연산 자원이 남는데도 속도가 오르지 않음 = 메모리 버스 포화.

**2) CPU 전문가 레이어당 비용이 완전히 선형**

`-ncmoe` 34 / 40 / 48 실측(IQ4_XS): 39.12 / 35.40 / 31.42 tok/s
→ 토큰당 25.56 / 28.25 / 31.83 ms → **레이어당 일정하게 0.448 ms**

여기서 시간 분해가 나옵니다. `ncmoe=34`일 때 토큰당 25.6ms 중 **15.2ms(약 60%)가 CPU 전문가 가중치 읽기**, 나머지 10.3ms가 GPU 연산 + 고정 오버헤드입니다.

**3) 실측 메모리 대역폭이 이론치와 일치**

`gcc -O3 -march=native -fopenmp` 로 컴파일한 순차 읽기 벤치마크: **29.1 GB/s**.
DDR5-5600 싱글채널 이론 피크 44.8 GB/s의 약 65%로, 전형적인 실측 효율입니다.

### 실용적 결론

- **더 낮은 비트 양자화가 그대로 속도가 됨.** 대역폭 병목이므로 전문가 가중치 크기 감소분이 거의 그대로 생성 속도로 환산됩니다. Q4_K_M → IQ4_XS → UD-Q3_K_XL로 내려오며 속도가 계속 올랐고, 이것이 최종 설정에서 3비트대 양자화를 택한 이유입니다.
- **RAM을 듀얼채널로 만들면 큰 이득이 예상됨.** 이 머신은 32GB 모듈이 `Controller0-DIMM1` 한 곳에만 꽂혀 있어 싱글채널로 동작합니다(`Controller1` 슬롯 2개 공석). 동일 규격 모듈을 하나 더 추가해 대역폭이 2배가 되면 CPU 구간 15.2ms → 약 8.5ms로 줄어 **50 tok/s 이상**이 기대됩니다. 단, VRAM이 큰 GPU로 교체해 모델 전체가 VRAM에 올라가면 이 병목 자체가 사라지므로, GPU 업그레이드 계획이 있다면 우선순위를 비교해 판단하세요.
- **CPU 코어 수를 늘려도 소용없음.** 위 스케일링 곡선이 보여주듯 이 워크로드는 연산이 아니라 대역폭에 묶여 있습니다.
- **`-ot` 텐서 단위 세밀 오프로드는 이 구성에서 무의미했음.** 작은 컨텍스트에서는 `-ncmoe`보다 약 3% 빨랐지만(레이어 단위보다 촘촘하게 VRAM을 채울 수 있어서), 실제 운용 컨텍스트(36864)에서는 이미 VRAM이 포화라 모든 변형이 OOM이었습니다. 참고로 `-ot` 구분자는 `llama-bench`가 `;`, `llama-cli`/`llama-server`가 `,`로 서로 다릅니다.

## 에이전트 워크로드 튜닝 — `tg128`이 말해주지 않는 것

`llama-bench`의 `tg128`은 에이전트 세션의 체감과 다릅니다. 실제 Cline 트래픽 36분치를 서버 로그에서 집계한 결과입니다(`-c 65536 -ncmoe 38 -ub 512` 기준):

| 구간 | 시간 | 토큰 | 속도 |
|---|---:|---:|---:|
| 프롬프트 처리 | 172.1 s | 56,032 | 326 t/s |
| 생성 | 192.9 s | 2,905 | **15.1 tok/s** |

두 가지가 드러납니다. 첫째, **22k 컨텍스트가 찬 상태의 생성 속도는 16~21 tok/s**이지 `tg128`의 45 tok/s가 아닙니다 — 컨텍스트 충전량이 지배하고 `ncmoe`는 거의 영향이 없습니다. 둘째, **프롬프트 처리가 벽시계 시간의 47%**이고 그 중 73%가 22~25k짜리 콜드 프롬프트 단 두 번(56초, 70초)이었습니다.

따라서 실제로 의미 있는 레버는 **재처리를 피하는 것**과 **불가피한 재처리를 빠르게 만드는 것** 둘뿐입니다.

### 배치 크기 `-ub` — 단일 변경 중 최대 이득

MoE 오프로드에서 전문가 가중치는 ubatch당 한 번 RAM에서 읽혀 배치 내 모든 토큰에 상각됩니다. 즉 물리 배치를 키우면 대역폭 병목이 그대로 처리량으로 바뀝니다. 21,970토큰 고정 프롬프트, `cache_prompt: false`, 3회 중앙값:

| `-ub` / `-ncmoe` | 프롬프트 처리 | 생성 @22k | VRAM |
|---|---:|---:|---:|
| 512 / 38 (이전) | 407 t/s | 21.02 tok/s | 7427 MiB |
| 1024 / 38 | 573 t/s | 21.36 | 7551 |
| 2048 / 38 | **OOM** — compute buffer 할당 실패 | — | — |
| **2048 / 40 (채택)** | **722 t/s** | **21.11** | **7334** |
| 2048 / 39 | 710 t/s | 20.92 | 7626 |
| 4096 / 42 | 703 t/s | 20.21 | 6832 |

`-ub 2048`은 `ncmoe 38`에서 VRAM이 모자라지만, 그 대가로 내주는 전문가 레이어 2개는 **사실상 공짜**입니다. 22k 컨텍스트에서 2레이어는 토큰당 시간의 1.9%로 측정 편차보다 작습니다. (짧은 컨텍스트에서는 분해가 되고 "레이어당 0.448 ms" 모델이 그대로 맞습니다 — `ncmoe 40`에서 실측 38.4 tok/s, 예측 38.9 tok/s.) 게다가 레이어 2개를 CPU로 내린 절감이 커진 compute buffer보다 커서 **VRAM이 오히려 줄었습니다**(7427 → 7334 MiB). `-ub 4096`은 무릎을 지난 지점이라 추가 이득이 없고 생성만 깎입니다.

### 튜닝 전후 실사용 비교

같은 방식으로 현재 설정에서 실제 Cline 세션을 집계한 결과입니다.

| | 이전 | 이후 |
|---|---:|---:|
| 모델 / 설정 | Qwen3-Coder-30B-A3B, `-ub 512 -c 65536` | Qwen3.6-35B-A3B, `-ub 2048 -np 2 -kvu -c 131072` |
| 생성 속도 | 15.1 tok/s | **34.3 tok/s** |
| 프롬프트 처리 | 172.1초 / 56,032토큰 | **49.9초 / 22,667토큰** |
| 벽시계 중 프롬프트 비중 | **47%** | **6%** |
| 접두부 캐시 적중 | 18회 중 4회 (22%) | **27회 중 26회 (96%)** |
| 최대 단일 재처리 | 24,761토큰 / **70.3초** | 5,488토큰 / **6.0초** (콜드 스타트) |
| 최대 도달 컨텍스트 | 28,312 | 44,721 |
| 서버 시간 1분당 생성 토큰 | 477 | **1,925** |

이 구간에서는 스톨이 없었습니다. 재처리는 최초 콜드 프롬프트 하나뿐이었고 나머지 26턴은 모두 캐시 적중이었습니다. 컨텍스트가 44,721까지 갔는데도 압축이 걸리지 않았는데, 원래 설정인 `-c 36864`였다면 29.5k에서 압축이 시작됐을 구간입니다.

**다만 이 13분 표본으로 "스톨이 사라졌다"고 볼 수는 없습니다.** 이후 몇 시간을 더 관측하니 40~80초짜리 재처리가 계속 나왔습니다. 이 짧은 표본이 도달하지 못한 두 가지 원인이 있으며, 다음 절에서 다룹니다.

생성 속도는 컨텍스트가 8배 늘어나는 동안 29%만 떨어졌습니다.

```
 5.5k → 33.9      21k → 33.5      38k → 30.9
 6.8k → 39.1      27k → 32.4      43k → 29.9
  12k → 36.6      32k → 32.3      45k → 28.0
```

이전 모델은 같은 범위에서 45% 하락했고(짧은 컨텍스트 38.4 → 22k에서 21.1) 실사용 평균이 15.1이었습니다. 44.7k에서의 28.0 tok/s는 이전 구성이 22k에서 내던 21.1보다도 33% 빠릅니다.

두 가지 해석 주의점이 있습니다. 프롬프트 처리 속도가 벤치마크(839 t/s)보다 낮은 454 t/s로 찍히는데, 이는 27회 중 26회가 캐시 적중이라 매번 증분 토큰 수십 개만 처리했고 그런 소형 호출에서는 고정 오버헤드가 지배하기 때문입니다. 총 처리 토큰이 56,032에서 22,667로 줄어든 것 자체가 캐시가 일하고 있다는 증거이지 성능 저하가 아닙니다. 그리고 이 둘은 서로 다른 실제 세션이지 통제된 실험이 아닙니다 — 통제 비교는 위의 고정 프롬프트 벤치마크 쪽입니다. 여기서 그대로 옮겨지는 것은 구조적 변화, 즉 캐시 적중률과 재처리에 쓰인 벽시계 비중입니다.

### 슬롯 수 `-np`

기본값 4슬롯에서는 세션 로그의 슬롯 선택 18회 중 **14회가 LCP 매칭 실패 후 LRU 폴백**이었습니다. 18~25k짜리 서로 다른 대화 4개가 동시에 올라가 **87k 수요가 65k 통합 KV 풀을 초과**하며 서로를 축출한 것입니다. Cline은 대화 1개 + 소형 보조 요청만 동시에 보내므로 2슬롯이면 충분하고, 정확히 2개일 때는 LRU 특성상 보조 요청이 항상 *반대쪽* 슬롯으로 가서 메인 접두부가 보존됩니다. 슬롯 수는 어느 쪽이든 VRAM에 영향이 없습니다.

> **`-np N`을 주면 통합 KV가 조용히 꺼집니다.** `--kv-unified`의 기본값이 "슬롯 수가 auto일 때만 활성"이라서, `-np 2`만 주면 `-c 65536`이 슬롯당 32768이 되고 컨텍스트 윈도우를 61440으로 설정한 클라이언트가 그대로 깨집니다. 반드시 `-kvu`를 함께 주세요.

### 슬롯 유사도 `-sps` — 그래도 남는 멈춤

장시간 관측하니 40~80초 재처리가 다시 나타났습니다. 원인은 둘이고, 고칠 수 있는 건 하나뿐입니다.

**압축 — 회피 불가.** `contextWindow` 126,976에서 Cline은 약 114k에 압축을 겁니다. 설계대로입니다(실측: 117,052 → 58,588토큰, 115,482 → 35,464토큰). 압축은 대화 앞부분을 요약본으로 갈아끼우므로 접두부가 바뀌고 캐시가 무효화됩니다. 위 `--cache-reuse` A/B에서 이것이 우회 불가임을 코드 수준으로 확인했습니다.

**거의 무관한 프롬프트가 살아 있는 슬롯을 차지 — 고칠 수 있음.** 그날 가장 큰 스톨 두 건의 직전 로그입니다.

```
f_sim_best = 0.103 (> 0.100 thold)  → 53,543토큰 재처리, 76.5초
f_sim_best = 0.100 (> 0.100 thold)  → 55,876토큰 재처리, 81.9초
```

유사도 10%입니다. `--slot-prompt-similarity` 기본값이 **0.10**이라, 시스템 프롬프트 말고는 공유하는 게 없는 요청도 6만 토큰 캐시를 든 슬롯을 차지할 자격이 생기고, 그대로 파괴합니다. 한나절 슬롯 선택 178회 중 156회는 정상(0.9 이상)이었고 0.2 미만은 3회뿐이었지만, 그 3회가 최악의 스톨을 만들었습니다.

임계를 `0.5`로 올리면 그런 요청은 반대쪽 슬롯으로 가고, 살아 있는 캐시는 보존되어 원래 대화가 돌아왔을 때 재처리를 면합니다. 그 요청 자체의 재처리 비용은 그대로이고, 달라지는 건 **6만 토큰 캐시를 같이 끌고 내려가지 않는다**는 점입니다.

더 높이지는 마세요. 0.5~0.9 구간(여기서는 178회 중 16회)은 부분 재사용이 여전히 이득입니다. 적용 후 중간 구간 재처리가 늘어난다면 0.3으로 낮추는 게 맞습니다.

### 컨텍스트 크기 `-c`

`-c`를 줄이면 `ncmoe`를 낮출 수 있어 벤치마크 기준 최대 9%를 법니다. 하지만 압축이 일어날 때마다 프롬프트 캐시가 무효화되며, 이는 원리적으로 회피 불가입니다(위 `--cache-reuse` 절). `-c 36864`는 Cline의 압축 임계를 29.5k에 놓는데 실측 세션은 이미 28.3k에 도달했습니다. 재구축 1회 비용이 40~70초이고, 약 20 tok/s에서 9%의 생성 이득으로 그 1회를 갚으려면 **생성 토큰 약 14,000개**가 필요합니다. 실측 세션은 36분 동안 2,905토큰을 생성했습니다.

## 상시 구동 (systemd)

```bash
mkdir -p ~/.config/systemd/user
cp scripts/llama-server.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now llama-server.service
loginctl enable-linger $(whoami)   # 로그아웃 후에도 계속 구동
```

OpenAI 호환 API가 `http://127.0.0.1:8080/v1`에서 제공됩니다.

## VSCode 연동 (Continue.dev)

1. VSCode + [Continue.dev](https://marketplace.visualstudio.com/items?itemName=Continue.continue) 확장 설치
2. `~/.continue/config.yaml`에 아래 내용 추가:

```yaml
models:
  - name: Qwen3-Coder-30B-A3B (local)
    provider: openai
    model: qwen3-coder-30b-a3b
    apiBase: http://127.0.0.1:8080/v1
    apiKey: none
    contextLength: 126976
    capabilities:
      - tool_use   # 빠뜨리면 Continue.dev가 파일 읽기 등 에이전트/툴 기능을 아예 시도하지 않음
    roles:
      - chat
      - edit
      - apply
      - autocomplete
```

## Cline 연동

- API Provider: `OpenAI Compatible`
- Base URL: `http://127.0.0.1:8080/v1`
- API Key: 아무 값이나(서버가 검사하지 않음)
- Model: `Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf`

### 반드시 `contextWindow`를 지정할 것 — "토큰 초과" 에러의 진짜 원인

Cline에 컨텍스트 한계를 알려주지 않으면 대화를 무한정 쌓다가 서버에서 하드 에러가 납니다:

```
error: request (41403 tokens) exceeds the available context size (36864 tokens)
```

`-c` 값을 계속 올려주는 건 두더지잡기입니다(25544 → 41403으로 계속 커짐). **Cline은 `contextWindow` 값을 받으면 0.9 / 0.7 임계치로 대화를 자동 압축(condense)** 하므로, 서버를 건드리지 말고 클라이언트에 한계를 알려주는 것이 정답입니다. 서버 설정을 그대로 두므로 **속도 손실이 전혀 없습니다.**

Cline CLI를 쓴다면 `~/.cline/data/settings/providers.json`:

```json
{
  "providers": {
    "openai-compatible": {
      "settings": {
        "provider": "openai-compatible",
        "apiKey": "sk-no-key-required",
        "model": "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf",
        "baseUrl": "http://127.0.0.1:8080/v1",
        "contextWindow": 126976,
        "maxTokens": 4096
      }
    }
  }
}
```

수정 후 Cline을 완전히 종료했다 재실행해야 반영됩니다(실행 중이면 종료 시 덮어씁니다).

`contextWindow`는 서버 `-c` 값에서 `maxTokens`를 뺀 값으로 맞추세요. 현재 `-c 131072` / `maxTokens 4096` 이므로 **`126976`** 입니다. 필요 이상으로 작게 잡지 마세요 — 압축이 프롬프트 캐시를 파괴하므로, 작은 윈도우는 아끼는 것보다 훨씬 큰 비용을 물립니다(바로 아래 절 참고).

참고로 서버 쪽에서 컨텍스트를 늘려 해결하려면 `ncmoe`를 올려야 해서 속도를 내줘야 합니다(UD-Q3_K_XL 기준 실측):

| 컨텍스트 | 필요 ncmoe | tg128 | 손실 |
|---|---:|---:|---:|
| **36864 (채택)** | **32** | **45.3** | — |
| 49152 | 34 | 43.6 | −4% |
| 65536 | 38 | 41.3 | −9% |
| 73728 (한계) | 40 | 40.3 | −11% |

### 프롬프트 캐시 동작 — 느려지는 진짜 지점

llama-server의 접두부 캐시는 **순차 대화에서 거의 완벽하게 동작**합니다. 22205토큰짜리 시스템 프롬프트로 3턴을 이어가며 측정한 결과:

| 턴 | 새로 처리한 토큰 | 소요 | 캐시 재사용 |
|---|---:|---:|---:|
| 1회차 | 22205 | 38539 ms | 0 |
| 2회차 | 14 | 177 ms | 22206 |
| 3회차 | 15 | 181 ms | 22221 |

3회차 캐시 히트율 **99.9%**, 재처리 시간 38.5초 → 0.18초.

따라서 체감 지연의 원인은 tok/s가 아니라 **캐시가 통째로 무효화되는 순간**입니다. 운영 로그상 슬롯 선택의 27%(56회 중 15회)가 캐시 미스였고, 한 번의 대가가 매우 큽니다:

| 재처리 토큰 | 소요 시간 |
|---:|---:|
| 30018 | 76.1 초 |
| 26681 | 69.3 초 |
| 14543 | 43.1 초 |

캐시 무효화는 주로 **Cline이 대화를 압축할 때** 일어납니다 — 앞부분 메시지가 요약본으로 교체되면서 접두부 자체가 바뀌기 때문에 구조적으로 회피할 수 없습니다. 그래서 `contextWindow`를 지나치게 작게 잡으면 압축이 잦아져 오히려 느려집니다. 서버 `-c` 값보다 약간 작게, 그러나 너무 작지 않게 잡는 것이 좋습니다.

### `--cache-reuse`로는 해결되지 않음 (A/B 실측)

llama-server의 `--cache-reuse N`은 접두부가 어긋나도 일치하는 청크를 찾아 KV를 새 위치로 이동시켜
재사용하는 옵션입니다. 위의 압축 캐시 무효화를 완화할 유력한 후보로 보여서, 동일 워크로드
(시스템 5,040토큰 + 대화 20,192토큰)로 서버를 두 번 기동해 A/B 했습니다.

| 턴 | 없음 (재사용/재처리) | `--cache-reuse 256` (재사용/재처리) |
|---|---:|---:|
| 1. 최초 (cold) | 0 / 20,192 — 0% | 0 / 20,192 — 0% |
| 2. 순차 대화 | 20,192 / 23 — 99.9% | 20,192 / 23 — 99.9% |
| **3. 압축 직후** | **5,040 / 4,603 — 52.3%** | **5,040 / 4,603 — 52.3%** |
| 4. 압축 후 순차 | 9,643 / 23 — 99.8% | 9,643 / 23 — 99.8% |

**네 턴 모두 토큰 단위까지 완전히 동일합니다.** 효과가 정확히 0입니다.
(2번 턴의 99.9%는 앞 절의 측정을 그대로 재현한 값입니다.)

원인은 구현에 있습니다. `tools/server/server-context.cpp`의 `n_cache_reuse` 블록은 포인터 두 개로
도는데, 캐시 쪽 `head_c`는 1씩 전진하지만 **프롬프트 쪽 `head_p`는 매치가 성립할 때만 전진**합니다.

```
while (head_c < cache.size() && head_p < prompt.size()) {
    n_match = (캐시[head_c..] 와 프롬프트[head_p..] 의 연속 일치 길이)
    if (n_match >= n_cache_reuse) { KV 시프트; head_c += n_match; head_p += n_match; }
    else                          { head_c += 1; }   // head_p 는 그대로
}
```

압축 시점에 `head_p`는 새로 삽입된 **요약문의 첫 토큰**에 놓입니다. 그 요약문은 캐시에 존재한 적이
없으니 `n_cache_reuse` 이상의 일치가 영영 성립하지 않고, 루프는 `head_c`만 끝까지 훑다가 끝납니다.
요약문 바로 뒤에 원문 그대로 남아 있는 최근 메시지 수천 토큰조차 **포인터가 요약문에 묶여
도달하지 못합니다**.

즉 이 옵션이 흡수할 수 있는 것은 캐시 내용의 **삭제**(뒤 내용이 앞으로 당겨지는 경우)뿐이고,
Cline의 압축처럼 요약문이 **삽입**되는 변형은 원리적으로 대상이 아닙니다. 앞 절의 "구조적으로
회피할 수 없다"가 코드 수준에서 확인된 셈입니다.

실질적인 완화책은 여전히 **`contextWindow`를 서버 `-c`에 최대한 근접하게 잡아 압축 횟수 자체를
줄이는 것**뿐입니다.

## 전체 벤치마크 로그

이 프로젝트 진행 중 실측한 모든 조건과 수치입니다. 표기: `pp512` = 프롬프트 처리(t/s), `tg128` = 생성 속도(tok/s). 별도 표기 없으면 `llama-bench` 기준(작은 컨텍스트), "프로덕션"은 실제 운용 컨텍스트(최종 36864)에서 API로 실측한 값입니다.

### Qwen2.5-Coder-32B-Instruct Q4_K_M (덴스, 18.48GB)

| 조건 | pp512 | tg128 | 비고 |
|---|---:|---:|---|
| ngl=0 (전체 CPU) | 139.0 | 2.11 | |
| ngl=8 | 147.7 | 2.44 | |
| ngl=12 | 155.6 | 2.58 | |
| ngl=16 | 161.1 | 2.76 | |
| ngl=20 | 168.1 | 2.96 | 8GB VRAM에서 최대 오프로드 |
| ngl=24 이상 | — | OOM | |
| ngl=20 + 추측 디코딩(0.5B draft, CPU) | — | 4.44~6.06 | 수락률 47.9~72.9%, 프롬프트에 따라 편차 큼 |

### Qwen2.5-Coder-14B-Instruct Q4_K_M (덴스, 8.37GB)

| 조건 | pp512 | tg128 | 비고 |
|---|---:|---:|---|
| ngl=20 | 457.9 | 7.97 | |
| ngl=30 | 556.0 | 10.92 | |
| ngl=40 | 706.8 | 17.54 | |
| ngl=41 | — | 18.27 | 최대 오프로드(bench 기준) |
| ngl=42 이상 | — | OOM | |
| 프로덕션 (ngl=38, 안전마진) | — | **15.1** | 실제 코딩 데모 |

### Qwen2.5-Coder-7B-Instruct Q5_K_M (덴스, 5.07GB, 전체 GPU 오프로드)

| 조건 | pp512 | tg128 | 비고 |
|---|---:|---:|---|
| ngl=99 (bench) | 2045.8 | 64.44 | |
| 프로덕션 | 1083.5 | **58.4** | 실제 코딩 데모 |

### Qwen3-Coder-30B-A3B-Instruct — MoE 오프로드 튜닝 (30.53B 총 / 3B 활성)

**양자화별 ncmoe 스윕** (컨텍스트·KV캐시 설정에 따라 같은 ncmoe도 결과가 다름, 각 행 옆에 조건 표기)

| 양자화(크기) | ncmoe | 조건 | tg128 | 비고 |
|---|---:|---|---:|---|
| Q4_K_M (18.56GB) | 48 | 기본 ctx | 27.67 | |
| Q4_K_M | 40 | 기본 ctx | 32.48 | |
| Q4_K_M | 32 | 기본 ctx | 37.62 | 8GB에서 최대 오프로드 |
| Q4_K_M | 32 | + KV q8_0 | 38.37 | |
| Q4_K_M | 32 | + KV q8_0 + `-lm none` | 37.85 | pp512은 360→483으로 +34%, tg는 동일 |
| Q4_K_M | 24/16/8/0 | — | OOM | |
| Q4_K_M | 34 | ctx=8192, 프로덕션 초기값 | **37.5** | 최초 프로덕션 설정 |
| Q4_K_M | 34 | ctx=24576(스레드=6) | 36.14 | |
| Q4_K_M | 34 | ctx=24576(스레드=12) | 32.38 | 하이퍼스레딩 −11% |
| Q4_K_M | 36 | ctx=32768 | 28.7 | Cline 25544토큰 요구 대응, ncmoe↑ |
| Q4_K_M | 38 | ctx=32768 | 31.2 | |
| Q4_K_M | 40 | ctx=32768 | 30.1 | |
| **IQ4_XS (16.38GB)** | 28/30/32 | ctx=32768 | OOM | |
| IQ4_XS | 34 | ctx=32768 | 29.3 | |
| IQ4_XS | 34 | ctx=36864 | 34.9 | |
| IQ4_XS | 40 | 대역폭 측정용(소형 ctx) | 35.40 | |
| IQ4_XS | 48 | 대역폭 측정용(소형 ctx) | 31.42 | |
| IQ4_XS | 34 | 대역폭 측정용(소형 ctx) | 39.12 | ncmoe당 0.448ms로 완전 선형 확인 |
| IQ4_XS | — | `-ot` 세밀 오프로드(소형 ctx) | 40.27~40.98 | 실제 ctx=36864에선 전부 OOM |
| IQ4_XS | 34 | 프로덕션(ctx=36864) | **37.8** | |
| **UD-Q3_K_XL (13.81GB)** | 26/28/30 | ctx=36864 | OOM | |
| UD-Q3_K_XL | 32 | ctx=36864 (bench) | 45.90 | pp512 450.4 |
| UD-Q3_K_XL | 32 | 프로덕션(ctx=36864) | **43.2** | **최종 채택** |

**컨텍스트 상한 ↔ ncmoe 교환비** (UD-Q3_K_XL, KV q8_0)

| 목표 컨텍스트 | 들어가는 최소 ncmoe | tg128 |
|---:|---:|---:|
| 36864 | 32 | 45.3 |
| 49152 | 34 | 43.6 |
| 65536 | 38 | 41.3 |
| 73728 | 40 | 40.3 |
| 81920 | — | 불가(ncmoe 40까지 OOM) |

**KV 캐시 양자화 비교** (UD-Q3_K_XL, ncmoe=32 고정) — *채택하지 않음*

| K / V | 최대 컨텍스트 | tg128(동일 조건) | 판정 |
|---|---:|---:|---|
| **q8_0 / q8_0** | 36864 | **44.8** | **채택** |
| q8_0 / q4_0 | 49152 | 38.8 | 컨텍스트는 +33%지만 **13.5% 느림** |
| q4_0 / q4_0 | 65536 | — | 컨텍스트는 +78%지만 동일한 이유로 기각 |

> V캐시를 4비트로 낮추면 같은 `ncmoe`에서 컨텍스트를 크게 늘릴 수 있어 매력적으로 보이지만, Turing(sm_75)에는 q4_0 V캐시용 최적화 flash-attention 커널이 없어 역양자화 비용을 그대로 물어 오히려 느려집니다. 프로덕션 실측으로도 43.2 → 34.0 tok/s로 떨어졌습니다. 최신 아키텍처 GPU에서는 결과가 다를 수 있습니다.

**프롬프트 접두부 캐시** (22205토큰 시스템 프롬프트, 순차 3턴)

| 턴 | 신규 처리 토큰 | 소요 | 캐시 재사용 |
|---|---:|---:|---:|
| 1 | 22205 | 38539 ms | 0 |
| 2 | 14 | 177 ms | 22206 |
| 3 | 15 | 181 ms | 22221 |

**스레드 스케일링** (IQ4_XS, ncmoe=34, 소형 ctx 기준 — 대역폭 포화 검증용)

| 스레드 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|
| tg128 | 25.00 | 32.39 | 36.73 | 38.10 | 38.66 |

**추측 디코딩** (MoE에는 역효과, 참고용)

| 조합 | tg | 수락률 |
|---|---:|---:|
| 30B-A3B Q4_K_M + Qwen3-0.6B draft(CPU) | 17.28 | 48.8% |

**RAM 대역폭 실측** (자체 컴파일 벤치마크, `gcc -O3 -march=native -fopenmp`)

| 측정 | 값 |
|---|---:|
| 순차 읽기 대역폭 | 29.1 GB/s |
| DDR5-5600 싱글채널 이론치 대비 | 65% |

### 전체 요약 (모델 간 최고 실측치)

| 모델 | 최고 tg128 | 프로덕션 실측 |
|---|---:|---:|
| Qwen2.5-Coder-32B (덴스+추측디코딩) | 6.06 | 6.06 |
| Qwen2.5-Coder-14B (덴스) | 18.27 | 15.1 |
| Qwen2.5-Coder-30B-A3B Q4_K_M (MoE) | 38.37 | 37.5 |
| Qwen2.5-Coder-30B-A3B IQ4_XS (MoE) | 40.98(소형ctx만) | 37.8 |
| **Qwen2.5-Coder-30B-A3B UD-Q3_K_XL (MoE)** | **45.90** | **43.2** |
| Qwen2.5-Coder-7B (덴스, 전체 GPU) | 64.44 | 58.4 |

> 7B가 raw 속도는 가장 빠르지만 버그 발생률 때문에 채택하지 않았습니다. 자세한 내용은 위 "모델 비교 벤치마크"와 "왜 MoE인가" 참고.

## 2026년 8월 기준 모델 지형

이 문서의 모델 선택은 2025년 중반 기준입니다. 이후 바뀐 것이 둘 있는데, 둘 다 "더 큰 MoE"는 아닙니다 — 다음 체급 MoE 중 32GB에 들어가는 것이 없고, 8GB는 더 말할 것도 없습니다.

### 활성 파라미터가 같은 신형 — Qwen3.6-35B-A3B

[Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)(2026년 4월)는 활성 파라미터가 약 3B로 현재 모델과 같아 **속도 체급이 동일**한데, SWE-bench Verified가 **73.4%**로 현재 모델의 50.3~51.6%를 크게 앞섭니다.

| | Qwen3-Coder-30B-A3B (현재) | Qwen3.6-35B-A3B |
|---|---:|---:|
| 총 / 활성 파라미터 | 30.5B / 3.3B | 35B / 3B |
| SWE-bench Verified | 50.3~51.6% | **73.4%** |
| Terminal-Bench 2.0 | — | 51.5% |
| MCPMark (툴 사용) | — | 37.0% |
| 네이티브 컨텍스트 | 262k | 262k (YaRN 1M+) |

GGUF 크기: UD-Q2_K_XL 12.3GB / **UD-Q3_K_XL 16.8GB** / IQ4_XS 17.7GB / UD-Q4_K_XL 22.4GB / UD-Q5_K_M 26.5GB / UD-Q6_K 29.3GB.

아키텍처가 Gated DeltaNet + Gated Attention 하이브리드라 **4개 레이어 중 1개만 풀 어텐션**입니다. 즉 KV 캐시가 훨씬 작아서, 위에서 다룬 압축(condense) 문제에 직접 유효합니다. 다만 UD-Q3_K_XL이 16.8GB로 현재 13.81GB보다 22% 크므로 8GB에서는 `ncmoe`를 올려야 합니다 — 그 교환이 남는 장사인지는 아래에서 실측합니다.

에이전트 용도로는 **non-thinking 모드**를 쓰세요. 사고 토큰이 매 툴 콜마다 지연으로 쌓입니다.

### 실측 A/B — 이 머신에서 직접 측정

동일한 21,970토큰 고정 프롬프트, 동일한 `-ub 2048 -np 2 -kvu`, 3회 중앙값입니다.

| 모델 | `-c` / `-ncmoe` | 프롬프트 처리 | 생성 @22k | 단컨텍스트 생성 | VRAM |
|---|---|---:|---:|---:|---:|
| Qwen3-Coder-30B-A3B UD-Q3_K_XL | 65536 / 40 | 722 t/s | 21.11 tok/s | 38.4 tok/s | 7334 MiB |
| Qwen3.6-35B-A3B UD-Q3_K_XL | 65536 / 32 | 844 t/s | 36.43 tok/s | 33.7 tok/s | 6845 MiB |
| Qwen3.6-35B-A3B UD-Q3_K_XL | 65536 / 30 | 879 t/s | **37.26 tok/s** | 34.7 tok/s | 7472 MiB |
| **Qwen3.6-35B-A3B UD-Q3_K_XL** | **131072 / 33** | **839 t/s** | **36.38 tok/s** | 32.4 tok/s | **7439 MiB** |
| Qwen3.6-35B-A3B UD-Q3_K_XL | 131072 / 34 | 809 t/s | 35.58 tok/s | 32.1 tok/s | 7110 MiB |
| Qwen3.6-35B-A3B UD-Q3_K_XL | 262144 / 40 | 759 t/s | 33.59 tok/s | 25.6 tok/s | 7031 MiB |

OOM 경계는 명확합니다: `-c 65536`은 `-ncmoe 29`에서, `-c 131072`는 32에서, `-c 262144`는 37에서 실패합니다. 오프로드 레이어당 VRAM은 실측 **313 MiB**였습니다(GGUF 헤더에서 균등 분배로 추정한 387 MiB보다 작음 — UD 계열이 레이어별로 비트를 다르게 할당하기 때문).

**파일이 22% 큰데 `ncmoe`는 오히려 10 낮습니다.** 풀 어텐션이 40개 중 10개뿐이라 65536 KV가 약 0.71GB로, 기존 모델의 3.4GB 대비 2.7GB가 그대로 남기 때문입니다. 여기에 활성 전문가가 레이어·토큰당 12.1MB(기존 17.1MB)라 CPU가 읽는 양이 절반으로 줄고, 그것이 **생성 +72%**의 정체입니다.

**생성 속도가 컨텍스트에 거의 평평합니다.** 22k에서 36.4 tok/s, 짧은 컨텍스트에서 32.4 tok/s로 오히려 긴 쪽이 빠릅니다. 40개 중 30개 레이어가 커지는 KV 대신 **고정 크기 순환 상태**를 쓰기 때문입니다. 기존 모델은 짧은 컨텍스트 38.4 → 22k에서 21.1로 반토막이 났습니다.

이 평평함 덕분에 `-c 131072`가 65536 대비 **−2.4%**밖에 안 되고, 그래서 취할 값어치가 있습니다 — Cline의 압축 임계가 114k로 올라가 이 문서의 핵심 비용이던 40~70초짜리 캐시 재구축이 **아예 발생하지 않게** 됩니다. `-c 262144`도 `-ncmoe 40`으로 들어가지만, 관측된 어떤 세션도 필요로 하지 않는 여유를 위해 8%를 내주는 선택입니다.

전환 시 서버 플래그 두 개가 추가로 필요합니다. **`-rea off`** — thinking 토큰이 매 툴 콜마다 지연으로 쌓이므로 에이전트 용도에서는 꺼야 합니다(끈 상태에서 `reasoning_content: None`, 툴 콜 정상 동작 확인). 그리고 샘플링은 GGUF 메타데이터의 모델 권장값인 **`--temp 1.0 --top-p 0.95 --top-k 20`** 으로 바꿔야 합니다 — Qwen3-Coder용 0.7/0.8/1.05가 아닙니다.

### 나머지는 사이드그레이드이거나 안 들어감

**Nemotron-Cascade-2-30B-A3B**(NVIDIA, Mamba2-Transformer 하이브리드 MoE)는 LiveCodeBench v6에서 87.2로 Kimi-K2.5-1T(85.0)까지 제칩니다. 그러나 그건 경쟁 프로그래밍이고, **SWE-bench Verified는 pass@1 49.9로 현재 모델과 동급**입니다. Cline은 리포지토리 에이전트 작업이라 후자가 결정적입니다. 게다가 llama.cpp에 Mamba 관련 assert 버그([#20570](https://github.com/ggml-org/llama.cpp/issues/20570))가 열려 있습니다. 알고리즘 문제 풀이에는 좋지만 Cline용은 아닙니다.

**Qwen3-Coder-Next(80B-A3B)** 는 진짜 상급 코더 MoE입니다(512 전문가 중 10개 활성, 256k 컨텍스트). 하지만 Q3_K_XL 36.3GB / Q4_K_M 48.5GB로 32GB에도 안 들어가고, Q2_K_XL 29.3GB는 KV 자리가 남지 않습니다. 64GB(예: MI50 2장)부터 현실적인 타깃입니다.

### gfx906(MI50/MI60)으로 옮길 계획이라면 먼저 읽을 것

llama.cpp [#19880](https://github.com/ggml-org/llama.cpp/issues/19880)에 따르면 **ROCm에서 신형 Qwen 계열이 크래시합니다** — `rocBLAS error ... 'hipErrorInvalidDeviceFunction':98`. 보고된 대상이 Qwen3.5-35B-A3B, Qwen3.5-27B, Qwen3.5-122B-A10B, Qwen3-Coder-Next로 **DeltaNet 하이브리드 계열 전부**이고, **Vulkan에서는 정상 동작**합니다. 이슈는 아직 열려 있습니다.

ROCm 6.4+에 gfx906용 rocBLAS TensileLibrary가 아예 빠져 있다는 점까지 합치면, 이 하드웨어에서는 **Vulkan 백엔드를 1순위로** 두고 gfx906 전용 포크([iacopPBK/llama.cpp-gfx906](https://github.com/iacopPBK/llama.cpp-gfx906) 등)와 함께 비교하는 편이 현실적입니다.

## 알아둘 점 / 한계

- 이 구성은 8GB VRAM 제약 아래 낸 절충안입니다. VRAM이 더 크면(예: MI50 32GB) `--n-cpu-moe`를 낮추거나 아예 0으로 두어 전체를 GPU에 올려 훨씬 더 빠르게 돌릴 수 있습니다.
- 생성된 코드는 사람의 검토가 필요합니다. 7B/14B 모델에서 실제 버그(잘못된 import, 존재하지 않는 파라미터명 등)를 여러 건 확인했습니다. 30B-A3B가 가장 안정적이었지만 무조건적인 신뢰는 금물입니다.
- `llama-cli`/`llama-bench`와 `llama-server`는 기본 컨텍스트/배치 크기가 달라 동일 `-ncmoe` 값에서도 VRAM 요구량이 다를 수 있습니다. 실제 값을 바꿀 때는 여유분을 두고 테스트하는 것을 권장합니다.

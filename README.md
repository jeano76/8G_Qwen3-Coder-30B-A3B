# Running Qwen3-Coder-30B-A3B on an 8GB GPU

**English** | [한국어](#8gb-vram에서-qwen3-coder-30b-a3b-로컬-코딩-어시스턴트-구동하기)

Measured notes and the final configuration for running a 30B-class coding LLM at usable speed on a single 8GB GPU. Every number below was measured on the machine described here — including the approaches that turned out to be slower.

**Result: 45.07 tok/s generation, 438.73 t/s prompt processing**, running a 30.5B-parameter model on a GPU that cannot hold it.

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

## Final configuration

```bash
llama-server \
  -m Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf \
  -ngl 99 -ncmoe 32 -fa on -t 6 -lm none \
  -c 36864 -ctk q8_0 -ctv q8_0 \
  --temp 0.7 --top-p 0.8 --top-k 20 --repeat-penalty 1.05 --min-p 0.0 \
  --host 127.0.0.1 --port 8080
```

| Flag | Why |
|---|---|
| `-ncmoe 32` | Keep 32 layers' expert weights on the CPU; everything else on the GPU |
| `-lm none` | Load fully into RAM instead of mmap — removes page-fault overhead on offloaded tensors (prompt processing +34%) |
| `-t 6` | Physical cores only. Using all 12 threads is **11% slower** (36.1 → 32.4 tok/s) |
| `-fa on` | Flash Attention — free speed, no quality cost |
| `-ctk/-ctv q8_0` | 8-bit KV cache. Do **not** drop V to `q4_0` on this GPU (see below) |
| sampling flags | Qwen3-Coder's recommended values. llama-server defaults (temp 0.80 / top_p 0.95 / top_k 40) are too loose for agent coding and invite wrong APIs and imports |

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
| Reducing `--parallel` | No VRAM freed; the server uses a unified KV cache across slots |

## Client setup (Cline)

Point any OpenAI-compatible client at `http://127.0.0.1:8080/v1`. For Cline, **`contextWindow` must be set** — without it Cline grows conversations unbounded until the server hard-errors:

```
error: request (41403 tokens) exceeds the available context size (36864 tokens)
```

Raising `-c` is the wrong fix: requests kept growing (25544 → 41403), and a larger context forces `ncmoe` up, costing 4–11% throughput. Cline condenses at 0.9 / 0.7 of `contextWindow` once told, so setting it leaves the fast server config untouched:

```json
{
  "provider": "openai-compatible",
  "baseUrl": "http://127.0.0.1:8080/v1",
  "apiKey": "sk-no-key-required",
  "model": "Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf",
  "contextWindow": 36864,
  "maxTokens": 4096
}
```

Note that Cline only sends sampling parameters when they are explicitly configured, so the server-side defaults above are what it actually gets.

## Caveats

- This is a compromise shaped by 8GB of VRAM. With a larger card the model fits entirely in VRAM, `--n-cpu-moe` drops toward 0, and the RAM-bandwidth bottleneck disappears.
- Generated code still needs review. Bugs were found in the 7B and 14B output (missing imports, non-existent parameter names); 30B-A3B was the most reliable but not infallible.
- `llama-cli` / `llama-bench` and `llama-server` use different default context and batch sizes, so the same `-ncmoe` can fit in one and OOM in another. Leave headroom when changing values.

Full measurement logs, per-flag reasoning, and the complete benchmark appendix are in the Korean document below.

---

# 8GB VRAM에서 Qwen3-Coder-30B-A3B 로컬 코딩 어시스턴트 구동하기

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

## 최종 설정

```bash
~/llama.cpp/build-cuda/bin/llama-server \
  -m ~/models/Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf \
  -ngl 99 -ncmoe 32 -fa on -t 6 -lm none \
  -c 36864 -ctk q8_0 -ctv q8_0 \
  --temp 0.7 --top-p 0.8 --top-k 20 --repeat-penalty 1.05 --min-p 0.0 \
  --host 127.0.0.1 --port 8080
```

- **양자화: UD-Q3_K_XL (13.81GB)**. Q4_K_M(18.56GB) → IQ4_XS(16.38GB) → UD-Q3_K_XL 순으로 내려왔음. 아래 "성능 병목 분석"에서 밝혔듯 **생성 속도가 메모리 대역폭에 묶여 있어서, 전문가 가중치가 작아지는 것이 그대로 속도가 됨**. IQ4_XS 대비 파일 16% 감소 → `ncmoe` 34→32 (GPU에 2레이어 더) → **생성 속도 +14~19%**. Unsloth UD(Dynamic) 계열은 중요 텐서를 고정밀로 유지해 3비트대에서도 품질 저하를 최소화함.
- `-ngl 99 -ncmoe 32`: 최대한 GPU에 올리되, 전문가 레이어 32개는 CPU에 남김
- `-ctk q8_0 -ctv q8_0`: KV 캐시 8비트 양자화 — 품질 손실 거의 없이 여유 확보, 프롬프트 처리 속도 34% 향상(360→483 t/s)의 부수 효과
- `-lm none`: `mmap` 대신 전량 RAM 직접 로드 — CPU 오프로드 텐서의 페이지 폴트 오버헤드 제거
- `-fa on`: Flash Attention, 품질 손실 없이 무료 속도 향상
- `-c 36864`: Cline처럼 시스템 프롬프트가 긴 툴에서 25544토큰까지 요청이 커지는 걸 확인(`request (25544 tokens) exceeds the available context size (24576 tokens)`) → IQ4_XS + ncmoe=34 조합으로 36864까지 확장(요구치 대비 44% 여유), 생성 속도는 오히려 34.9 tok/s로 준수함
- `-t 6`: 물리 코어 수. 하이퍼스레딩(12)을 켜면 오히려 생성 속도가 11% 느려짐(36.1→32.4 tok/s, 캐시 경합 추정) — 실측으로 확인
- **`--temp 0.7 --top-p 0.8 --top-k 20 --repeat-penalty 1.05 --min-p 0.0`**: Qwen3-Coder 공식 권장 샘플링 값. llama-server 기본값(temp 0.80 / top_p 0.95 / top_k 40 / repeat 1.00)은 코딩 에이전트용으로 지나치게 무작위해서 잘못된 API·임포트를 생성할 여지가 큽니다. 속도 비용은 0이므로 반드시 지정하세요. **Cline은 샘플링 값을 명시 설정했을 때만 요청에 실어 보내므로**(전부 optional), 클라이언트에서 따로 지정하지 않으면 이 서버 기본값이 그대로 적용됩니다.
- **KV 캐시를 `q4_0`으로 더 낮추는 것은 이 GPU에서 역효과**였음. `-ctv q4_0`은 컨텍스트를 36864→49152로 늘려주지만(같은 `ncmoe`에서), 동일 조건 비교에서 생성 속도가 44.8→38.8 tok/s로 **13.5% 느려집니다**. Turing(sm_75)에 q4_0 V캐시용 최적화된 flash-attention 경로가 없어 역양자화 비용을 그대로 무는 것으로 보입니다. K/V 모두 `q8_0` 유지가 정답.
- 참고: `--parallel`(동시 슬롯 수)은 `kv_unified` 모드라 슬롯 수를 줄여도 VRAM 여유가 안 생김. 데스크톱 GPU 점유(Xorg/GNOME ~220MB)도 회수 시도했으나 이 CPU(i5-12400F "F"모델)는 내장 그래픽이 없어 구조적으로 불가능 — 컨텍스트를 늘리려면 `-ncmoe`를 올리거나 더 작은 양자화를 쓰는 것만 유효했음

전체 실행 스크립트: [`scripts/run-server.sh`](scripts/run-server.sh)
systemd 유저 서비스 유닛: [`scripts/llama-server.service`](scripts/llama-server.service)

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
    contextLength: 36864
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
        "model": "Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf",
        "baseUrl": "http://127.0.0.1:8080/v1",
        "contextWindow": 36864,
        "maxTokens": 4096
      }
    }
  }
}
```

수정 후 Cline을 완전히 종료했다 재실행해야 반영됩니다(실행 중이면 종료 시 덮어씁니다).

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

## 알아둘 점 / 한계

- 이 구성은 8GB VRAM 제약 아래 낸 절충안입니다. VRAM이 더 크면(예: MI50 32GB) `--n-cpu-moe`를 낮추거나 아예 0으로 두어 전체를 GPU에 올려 훨씬 더 빠르게 돌릴 수 있습니다.
- 생성된 코드는 사람의 검토가 필요합니다. 7B/14B 모델에서 실제 버그(잘못된 import, 존재하지 않는 파라미터명 등)를 여러 건 확인했습니다. 30B-A3B가 가장 안정적이었지만 무조건적인 신뢰는 금물입니다.
- `llama-cli`/`llama-bench`와 `llama-server`는 기본 컨텍스트/배치 크기가 달라 동일 `-ncmoe` 값에서도 VRAM 요구량이 다를 수 있습니다. 실제 값을 바꿀 때는 여유분을 두고 테스트하는 것을 권장합니다.

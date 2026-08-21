# 8GB VRAM에서 Qwen3-Coder-30B-A3B 로컬 코딩 어시스턴트 구동하기

단일 8GB급 GPU(NVIDIA RTX 2070 SUPER)에서 실사용 가능한 속도로 로컬 코딩 LLM을 돌리기 위한 실험 기록과 최종 설정입니다. 여러 모델 크기(7B/14B/32B/MoE)를 직접 벤치마크하고, 8GB VRAM 제약 아래 속도와 코드 품질의 균형점을 찾았습니다.

## 하드웨어 / 환경

| 항목 | 사양 |
|---|---|
| GPU | NVIDIA GeForce RTX 2070 SUPER (8GB VRAM, 실사용 가능 ~7.6GB) |
| CPU | Intel Core i5-12400F (6코어 / 12스레드) |
| RAM | 32GB |
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
| **Qwen3-Coder-30B-A3B-Instruct Q4_K_M** | **MoE (30B 총/3B 활성), `--n-cpu-moe 34`** | **37.5~38.4 tok/s** | **최종 채택** — 속도와 코드 구조 완성도의 최적 균형점 |

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
  -m ~/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  -ngl 99 -ncmoe 34 -fa on -t 6 -lm none \
  -c 8192 -ctk q8_0 -ctv q8_0 \
  --host 127.0.0.1 --port 8080
```

- `-ngl 99 -ncmoe 34`: 최대한 GPU에 올리되, VRAM 한계로 전문가 레이어 34개는 CPU에 남김 (실측 최대 오프로드 한계는 32, 서버 안정성을 위해 여유 2 추가)
- `-ctk q8_0 -ctv q8_0`: KV 캐시 8비트 양자화 — 품질 손실 거의 없이 여유 확보, 프롬프트 처리 속도 34% 향상(360→483 t/s)의 부수 효과
- `-lm none`: `mmap` 대신 전량 RAM 직접 로드 — CPU 오프로드 텐서의 페이지 폴트 오버헤드 제거
- `-fa on`: Flash Attention, 품질 손실 없이 무료 속도 향상

전체 실행 스크립트: [`scripts/run-server.sh`](scripts/run-server.sh)
systemd 유저 서비스 유닛: [`scripts/llama-server.service`](scripts/llama-server.service)

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
    contextLength: 8192
    roles:
      - chat
      - edit
      - apply
      - autocomplete
```

## 알아둘 점 / 한계

- 이 구성은 8GB VRAM 제약 아래 낸 절충안입니다. VRAM이 더 크면(예: MI50 32GB) `--n-cpu-moe`를 낮추거나 아예 0으로 두어 전체를 GPU에 올려 훨씬 더 빠르게 돌릴 수 있습니다.
- 생성된 코드는 사람의 검토가 필요합니다. 7B/14B 모델에서 실제 버그(잘못된 import, 존재하지 않는 파라미터명 등)를 여러 건 확인했습니다. 30B-A3B가 가장 안정적이었지만 무조건적인 신뢰는 금물입니다.
- `llama-cli`/`llama-bench`와 `llama-server`는 기본 컨텍스트/배치 크기가 달라 동일 `-ncmoe` 값에서도 VRAM 요구량이 다를 수 있습니다. 실제 값을 바꿀 때는 여유분을 두고 테스트하는 것을 권장합니다.

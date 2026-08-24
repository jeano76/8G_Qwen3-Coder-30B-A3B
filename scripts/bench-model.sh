#!/bin/bash
# 고정 프롬프트로 llama-server 설정을 벤치마크한다.
#
#   ./scripts/bench-model.sh <model.gguf> <ncmoe> <ub> <ctx> <label> [runs]
#
# 환경변수: LLAMA_SERVER (바이너리 경로), EXTRA_ARGS (추가 서버 플래그)
#
# 같은 프롬프트(scripts/bench-prompt.txt, 약 22K 토큰)를 cache_prompt:false로
# 반복 측정하므로 README의 수치와 직접 비교할 수 있다. 측정값은 프롬프트 처리
# t/s, 긴 컨텍스트 생성 tok/s, 짧은 컨텍스트 생성 tok/s, VRAM.
#
# 8080 포트를 쓰므로 상시 구동 중이면 먼저 내릴 것:
#   systemctl --user stop llama-server
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 빌드 A/B 는 LLAMA_SERVER 로, 플래그 A/B 는 EXTRA_ARGS 로 한다.
#   LLAMA_SERVER=~/llama.cpp/build-cuda-new/bin/llama-server \
#   EXTRA_ARGS="--spec-type ngram-mod" ./scripts/bench-model.sh ...
BIN="${LLAMA_SERVER:-$HOME/llama.cpp/build-cuda/bin/llama-server}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
M=$1; NCMOE=$2; UB=$3; CTX=$4; LABEL=$5; RUNS=${6:-3}
LOG="$(mktemp -t bench-$LABEL-XXXX.log)"

# shellcheck disable=SC2086
"$BIN" -m "$M" \
  -ngl 99 -ncmoe "$NCMOE" -fa on -t 6 -lm none -np 2 -kvu \
  -c "$CTX" -ctk q8_0 -ctv q8_0 -ub "$UB" $EXTRA_ARGS \
  --host 127.0.0.1 --port 8080 > "$LOG" 2>&1 &
PID=$!

READY=0
for _ in $(seq 1 180); do
  curl -sf -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1 && { READY=1; break; }
  kill -0 $PID 2>/dev/null || break
  read -t 1 < /dev/zero 2>/dev/null || true
done
if [ $READY -ne 1 ]; then
  echo "RESULT $LABEL ncmoe=$NCMOE ub=$UB ctx=$CTX FAILED"
  grep -iE "error|out of memory|failed" "$LOG" | tail -4
  kill $PID 2>/dev/null; wait $PID 2>/dev/null; exit 1
fi

echo "--- $LABEL (ncmoe=$NCMOE ub=$UB ctx=$CTX) VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || echo n/a)"
PROMPT_FILE="$DIR/bench-prompt.txt" RUNS="$RUNS" LABEL="$LABEL" NCMOE="$NCMOE" UB="$UB" python3 - <<'PY'
import json, os, urllib.request
prompt = open(os.environ['PROMPT_FILE']).read()
def call(p, npred, cache=False):
    req = urllib.request.Request('http://127.0.0.1:8080/completion',
        data=json.dumps({'prompt': p, 'n_predict': npred,
                         'cache_prompt': cache, 'temperature': 0}).encode(),
        headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=900))['timings']
call("warmup", 4)
short = call("Write a Python function that reverses a linked list.", 128)['predicted_per_second']
pp, tg = [], []
for i in range(int(os.environ['RUNS'])):
    t = call(prompt, 64, False)
    pp.append(t['prompt_per_second']); tg.append(t['predicted_per_second'])
    print(f"  run{i+1}: pp {t['prompt_n']} tok / {t['prompt_ms']/1000:.1f}s "
          f"= {t['prompt_per_second']:.1f} t/s | tg {t['predicted_per_second']:.2f} tok/s")
med = lambda x: sorted(x)[len(x)//2]
print(f"RESULT {os.environ['LABEL']} ncmoe={os.environ['NCMOE']} ub={os.environ['UB']} "
      f"pp={med(pp):.1f} tg_long={med(tg):.2f} tg_short={short:.1f}")
PY
kill $PID 2>/dev/null; wait $PID 2>/dev/null

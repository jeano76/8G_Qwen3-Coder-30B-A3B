#!/usr/bin/env bash
# llama-server 스톨 감지 워치독.
#
# 배경: llama-server 는 크래시가 아니라 "행"으로 죽는다 (2026-08-23 21분 30초 스톨).
# 크래시가 아니므로 systemd 의 Restart=always 는 개입하지 않는다.
# /health 는 메인 루프를 거치지 않는 단순 핸들러라 행 상태에서도 200 을 반환한다.
# /slots 는 메인 태스크 큐를 경유하므로, 여기가 막히면 추론 루프가 막힌 것이다.
set -uo pipefail

UNIT="llama-server.service"
ENDPOINT="${LLAMA_ENDPOINT:-http://127.0.0.1:8080}"
PROBE_TIMEOUT="${PROBE_TIMEOUT:-15}"     # /slots 응답 대기 (초)
FAIL_THRESHOLD="${FAIL_THRESHOLD:-5}"    # 연속 실패 몇 회면 조치 (타이머 60초 => 5분)
COOLDOWN="${COOLDOWN:-900}"              # 재시작 간 최소 간격 (초)
MAX_PER_HOUR="${MAX_PER_HOUR:-3}"        # 시간당 재시작 상한 (플랩 방지)
PROGRESS_WINDOW="${PROGRESS_WINDOW:-6}"  # 진행 흔적을 찾을 기간 (분)

STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/llama-watchdog"
FAIL_FILE="$STATE_DIR/consecutive_failures"
HIST_FILE="$STATE_DIR/restart_history"     # 재시작 epoch 목록 (한 줄에 하나)
PRESSURE_LOG="$STATE_DIR/pressure.log"     # 프리즈 사후 분석용 압력 샘플
PRESSURE_LOG_MAX="${PRESSURE_LOG_MAX:-5242880}"   # 5MB 넘으면 .1 로 회전
CLINE_PROVIDERS="${CLINE_PROVIDERS:-$HOME/.cline/data/settings/providers.json}"
DRIFT_FILE="$STATE_DIR/ctx_drift_warned"   # 마지막 불일치 경고 epoch
DRIFT_QUIET="${DRIFT_QUIET:-3600}"         # 같은 불일치는 1시간에 한 번만 경고
# 남겨둘 컨텍스트. "생성용 여유"가 아니라 "툴 결과 하나가 더 들어올 자리"다.
# Cline 의 자동 압축은 턴과 턴 사이에서만 사용률을 보므로, 여유보다 큰 툴 결과
# 하나가 대화를 서버의 벽 너머로 한 번에 밀어버린다. 317턴 세션에서 측정한 턴 간
# 증가폭은 46,102 / 34,859 / 17,134 / 16,500 / 14,706 … 이었다.
# README "자동 압축이 항상 구해주지는 않는 이유" 절.
CTX_RESERVE="${CTX_RESERVE:-16384}"
mkdir -p "$STATE_DIR"

log() { echo "[watchdog] $*"; }

# ── 압력 샘플러 ────────────────────────────────────────────────────────────
# 하드 프리즈는 저널의 마지막 수 초를 가져간다. 2026-08-23 22:39 프리즈가
# OOM 킬도 Xid 도 hung task 도 없이 "원인 미상"으로 남은 이유다.
# 그래서 매 틱마다 한 줄을 파일에 쓰고 **즉시 fsync** 한다. 페이지 캐시에만
# 있으면 리셋과 함께 사라지므로 fsync 없이는 이 로그도 저널과 같이 없어진다.
# 틱은 60초지만 PSI 의 avg10/avg60 을 같이 적으므로 그 사이 구간도 복원된다.

_gib() { awk -v b="${1:-0}" 'BEGIN{ if (b=="") b=0; printf "%.2fG", b/1073741824 }'; }

# memory.stat / memory.events 에서 키 하나
_kv() { awk -v k="$2" '$1==k {print $2; exit}' "$1" 2>/dev/null; }

# PSI 파일에서 "avg10/avg60"
_psi() {
  awk -v k="$2" '$1==k {
    a="?"; b="?"
    for (i=2; i<=NF; i++) {
      if ($i ~ /^avg10=/) { split($i, x, "="); a=x[2] }
      if ($i ~ /^avg60=/) { split($i, y, "="); b=y[2] }
    }
    print a"/"b; exit
  }' "$1" 2>/dev/null
}

sample_pressure() {
  local ts state cg dir line sz
  ts=$(date -Is)
  if systemctl --user is-active --quiet "$UNIT"; then state=active; else state=inactive; fi

  dir=""
  cg=$(systemctl --user show "$UNIT" -p ControlGroup --value 2>/dev/null || true)
  [ -n "$cg" ] && [ -d "/sys/fs/cgroup$cg" ] && dir="/sys/fs/cgroup$cg"

  line="$ts state=$state"
  if [ -n "$dir" ]; then
    line="$line cur=$(_gib "$(cat "$dir/memory.current" 2>/dev/null)")"
    line="$line high=$(_kv "$dir/memory.events" high)"
    line="$line max=$(_kv "$dir/memory.events" max)"
    line="$line oomk=$(_kv "$dir/memory.events" oom_kill)"
    line="$line anon=$(_gib "$(_kv "$dir/memory.stat" anon)")"
    line="$line file=$(_gib "$(_kv "$dir/memory.stat" file)")"
    line="$line majflt=$(_kv "$dir/memory.stat" pgmajfault)"
    line="$line cg_mem=$(_psi "$dir/memory.pressure" some)"
    line="$line cg_io=$(_psi "$dir/io.pressure" some)"
  fi
  line="$line sys_mem=$(_psi /proc/pressure/memory some)"
  line="$line sys_io=$(_psi /proc/pressure/io full)"
  line="$line avail=$(awk '/^MemAvailable:/{printf "%.2fG", $2/1048576}' /proc/meminfo)"
  line="$line swapfree=$(awk '/^SwapFree:/{printf "%.2fG", $2/1048576}' /proc/meminfo)"

  if [ -f "$PRESSURE_LOG" ]; then
    sz=$(stat -c %s "$PRESSURE_LOG" 2>/dev/null || echo 0)
    [ "$sz" -gt "$PRESSURE_LOG_MAX" ] && mv -f "$PRESSURE_LOG" "$PRESSURE_LOG.1"
  else
    printf '%s\n' "# ts state cur=cgroup메모리 high/max/oomk=memory.events anon/file/majflt=memory.stat" \
                   "# cg_*/sys_*=PSI stall avg10/avg60 (sys_io 는 full, 나머지는 some) avail/swapfree=/proc/meminfo" \
      >> "$PRESSURE_LOG"
  fi
  printf '%s\n' "$line" >> "$PRESSURE_LOG"
  sync -d "$PRESSURE_LOG" 2>/dev/null || sync
}

read_fails() { [ -f "$FAIL_FILE" ] && cat "$FAIL_FILE" 2>/dev/null || echo 0; }
reset_fails() { echo 0 > "$FAIL_FILE"; }

# ── 컨텍스트 설정 드리프트 검사 ────────────────────────────────────────────
# 서버 -c 를 바꾸고 Cline 쪽 contextWindow 를 안 고치면, Cline 은 없는 창을
# 있다고 믿고 대화를 키우다 서버에서 하드 에러를 맞는다:
#   error: request (67992 tokens) exceeds the available context size (65536 tokens)
# 2026-08-23 에 -c 를 131072 -> 65536 으로 내리면서 실제로 이 일이 났다.
# 반대로 너무 작게 잡아도 손해다 - 불필요한 압축이 프롬프트 캐시를 버린다.
check_context_drift() {
  [ -f "$CLINE_PROVIDERS" ] || return 0

  local pid ctx vals cw mt pct msg now last
  pid=$(systemctl --user show "$UNIT" -p MainPID --value 2>/dev/null || echo 0)
  [ -n "${pid:-}" ] && [ "$pid" != "0" ] || return 0

  # 실행 중인 프로세스의 실제 -c (스크립트가 아니라 프로세스가 진실이다)
  ctx=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null \
        | awk '$0=="-c" || $0=="--ctx-size" { getline; print; exit }')
  [ -n "${ctx:-}" ] || return 0

  vals=$(python3 - "$CLINE_PROVIDERS" <<'PY' 2>/dev/null || true
import json, sys
d = json.load(open(sys.argv[1]))
for prov in d.get("providers", {}).values():
    s = prov.get("settings", {})
    if "127.0.0.1:8080" in s.get("baseUrl", ""):
        print(s.get("contextWindow", 0), s.get("maxTokens", 0))
        break
PY
)
  [ -n "${vals:-}" ] || return 0
  set -- $vals; cw=$1; mt=$2
  [ "${cw:-0}" -gt 0 ] 2>/dev/null || return 0

  # providers.json 의 maxTokens 는 실제로 전송되는 값이 아니다. 2026-08-24 실측:
  # maxTokens 4096 인데 슬롯의 n_predict 는 32000 이었다. 예약해야 할 양은
  # 설정값이 아니라 서버가 실제로 받은 n_predict 다. 못 읽으면 설정값으로 돌아간다.
  local npred
  npred=$(curl -s -m "$PROBE_TIMEOUT" "$ENDPOINT/slots" 2>/dev/null \
          | python3 -c 'import json,sys; d=json.load(sys.stdin); print(max((s.get("params",{}).get("n_predict",0) or 0) for s in d))' 2>/dev/null || true)
  if [ -n "${npred:-}" ] && [ "${npred:-0}" -gt 0 ] 2>/dev/null && [ "$npred" -gt "$mt" ]; then
    mt="$npred"
  fi

  # 판정 기준은 "선언된 생성 상한"이 아니라 실용 예약분이다. n_predict 는 클라이언트가
  # 선언한 천장일 뿐이고(2026-08-24 실측 32000), 실제 생성은 중앙값 119 / 최대 1897 이었다.
  # 32000 을 통째로 예약하면 창의 절반을 버리게 되므로, 여유는 CTX_RESERVE 로 잡고
  # 선언된 상한은 경고 문구에 근거로만 싣는다.
  local want=$(( ctx - CTX_RESERVE ))
  msg=""
  if [ "$cw" -gt "$want" ]; then
    msg="Cline contextWindow($cw) 가 서버 -c ($ctx) 에 너무 붙어 있다(여유 $(( ctx - cw ))토큰). 슬롯에 선언된 생성 상한은 ${mt} 이고, 그보다 큰 툴 결과 한 번이면 'exceeds the available context size' 로 떨어진다. contextWindow ${want} 권장(예약 ${CTX_RESERVE})"
  else
    pct=$(( cw * 100 / ctx ))
    if [ "$pct" -lt 70 ]; then
      msg="Cline contextWindow($cw) 가 서버 -c ($ctx) 의 ${pct}% 뿐이다. 압축이 불필요하게 자주 돌아 프롬프트 캐시를 버린다. contextWindow ${want} 권장"
    fi
  fi

  if [ -z "$msg" ]; then
    rm -f "$DRIFT_FILE"
    return 0
  fi

  now=$(date +%s)
  last=$(cat "$DRIFT_FILE" 2>/dev/null || echo 0)
  [ $(( now - ${last:-0} )) -lt "$DRIFT_QUIET" ] && return 0
  echo "$now" > "$DRIFT_FILE"
  log "경고: $msg"
}

# 프로브보다 먼저 샘플을 남긴다. 프로브가 걸린 채 머신이 죽어도 직전 상태는 남는다.
sample_pressure

# 유닛이 안 돌면 워치독이 할 일이 없다 (크래시는 Restart=always 담당).
if ! systemctl --user is-active --quiet "$UNIT"; then
  log "유닛 비활성 - 건너뜀"
  reset_fails
  exit 0
fi

check_context_drift

# 모델 로딩 중(503)은 정상 상태다. 로딩에 30~60초가 걸린다.
health=$(curl -s -m 5 "$ENDPOINT/health" 2>/dev/null || true)
if echo "$health" | grep -q "Loading model"; then
  log "모델 로딩 중 - 카운터 초기화"
  reset_fails
  exit 0
fi

# 핵심 프로브: /slots 가 시간 내에 유효한 JSON 배열을 반환하는가.
if curl -s -m "$PROBE_TIMEOUT" "$ENDPOINT/slots" 2>/dev/null \
     | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if isinstance(d,list) else 1)' 2>/dev/null; then
  prev=$(read_fails)
  [ "$prev" -gt 0 ] && log "정상 복구 (직전 연속 실패 ${prev}회)"
  reset_fails
  exit 0
fi

fails=$(( $(read_fails) + 1 ))
echo "$fails" > "$FAIL_FILE"
log "프로브 실패 ${fails}/${FAIL_THRESHOLD} (/slots 가 ${PROBE_TIMEOUT}초 내 무응답)"

[ "$fails" -lt "$FAIL_THRESHOLD" ] && exit 0

# 임계 도달. 재시작 전에 "정말 멈췄는지" 교차 확인한다.
# 아주 긴 프롬프트 평가 중이면 /slots 는 느려도 저널에는 진행 로그가 찍힌다.
# 그런 경우까지 재시작하면 멀쩡한 작업을 죽이게 된다.
progress=$(journalctl --user -u "$UNIT" --since "-${PROGRESS_WINDOW}min" --no-pager -o cat 2>/dev/null \
           | grep -cE "release: id|print_timing|prompt processing" || true)
if [ "${progress:-0}" -gt 0 ]; then
  log "프로브는 실패했으나 최근 ${PROGRESS_WINDOW}분간 진행 로그 ${progress}건 - 작업 중으로 판단, 재시작 보류"
  reset_fails
  exit 0
fi

now=$(date +%s)

# 쿨다운: 방금 재시작했다면 다시 건드리지 않는다.
if [ -f "$HIST_FILE" ]; then
  last=$(tail -1 "$HIST_FILE" 2>/dev/null || echo 0)
  if [ -n "$last" ] && [ $(( now - last )) -lt "$COOLDOWN" ]; then
    log "쿨다운 중 ($(( now - last ))초 경과 / ${COOLDOWN}초) - 재시작 보류"
    exit 0
  fi
fi

# 플랩 방지: 한 시간에 MAX_PER_HOUR 회를 넘기면 사람이 봐야 할 문제다.
recent=$(awk -v c=$(( now - 3600 )) '$1 > c' "$HIST_FILE" 2>/dev/null | wc -l)
if [ "$recent" -ge "$MAX_PER_HOUR" ]; then
  log "경고: 최근 1시간 재시작 ${recent}회 (상한 ${MAX_PER_HOUR}) - 자동 재시작 중단. 수동 확인 필요."
  exit 1
fi

log "스톨 확정 (연속 ${fails}회 무응답 + 진행 로그 없음) - ${UNIT} 재시작"
# 재시작하면 cgroup 카운터가 0 으로 리셋되므로 직전 샘플을 저널에 박아둔다.
while read -r l; do log "직전 샘플: $l"; done < <(grep -v '^#' "$PRESSURE_LOG" 2>/dev/null | tail -3)
echo "$now" >> "$HIST_FILE"
if systemctl --user restart "$UNIT"; then
  log "재시작 완료"
else
  log "재시작 실패 (exit $?)"
fi
reset_fails
exit 0

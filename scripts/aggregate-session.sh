#!/bin/bash
# 실제 Cline 세션의 llama-server 저널을 집계한다.
#
#   ./scripts/aggregate-session.sh ["YYYY-MM-DD HH:MM:SS"]
#
# 인자를 주면 그 시각 이후, 없으면 최근 1시간을 집계한다. 세션 시작 직전에
# `date '+%F %T'`로 시각을 적어두고 그 값을 넘기면 해당 세션만 잘라 볼 수 있다.
#
# 출력: 프롬프트 처리/생성 합계와 속도, 벽시계 중 프롬프트 비중, 접두부 캐시
# 적중률, 대형 재처리 목록, 최대 도달 컨텍스트. README "튜닝 전후 실사용 비교"
# 표가 이 스크립트의 출력이다.
set -u
SINCE=${1:--1 hour}
LOG="$(mktemp -t llama-session-XXXX.log)"
journalctl --user -u llama-server --since "$SINCE" --no-pager > "$LOG" 2>/dev/null

echo "=== 집계 구간: $SINCE ~ 현재 ==="
awk '
/prompt eval time =/ {
  if (match($0,/= +[0-9.]+ ms \/ +[0-9]+ tokens/)) {
    s=substr($0,RSTART,RLENGTH); split(s,a," ");
    pms+=a[2]; ptok+=a[5]; pn++;
    if (a[5]+0>maxp) { maxp=a[5]+0; maxpms=a[2]+0 }
  } }
/\| *eval time =/ && !/prompt eval/ {
  if (match($0,/= +[0-9.]+ ms \/ +[0-9]+ tokens/)) {
    s=substr($0,RSTART,RLENGTH); split(s,a," "); ems+=a[2]; etok+=a[5]; en++
  } }
END {
  if (pn==0) { print "이 구간에 요청 없음"; exit }
  printf "프롬프트 처리: %d회, %d토큰, %.1f초 (%.0f t/s)\n", pn, ptok, pms/1000, ptok/(pms/1000)
  printf "생성        : %d회, %d토큰, %.1f초 (%.1f tok/s)\n", en, etok, ems/1000, etok/(ems/1000)
  printf "벽시계 합계 : %.1f초  (프롬프트 %.0f%%)\n", (pms+ems)/1000, pms*100/(pms+ems)
  printf "최대 단일 재처리: %d토큰 / %.1f초\n", maxp, maxpms/1000
}' "$LOG"

echo
echo "--- 슬롯 선택 (캐시 적중 여부)"
echo "LCP 적중: $(grep -c 'selected slot by LCP' "$LOG")회 | LRU 폴백: $(grep -c 'selected slot by LRU' "$LOG")회 (그 중 최초 콜드 $(grep -o 't_last = -1' "$LOG" | wc -l)회)"

echo
echo "--- 대형 재처리(3000토큰 초과)"
grep -oE "prompt eval time = +[0-9.]+ ms / +[0-9]+ tokens" "$LOG" \
  | awk '{ if ($8>3000) printf "  %s토큰 / %.1f초\n", $8, $5/1000 }' | sort -rn

echo
echo "--- 컨텍스트별 생성 속도"
paste <(grep -oE "\| *eval time = [^|]*tokens per second" "$LOG" | grep -v prompt \
          | grep -oE "[0-9.]+ tokens per second" | awk '{print $1}') \
      <(grep -oE "n_tokens = [0-9]+" "$LOG" | awk '{print $3}') 2>/dev/null \
  | awk 'NF==2 { printf "  %s tok/s @ %s\n", $1, $2 }'

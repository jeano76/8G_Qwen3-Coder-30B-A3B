#!/bin/bash
# GPU/CPU/메모리 모니터링 대시보드 실행기.
# 이미 떠 있으면 새로 안 띄우고 브라우저만 연다 (아이콘 여러 번 클릭해도 안전).
set -u
PORT=5757
DIR="/home/jeano/projects/8G_Qwen3-Coder-30B-A3B/scripts/sysmon"
LOG="/tmp/sysmon-dashboard.log"

is_running() {
  curl -s -o /dev/null -m 1 "http://127.0.0.1:${PORT}/api/stats"
}

if ! is_running; then
  nohup python3 "${DIR}/server.py" >"${LOG}" 2>&1 &
  disown
  for _ in $(seq 1 30); do
    is_running && break
    sleep 0.2
  done
fi

xdg-open "http://127.0.0.1:${PORT}/" >/dev/null 2>&1 &

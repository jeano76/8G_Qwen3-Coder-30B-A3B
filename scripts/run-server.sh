#!/bin/bash
# MTP(멀티 토큰 예측) 설정. --spec-type draft-mtp 에는 07822bd 이후 빌드가 필요하다.
# 빌드 갱신 자체는 속도에 영향이 없었다(README "투기적 디코딩" 절 A/B 표).
#
# 경로의 -new 는 임시 이름이 아니다. 이 바이너리는 17KB 런처이고
# libllama-server-impl.so 를 절대 경로 RPATH 로 찾는다. 디렉터리를 rename 하면
# 'error while loading shared libraries' 로 즉시 죽는다. 옮기려면 재빌드해야 한다.
#
# ngram-mod 제거 (2026-08-28): 반복 요청(=실사용 세션 패턴)에서 2번째 요청부터
# tg가 44.4 -> 36.9 tok/s로 계속 저하됨을 확인. 단일 연속 생성 안에서만 검증됐던
# 기존 판정(README)과 달리, 세션당 다수 요청이 오가는 실사용에서는 손해였다.
# README "정정 (2026-08-28)" 절 참고.
exec ~/llama.cpp/build-cuda-new/bin/llama-server \
  -m ~/models/Qwen3.6-35B-A3B-MTP-UD-Q3_K_XL.gguf \
  -ngl 99 -ncmoe 35 -fa on -t 6 -lm none -np 1 -kvu -ub 2048 \
  -c 65536 -ctk q8_0 -ctv q8_0 \
  -sps 0.5 \
  -rea off \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --host 127.0.0.1 --port 8080

#!/bin/bash
# MTP(멀티 토큰 예측) 빌드. --spec-type draft-mtp 는 07822bd 이후 빌드가 필요해
# build-cuda-new 를 쓴다. 빌드 자체는 속도에 영향이 없었다(README A/B 표).
exec ~/llama.cpp/build-cuda-new/bin/llama-server \
  -m ~/models/Qwen3.6-35B-A3B-MTP-UD-Q3_K_XL.gguf \
  -ngl 99 -ncmoe 35 -fa on -t 6 -lm none -np 2 -kvu -ub 2048 \
  -c 65536 -ctk q8_0 -ctv q8_0 \
  -sps 0.5 \
  -rea off \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --host 127.0.0.1 --port 8080

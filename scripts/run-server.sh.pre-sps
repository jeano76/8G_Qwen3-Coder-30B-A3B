#!/bin/bash
exec ~/llama.cpp/build-cuda/bin/llama-server \
  -m ~/models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf \
  -ngl 99 -ncmoe 33 -fa on -t 6 -lm none -np 2 -kvu -ub 2048 \
  -c 131072 -ctk q8_0 -ctv q8_0 \
  -rea off \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --host 127.0.0.1 --port 8080

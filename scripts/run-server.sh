#!/bin/bash
exec ~/llama.cpp/build-cuda/bin/llama-server \
  -m ~/models/Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf \
  -ngl 99 -ncmoe 40 -fa on -t 6 -lm none -np 2 -kvu -ub 2048 \
  -c 65536 -ctk q8_0 -ctv q8_0 \
  --temp 0.7 --top-p 0.8 --top-k 20 --repeat-penalty 1.05 --min-p 0.0 \
  --host 127.0.0.1 --port 8080

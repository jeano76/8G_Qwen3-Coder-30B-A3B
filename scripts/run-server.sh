#!/bin/bash
exec ~/llama.cpp/build-cuda/bin/llama-server \
  -m ~/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  -ngl 99 -ncmoe 34 -fa on -t 6 -lm none \
  -c 24576 -ctk q8_0 -ctv q8_0 \
  --host 127.0.0.1 --port 8080

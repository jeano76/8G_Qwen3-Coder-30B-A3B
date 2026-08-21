#!/bin/bash
exec ~/llama.cpp/build-cuda/bin/llama-server \
  -m ~/models/Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf \
  -ngl 99 -ncmoe 32 -fa on -t 6 -lm none \
  -c 36864 -ctk q8_0 -ctv q8_0 \
  --host 127.0.0.1 --port 8080

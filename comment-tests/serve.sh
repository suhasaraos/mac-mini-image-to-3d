#!/bin/zsh
# OpenAI-compatible server at http://127.0.0.1:8080/v1 (model name: see `curl :8080/v1/models`).
# Thinking off + Qwen's recommended non-thinking sampling. Context up to ~64K+ fits in memory.
cd "${0:A:h}"
exec .venv/bin/mlx_lm.server --model models/qwen3.8-27b-dwq3.5 \
  --temp 0.7 --top-p 0.8 --top-k 20 --max-tokens 4096 \
  --chat-template-args '{"enable_thinking": false}' "$@"

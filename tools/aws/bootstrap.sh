#!/usr/bin/env bash
# One-time setup ON THE INSTANCE (run after the first push.sh):
#   ssh cse291a 'bash ~/cse291/agentserve/tools/aws/bootstrap.sh'
# Builds the same llama.cpp commit as the Mac (with CUDA), downloads and verifies the same
# model, creates the Python venv, and installs the idle auto-stop. Safe to re-run.
# Layout matches the Mac, so the configs' relative paths work unchanged:
#   ~/cse291/{agentserve, llama.cpp, models, .venv}
set -euo pipefail
ROOT="$HOME/cse291"
LLAMA_COMMIT=a37cc46104eb50900d5ab4a9ba02a0abdac039db     # build b11803, same as Phase 1 on the Mac
MODEL=Llama-3.2-3B-Instruct-Q4_K_M.gguf
MODEL_SHA256=6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff
export PATH="/usr/local/cuda/bin:$PATH"

echo "== GPU"; nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
CUDA_ARCH="${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)}"

echo "== packages"
if command -v dnf >/dev/null; then          # Amazon Linux 2023 (the course's Deep Learning AMIs)
  sudo dnf install -y -q gcc gcc-c++ make cmake git jq tmux rsync cronie sysstat iproute procps-ng python3.11 >/dev/null
  sudo systemctl enable --now crond
  PY=python3.11
else                                        # Ubuntu
  sudo apt-get update -qq
  sudo apt-get install -y -qq build-essential cmake git jq tmux rsync cron sysstat python3-venv >/dev/null
  PY=python3
fi

echo "== llama.cpp @ ${LLAMA_COMMIT:0:9} (CUDA arch $CUDA_ARCH)"
[ -d "$ROOT/llama.cpp/.git" ] || git clone -q https://github.com/ggml-org/llama.cpp "$ROOT/llama.cpp"
git -C "$ROOT/llama.cpp" fetch -q --tags origin
git -C "$ROOT/llama.cpp" checkout -q "$LLAMA_COMMIT"
cmake -S "$ROOT/llama.cpp" -B "$ROOT/llama.cpp/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" >/dev/null
cmake --build "$ROOT/llama.cpp/build" --config Release -j "$(nproc)" --target llama-server llama-bench llama-batched-bench >/dev/null
git -C "$ROOT/llama.cpp" rev-parse HEAD > "$ROOT/llama_commit.txt"
"$ROOT/llama.cpp/build/bin/llama-server" --version 2>&1 | head -2

echo "== python venv"
[ -d "$ROOT/.venv" ] || "$PY" -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/pip" install -q -U pip "huggingface_hub[cli]" -r "$ROOT/agentserve/requirements.txt"
"$ROOT/.venv/bin/python" --version

echo "== model"
mkdir -p "$ROOT/models"
if ! echo "$MODEL_SHA256  $ROOT/models/$MODEL" | sha256sum -c --quiet 2>/dev/null; then
  "$ROOT/.venv/bin/hf" download bartowski/Llama-3.2-3B-Instruct-GGUF "$MODEL" --local-dir "$ROOT/models"
  echo "$MODEL_SHA256  $ROOT/models/$MODEL" | sha256sum -c
fi
sha256sum "$ROOT/models/$MODEL" > "$ROOT/model_sha256.txt"

echo "== idle auto-stop (every 5 min; stops after 30 idle min)"
sudo install -m 755 "$ROOT/agentserve/tools/aws/idle_shutdown.sh" /usr/local/bin/cse291a-idle-shutdown
echo "*/5 * * * * root IDLE_MIN=30 /usr/local/bin/cse291a-idle-shutdown" | sudo tee /etc/cron.d/cse291a-idle >/dev/null
sudo chmod 644 /etc/cron.d/cse291a-idle
echo "!! the instance's Shutdown behavior must be Stop (Terminate would DELETE it when idle)"

grep -q 'cse291/.venv' ~/.bashrc || cat >> ~/.bashrc <<'EOF'
# cse291a
export PATH="/usr/local/cuda/bin:$PATH"
source ~/cse291/.venv/bin/activate
cd ~/cse291/agentserve
EOF
echo "== done. Next: python -m agentserve.sweep configs/sweep_mock.json && python -m agentserve.analyze runs/mock"

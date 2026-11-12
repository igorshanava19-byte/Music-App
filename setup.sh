#!/usr/bin/env bash
# One-time installer for macOS / Linux.
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
uv venv --python 3.12 --allow-existing .venv
if [[ "$(uname)" == "Darwin" ]]; then
  uv pip install --python .venv torch==2.14.1
elif command -v nvidia-smi >/dev/null; then
  uv pip install --python .venv torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
else
  uv pip install --python .venv torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
fi
uv pip install --python .venv -r requirements.txt
uv pip install --python .venv --no-deps basic-pitch==0.4.0
.venv/bin/python -m app.check
echo "Setup finished. Start the app with ./run.sh"

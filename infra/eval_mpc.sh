#!/bin/bash
# Evaluate the LeWM MPC controller on 10 duckietown episodes.
#
# Usage:
#   bash infra/eval_mpc.sh [--ckpt PATH] [--steps N] [--map NAME] [--video out.mp4]
#
# Designed to run on EC2 Ubuntu 22.04 (Python 3.10) after the standard bootstrap,
# or locally if duckietown and stable-worldmodel are already installed.
# Uses uv for env management; installs packages if the venv is missing.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CKPT="data/lewm_best.pt"
STEPS=300
MAP_ARG=""
VIDEO_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)  CKPT=$2;              shift 2 ;;
        --steps) STEPS=$2;             shift 2 ;;
        --map)   MAP_ARG="--map $2";   shift 2 ;;
        --video) VIDEO_ARG="--video $2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Python env ────────────────────────────────────────────────────────────────
VENV=".venv-mpc"

if [[ ! -d "$VENV" ]]; then
    echo "==> Creating Python 3.10 venv with uv ..."
    if ! command -v uv &>/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    fi
    uv venv "$VENV" --python 3.10

    echo "==> Installing packages ..."
    "$VENV/bin/pip" install --quiet --no-cache-dir \
        "numpy<2.0.0" "pyglet==1.5.27" \
        boto3 h5py einops pillow "imageio[ffmpeg]" \
        "stable-worldmodel[train]" duckietown-gym-daffy

    # Force CPU-only torch+torchvision after stable-worldmodel (it pulls CUDA build)
    "$VENV/bin/pip" install --quiet --no-cache-dir --force-reinstall \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu

    # Re-pin numpy<2 (stable-worldmodel upgrades it)
    "$VENV/bin/pip" install --quiet --no-cache-dir "numpy<2.0.0"

    echo "==> Packages installed."
fi

PYTHON="$VENV/bin/python3"

# Patch pwm_dynamics if needed
"$PYTHON" -c "
import pathlib
try:
    import duckietown_world
    p = pathlib.Path(duckietown_world.__file__).parent / 'world_duckietown/pwm_dynamics.py'
    t = p.read_text()
    old = '        linear = [longitudinal, lateral]'
    new = '        linear = [float(longitudinal), lateral]'
    if old in t:
        p.write_text(t.replace(old, new))
        print('pwm_dynamics patched')
except Exception as e:
    print(f'patch skip: {e}')
" 2>/dev/null || true

# ── Xvfb ──────────────────────────────────────────────────────────────────────
if ! pgrep -x Xvfb >/dev/null 2>&1; then
    echo "==> Starting Xvfb :99 ..."
    Xvfb :99 -screen 0 1024x768x24 &
    sleep 2
fi
export DISPLAY=:99

# ── Duckietown map symlink ────────────────────────────────────────────────────
DT_WORLD=$("$PYTHON" -c \
    "import duckietown_world,os;print(os.path.dirname(duckietown_world.__file__))" \
    2>/dev/null | tail -1)
DT_GYM=$("$PYTHON" -c \
    "import gym_duckietown,os;print(os.path.dirname(gym_duckietown.__file__))" \
    2>/dev/null | tail -1)
ln -sf "${DT_WORLD}/data/gd1/maps" "${DT_GYM}/maps" 2>/dev/null || true

# ── Run 10 episodes ───────────────────────────────────────────────────────────
echo ""
echo "==> Running 10 episodes (ckpt=${CKPT}  steps=${STEPS}) ..."
echo ""

"$PYTHON" src/mpc_controller.py \
    --ckpt   "$CKPT"  \
    --steps  "$STEPS" \
    --episodes 10     \
    $MAP_ARG          \
    $VIDEO_ARG        \
    --verbose

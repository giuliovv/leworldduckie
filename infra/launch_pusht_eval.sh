#!/bin/bash
# Launch an EC2 instance to run the Push-T evaluation + diagnostics.
#
# Steps on the instance:
#   1. Install le-wm + stable-worldmodel[train,env]
#   2. Download Push-T dataset (if needed) + checkpoint from S3
#   3. python eval.py  → CEM-based MPC on swm/PushT-v1 (50 envs, 100 episodes)
#   4. python pusht_diagnostics.py → T5/T6/sensitivity
#   5. Upload all results to s3://leworldduckie/evals/pusht/<run_id>/
#
# Results:
#   results.txt        — eval.py success rate
#   diagnostics.txt    — T5/T6/sensitivity
#   stdout.txt         — full log
#   instance.log       — system boot log
#
# Usage:
#   bash infra/launch_pusht_eval.sh [--ckpt s3://.../lewm_object.ckpt]
#                                    [--data s3://.../pusht_expert_train.h5]
#                                    [--n-eval-episodes 100]
#                                    [--subnet subnet-xxxxxxxx]
#                                    [--instance-type c7i.xlarge]
#                                    [--on-demand]

set -euo pipefail

REGION=us-east-1
AMI_ID=ami-09d0a18beb02cc7d4   # Deep Learning OSS Nvidia PyTorch 2.7 Ubuntu 22.04
INSTANCE_TYPE=g4dn.xlarge
INSTANCE_PROFILE=lewm-ec2-training
SECURITY_GROUP=sg-03bbca875466eb52a
SUBNET=subnet-0b799a4832af70f5b
S3_BUCKET=leworldduckie

RUN_ID=$(date -u +%Y%m%d_%H%M%S)
CKPT="s3://${S3_BUCKET}/training/pusht/pusht/lewm_object.ckpt"
DATA="s3://${S3_BUCKET}/data/pusht_expert_train.h5"
N_EVAL=100
KEEP_ON_FAIL=0
MARKET_MODE=spot

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)           CKPT=$2;   shift 2 ;;
        --data)           DATA=$2;   shift 2 ;;
        --n-eval-episodes) N_EVAL=$2; shift 2 ;;
        --subnet)         SUBNET=$2; shift 2 ;;
        --instance-type)  INSTANCE_TYPE=$2; shift 2 ;;
        --on-demand)      MARKET_MODE=ondemand; shift ;;
        --keep-on-fail)   KEEP_ON_FAIL=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "==> Uploading pusht_diagnostics.py to S3 ..."
aws s3 cp "$(dirname "$0")/../src/pusht_diagnostics.py" \
    "s3://${S3_BUCKET}/evals/pusht_diagnostics.py" --region "$REGION"

USER_DATA=$(cat <<USERDATA
#!/bin/bash
LOG=/var/log/pusht-eval.log
exec >>"\$LOG" 2>&1
set -Eeuo pipefail
set -x
echo "=== Push-T eval bootstrap \$(date -u) run=${RUN_ID} ==="
export HOME=/root DEBIAN_FRONTEND=noninteractive MUJOCO_GL=egl

# Live log upload every 30s
(while true; do
    aws s3 cp "\$LOG" "s3://${S3_BUCKET}/evals/pusht/${RUN_ID}/live.log" --quiet 2>/dev/null || true
    sleep 30
done) &
LOG_SYNC_PID=\$!

finish() {
    EXIT_CODE=\$?
    set +e
    echo "=== finish trap: exit_code=\${EXIT_CODE} at \$(date -u) ==="
    [ -n "\${LOG_SYNC_PID:-}" ] && kill "\$LOG_SYNC_PID" 2>/dev/null || true
    aws s3 cp "\$LOG" "s3://${S3_BUCKET}/evals/pusht/${RUN_ID}/live.log" --quiet 2>/dev/null || true
    python3 -c "
import boto3, os
s3 = boto3.client('s3', region_name='us-east-1')
for src, key in [
    ('/var/log/pusht-eval.log', 'evals/pusht/${RUN_ID}/instance.log'),
    ('/tmp/eval_stdout.txt',    'evals/pusht/${RUN_ID}/eval_stdout.txt'),
    ('/tmp/diag_stdout.txt',    'evals/pusht/${RUN_ID}/diag_stdout.txt'),
    ('/root/.stable-wm/pusht_results.txt', 'evals/pusht/${RUN_ID}/results.txt'),
    ('/tmp/pusht_diagnostics.txt', 'evals/pusht/${RUN_ID}/diagnostics.txt'),
]:
    if os.path.exists(src):
        try:
            s3.upload_file(src, '${S3_BUCKET}', key)
            print(f'uploaded {key}')
        except Exception as e:
            print(f'upload failed {key}: {e}')
" || true
    echo "exit_code=\${EXIT_CODE}" > /tmp/exit_code.txt
    aws s3 cp /tmp/exit_code.txt "s3://${S3_BUCKET}/evals/pusht/${RUN_ID}/exit_code.txt" --quiet 2>/dev/null || true
    if [ "\${EXIT_CODE}" -eq 0 ]; then
        echo "=== success, shutting down ==="
        shutdown -h now
    else
        if [ "${KEEP_ON_FAIL}" -eq 1 ]; then
            echo "=== failure, keeping instance alive for debugging (--keep-on-fail) ==="
            sleep infinity
        else
            echo "=== failure, shutting down (logs uploaded) ==="
            shutdown -h now
        fi
    fi
}
trap finish EXIT

export PATH="\$PATH:/usr/local/cuda/bin"

# Install Python deps
apt-get update -y
while pgrep -x unattended-upgr >/dev/null 2>&1; do
    echo "waiting for unattended-upgr to release dpkg lock..."
    sleep 10
done
for i in 1 2 3 4 5; do
    if apt-get install -y swig python3-boto3; then
        break
    fi
    echo "apt install failed (attempt \${i}), retrying in 10s..."
    sleep 10
done
pip3 install -q "pip<25.0" "setuptools<66" wheel && echo "packaging pins ok"
pip3 install -q zstandard huggingface_hub && echo "core deps ok"
pip3 install -q "numpy<2.0.0" && echo "numpy pin ok"
pip3 install -q "stable-worldmodel[train,env]" einops pillow scikit-learn zstandard huggingface_hub && echo "stable-worldmodel ok"

# Clone le-wm
git clone --depth 1 https://github.com/lucas-maes/le-wm.git /tmp/le-wm && echo "le-wm cloned"

# CPU fallback: eval.py in upstream le-wm hardcodes model.to("cuda")
python3 - <<'PY'
from pathlib import Path
p = Path('/tmp/le-wm/eval.py')
t = p.read_text()
old = '    model = model.to("cuda")'
new = '    model = model.to("cuda" if torch.cuda.is_available() else "cpu")'
if old in t:
    p.write_text(t.replace(old, new, 1))
    print('patched eval.py for cpu fallback')
else:
    print('eval.py cuda line not found (already patched?)')
PY

# Download checkpoint
python3 -c "
import boto3, sys
from urllib.parse import urlparse
u = urlparse('${CKPT}')
s3 = boto3.client('s3', region_name='${REGION}')
local = '/tmp/lewm_object.ckpt'
print(f'Downloading ${CKPT} ...')
s3.download_file(u.netloc, u.path.lstrip('/'), local)
print('checkpoint ok')
"

# Download Push-T data: try S3 first, fall back to HuggingFace
python3 -c "
import boto3, os, sys, zstandard
from pathlib import Path
from urllib.parse import urlparse

local = Path('/tmp/pusht_expert_train.h5')
zst   = Path('/tmp/pusht_expert_train.h5.zst')

if local.exists():
    print(f'data already cached: {local}')
    sys.exit(0)

# Try S3 first
data_uri = '${DATA}'
if data_uri.startswith('s3://'):
    try:
        u = urlparse(data_uri)
        s3 = boto3.client('s3', region_name='${REGION}')
        print(f'Downloading from S3: {data_uri}')
        s3.download_file(u.netloc, u.path.lstrip('/'), str(local))
        print('data ok (S3)')
        sys.exit(0)
    except Exception as e:
        print(f'S3 download failed ({e}), falling back to HuggingFace')

# Fall back to HuggingFace (quentinll/lewm-pusht)
from huggingface_hub import hf_hub_download
print('Downloading pusht_expert_train.h5.zst from HuggingFace (~13 GB)...')
dl = hf_hub_download(
    repo_id='quentinll/lewm-pusht',
    filename='pusht_expert_train.h5.zst',
    repo_type='dataset',
    local_dir='/tmp',
)
import shutil
dl_p = Path(dl)
if dl_p.resolve() != zst.resolve():
    shutil.copy2(dl_p, zst)

print('Decompressing...')
dctx = zstandard.ZstdDecompressor()
with open(zst, 'rb') as fin, open(local, 'wb') as fout:
    dctx.copy_stream(fin, fout)
zst.unlink()
print(f'data ok (HuggingFace): {local.stat().st_size / 1e9:.1f} GB')
"

# Download diagnostics script
aws s3 cp "s3://${S3_BUCKET}/evals/pusht_diagnostics.py" /tmp/pusht_diagnostics.py

# ── Run eval.py (CEM MPC on PushT-v1) ────────────────────────────────────────
mkdir -p /root/.stable-wm/pusht
cp /tmp/lewm_object.ckpt /root/.stable-wm/pusht/lewm_object.ckpt
cp /tmp/pusht_expert_train.h5 /root/.stable-wm/pusht_expert_train.h5
export STABLEWM_HOME=/root/.stable-wm

cd /tmp/le-wm
EVAL_DEVICE=\$(python3 - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
)
echo "eval device: \${EVAL_DEVICE}"
python3 eval.py \
    --config-name=pusht.yaml \
    policy=pusht/lewm \
    eval.num_eval=${N_EVAL} \
    solver.device=\${EVAL_DEVICE} \
    output.filename=pusht_results.txt \
    2>&1 | tee /tmp/eval_stdout.txt
EVAL_EXIT=\${PIPESTATUS[0]}
echo "eval exit: \${EVAL_EXIT}"

# ── Run diagnostics ──────────────────────────────────────────────────────────
cd /tmp
mkdir -p /usr/local/lib/plugin
export HDF5_PLUGIN_PATH=/usr/local/lib/plugin
python3 pusht_diagnostics.py \
    --ckpt /tmp/lewm_object.ckpt \
    --data /tmp/pusht_expert_train.h5 \
    --n-samples 200 \
    --out /tmp/pusht_diagnostics.txt \
    2>&1 | tee /tmp/diag_stdout.txt
DIAG_EXIT=\${PIPESTATUS[0]}
echo "diagnostics exit: \${DIAG_EXIT}"

echo "=== done ==="
USERDATA
)

echo "==> Launching ${MARKET_MODE} ${INSTANCE_TYPE} for Push-T eval (run_id=${RUN_ID}) ..."

COMMON_ARGS=(
    --region "$REGION"
    --image-id "$AMI_ID"
    --instance-type "$INSTANCE_TYPE"
    --iam-instance-profile Name="$INSTANCE_PROFILE"
    --security-group-ids "$SECURITY_GROUP"
    --subnet-id "$SUBNET"
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":150,"VolumeType":"gp3","DeleteOnTermination":true}}]'
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=lewm-pusht-eval-${RUN_ID}},{Key=Project,Value=leworldduckie}]"
    --user-data "$USER_DATA"
    --query 'Instances[0].InstanceId'
    --output text
)

if [[ "$MARKET_MODE" == "spot" ]]; then
    INSTANCE_ID=$(aws ec2 run-instances "${COMMON_ARGS[@]}" \
        --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}')
else
    INSTANCE_ID=$(aws ec2 run-instances "${COMMON_ARGS[@]}")
fi

echo ""
echo "==> Instance  : ${INSTANCE_ID}"
echo "==> Run ID    : ${RUN_ID}"
echo "==> Checkpoint: ${CKPT}"
echo ""
echo "Monitor (bootstrap ~15 min, eval ~30 min, diag ~10 min):"
echo ""
echo "  Live log:"
echo "    aws s3 cp s3://${S3_BUCKET}/evals/pusht/${RUN_ID}/live.log -"
echo ""
echo "  Eval results:"
echo "    aws s3 cp s3://${S3_BUCKET}/evals/pusht/${RUN_ID}/results.txt -"
echo ""
echo "  Diagnostics:"
echo "    aws s3 cp s3://${S3_BUCKET}/evals/pusht/${RUN_ID}/diagnostics.txt -"
echo ""
echo "  Instance status:"
echo "    aws ec2 describe-instances --instance-ids ${INSTANCE_ID} --region ${REGION} --query 'Reservations[0].Instances[0].State.Name' --output text"

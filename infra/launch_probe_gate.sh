#!/bin/bash
# Launch spot CPU instance to run obs->action probe gate (encoder mode by default).
# Uploads live logs + final results to S3 and force-shuts down on timeout.
#
# Usage:
#   bash infra/launch_probe_gate.sh [--mode encoder|cnn] [--max-samples 50000] [--epochs 8] [--batch-size 256] [--hard-timeout-min 60]
#
# Artifacts:
#   s3://leworldduckie/evals/probe_gate/<run_id>/{live.log,probe_results.txt,instance.log,exit_code.txt,phase_*.txt,error.txt}

set -euo pipefail

REGION=us-east-1
AMI_ID=ami-05e86b3611c60b0b4
INSTANCE_TYPE=c7a.2xlarge
INSTANCE_PROFILE=lewm-ec2-training
SECURITY_GROUP=sg-03bbca875466eb52a
SUBNET=subnet-00ef452a9147da192
S3_BUCKET=leworldduckie
MARKET_MODE=spot

MODE=encoder
MAX_SAMPLES=50000
EPOCHS=8
BATCH_SIZE=256
ENCODE_BATCH_SIZE=512
HARD_TIMEOUT_MIN=60
RUN_ID=$(date -u +%Y%m%d_%H%M%S)

DATA_NEW_S3="s3://${S3_BUCKET}/data/duckie_explore.h5"
DATA_OLD_S3="s3://${S3_BUCKET}/data/duckietown_100k.h5"
CKPT_S3="s3://${S3_BUCKET}/training/runs/notebook/checkpoint_best.pt"

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) MODE=$2; shift 2 ;;
        --max-samples) MAX_SAMPLES=$2; shift 2 ;;
        --epochs) EPOCHS=$2; shift 2 ;;
        --batch-size) BATCH_SIZE=$2; shift 2 ;;
        --encode-batch-size) ENCODE_BATCH_SIZE=$2; shift 2 ;;
        --hard-timeout-min) HARD_TIMEOUT_MIN=$2; shift 2 ;;
        --instance-type) INSTANCE_TYPE=$2; shift 2 ;;
        --subnet) SUBNET=$2; shift 2 ;;
        --on-demand) MARKET_MODE=ondemand; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "==> Uploading probe script to S3 ..."
aws s3 cp "$(dirname "$0")/../src/probe_obs_to_action.py" \
    "s3://${S3_BUCKET}/evals/probe_obs_to_action.py" --region "$REGION"

USER_DATA=$(cat <<USERDATA
#!/bin/bash
set -uo pipefail
LOG=/var/log/probe-gate.log
exec >>"\$LOG" 2>&1
export HOME=/root DEBIAN_FRONTEND=noninteractive
echo "=== probe gate bootstrap \$(date -u) run=${RUN_ID} ==="

S3_PREFIX="s3://${S3_BUCKET}/evals/probe_gate/${RUN_ID}"
phase() {
  local name="\$1"
  echo "=== phase:\$name @ \$(date -u) ==="
  aws s3 cp <(echo "\$(date -u) phase=\$name") "\${S3_PREFIX}/phase_\${name}.txt" --quiet 2>/dev/null || true
}
fail() {
  local msg="\$1"
  echo "FATAL: \$msg"
  echo "\$msg" >/tmp/error.txt
  echo 1 >/tmp/exit_code.txt
  aws s3 cp /tmp/error.txt "\${S3_PREFIX}/error.txt" --quiet 2>/dev/null || true
  aws s3 cp /tmp/exit_code.txt "\${S3_PREFIX}/exit_code.txt" --quiet 2>/dev/null || true
  shutdown -h now || true
  exit 1
}

# Hard watchdog timeout (minutes): force shutdown even if hung
(
  sleep \$(( ${HARD_TIMEOUT_MIN} * 60 ))
  echo "=== watchdog timeout reached (${HARD_TIMEOUT_MIN} min), forcing shutdown ==="
  echo 124 >/tmp/exit_code.txt || true
  shutdown -h now || true
) &

# Continuous live-log upload
(
  while true; do
    aws s3 cp "\$LOG" "\${S3_PREFIX}/live.log" --quiet 2>/dev/null || true
    sleep 30
  done
) &

phase bootstrap
apt-get update -q || fail "apt-get update failed"
apt-get install -y -q python3-pip python3-venv git awscli || fail "apt-get install failed"
echo "apt complete @ \$(date -u)"
python3 -V
df -h
free -h

# Preflight resource checks
MEM_TOTAL_KB=\$(awk '/MemTotal/ {print \$2}' /proc/meminfo)
if [[ -z "\$MEM_TOTAL_KB" || "\$MEM_TOTAL_KB" -lt 7000000 ]]; then
  fail "insufficient RAM: MemTotalKB=\${MEM_TOTAL_KB:-unknown} (need >= 7000000)"
fi
ROOT_AVAIL_KB=\$(df --output=avail / | tail -1 | tr -d ' ')
if [[ -z "\$ROOT_AVAIL_KB" || "\$ROOT_AVAIL_KB" -lt 12000000 ]]; then
  fail "insufficient disk: availKB=\${ROOT_AVAIL_KB:-unknown} (need >= 12000000)"
fi

phase venv
python3 -m venv /opt/probe-venv || fail "venv creation failed"
/opt/probe-venv/bin/pip install --no-cache-dir -U pip || fail "pip upgrade failed"
/opt/probe-venv/bin/pip install --no-cache-dir boto3 h5py numpy scikit-learn || fail "base pip deps failed"
/opt/probe-venv/bin/pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu || fail "cpu torch install failed"
/opt/probe-venv/bin/pip install --no-cache-dir "stable-worldmodel[train]" || fail "stable-worldmodel install failed"
echo "venv deps complete @ \$(date -u)"

python3 - <<'PY'
import boto3
boto3.client('s3', region_name='us-east-1').put_object(
    Bucket='${S3_BUCKET}',
    Key='evals/probe_gate/${RUN_ID}/started.txt',
    Body=b'bootstrap_started'
)
print('started marker uploaded')
PY

phase sources
mkdir -p /tmp/leworldduckie
cd /tmp/leworldduckie
aws s3 cp "s3://${S3_BUCKET}/evals/probe_obs_to_action.py" ./probe_obs_to_action.py --region ${REGION}
git clone --depth 1 https://github.com/lucas-maes/le-wm.git /tmp/lewm-src || fail "clone le-wm failed"
LEWM_DIR="/tmp/lewm-src"
if [[ -f /tmp/lewm-src/le-wm/jepa.py ]]; then LEWM_DIR="/tmp/lewm-src/le-wm"; fi
echo "sources ready @ \$(date -u)"

# Progress heartbeat
(
  while true; do
    echo "heartbeat: \$(date -u) still running"
    free -h | sed 's/^/mem: /'
    sleep 60
  done
) &

phase probe_start
/opt/probe-venv/bin/python -u ./probe_obs_to_action.py \
  --mode ${MODE} \
  --ckpt ${CKPT_S3} \
  --data ${DATA_NEW_S3} \
  --baseline-data ${DATA_OLD_S3} \
  --lewm-dir "\${LEWM_DIR}" \
  --max-samples ${MAX_SAMPLES} \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --encode-batch-size ${ENCODE_BATCH_SIZE} \
  2>&1 | tee /tmp/probe_results.txt
EXIT_CODE=\${PIPESTATUS[0]}
echo "\${EXIT_CODE}" >/tmp/exit_code.txt
echo "probe exit code: \${EXIT_CODE}"
phase probe_end

python3 - <<'PY'
import boto3
from pathlib import Path
s3 = boto3.client('s3', region_name='us-east-1')
bucket = '${S3_BUCKET}'
prefix = 'evals/probe_gate/${RUN_ID}/'
for src, key in [
    ('/tmp/probe_results.txt', prefix + 'probe_results.txt'),
    ('/var/log/probe-gate.log', prefix + 'instance.log'),
    ('/tmp/exit_code.txt', prefix + 'exit_code.txt'),
    ('/tmp/error.txt', prefix + 'error.txt'),
]:
    p = Path(src)
    if p.exists():
        s3.upload_file(str(p), bucket, key)
        print('uploaded', key)
PY

echo "=== done @ \$(date -u); shutting down ==="
shutdown -h now
USERDATA
)

echo "==> Launching ${MARKET_MODE} ${INSTANCE_TYPE} (run_id=${RUN_ID}) ..."

COMMON_ARGS=(
    --region "$REGION"
    --image-id "$AMI_ID"
    --instance-type "$INSTANCE_TYPE"
    --iam-instance-profile Name="$INSTANCE_PROFILE"
    --security-group-ids "$SECURITY_GROUP"
    --subnet-id "$SUBNET"
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":40,"VolumeType":"gp3","DeleteOnTermination":true}}]'
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=lewm-probe-${RUN_ID}},{Key=Project,Value=leworldduckie}]"
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
echo "==> Instance : ${INSTANCE_ID}"
echo "==> Run ID   : ${RUN_ID}"
echo ""
echo "Monitor:"
echo "    aws s3 cp s3://${S3_BUCKET}/evals/probe_gate/${RUN_ID}/live.log -"
echo "    aws s3 cp s3://${S3_BUCKET}/evals/probe_gate/${RUN_ID}/probe_results.txt -"
echo "    aws s3 cp s3://${S3_BUCKET}/evals/probe_gate/${RUN_ID}/exit_code.txt -"
echo "    aws ec2 describe-instances --instance-ids ${INSTANCE_ID} --region ${REGION} --query 'Reservations[0].Instances[0].State.Name' --output text"

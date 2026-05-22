#!/bin/bash
# Launch spot CPU instance to run obs->action probe gate (encoder mode by default).
# Uploads live logs + final results to S3 and force-shuts down on timeout.
#
# Usage:
#   bash infra/launch_probe_gate.sh [--mode encoder|cnn] [--max-samples 50000] [--epochs 8] [--batch-size 256] [--hard-timeout-min 60]
#
# Artifacts:
#   s3://leworldduckie/evals/probe_gate/<run_id>/{live.log,probe_results.txt,instance.log,exit_code.txt}

set -euo pipefail

REGION=us-east-1
AMI_ID=ami-05e86b3611c60b0b4
INSTANCE_TYPE=c7i.2xlarge
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
set -euo pipefail
LOG=/var/log/probe-gate.log
exec >>"\$LOG" 2>&1
export HOME=/root DEBIAN_FRONTEND=noninteractive
echo "=== probe gate bootstrap \$(date -u) run=${RUN_ID} ==="

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
    aws s3 cp "\$LOG" "s3://${S3_BUCKET}/evals/probe_gate/${RUN_ID}/live.log" --quiet 2>/dev/null || true
    sleep 30
  done
) &

apt-get update -q
apt-get install -y -q python3-pip git awscli
echo "apt complete @ \$(date -u)"
python3 -V

pip3 install --no-cache-dir boto3 h5py numpy scikit-learn torch torchvision
echo "pip installs complete @ \$(date -u)"

python3 - <<'PY'
import boto3
boto3.client('s3', region_name='us-east-1').put_object(
    Bucket='${S3_BUCKET}',
    Key='evals/probe_gate/${RUN_ID}/started.txt',
    Body=b'bootstrap_started'
)
print('started marker uploaded')
PY

mkdir -p /tmp/leworldduckie
cd /tmp/leworldduckie
aws s3 cp "s3://${S3_BUCKET}/evals/probe_obs_to_action.py" ./probe_obs_to_action.py --region ${REGION}
git clone --depth 1 https://github.com/lucas-maes/le-wm.git /tmp/le-wm
echo "sources ready @ \$(date -u)"

# Progress heartbeat
(
  while true; do
    echo "heartbeat: \$(date -u) still running"
    sleep 60
  done
) &

python3 ./probe_obs_to_action.py \
  --mode ${MODE} \
  --ckpt ${CKPT_S3} \
  --data ${DATA_NEW_S3} \
  --baseline-data ${DATA_OLD_S3} \
  --lewm-dir /tmp/le-wm \
  --max-samples ${MAX_SAMPLES} \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --encode-batch-size ${ENCODE_BATCH_SIZE} \
  2>&1 | tee /tmp/probe_results.txt
EXIT_CODE=\${PIPESTATUS[0]}
echo "\${EXIT_CODE}" >/tmp/exit_code.txt
echo "probe exit code: \${EXIT_CODE}"

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

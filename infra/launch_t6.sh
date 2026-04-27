#!/bin/bash
# Launch a spot t3.medium to run the proper T6 / T4 evaluation.
# Uses real encoder outputs from real Duckietown frames.
#
# T6: action discriminability L2(z_right_k, z_left_k) / noise_floor  (pass > 2×)
# T4: rollout prediction error vs horizon k=1..5
# ID: identity shortcut check ||z_{t+fs} - z_t||^2 vs best_val
#
# Results → s3://leworldduckie/evals/t6/<run_id>/results.txt
#
# Usage: bash infra/launch_t6.sh [--ckpt s3://...] [--n-samples 100]

set -euo pipefail

REGION=us-east-1
AMI_ID=ami-05e86b3611c60b0b4
INSTANCE_TYPE=t3a.medium
INSTANCE_PROFILE=lewm-ec2-training
SECURITY_GROUP=sg-03bbca875466eb52a
SUBNET=subnet-0adb09e73717bacf0
S3_BUCKET=leworldduckie

RUN_ID=$(date -u +%Y%m%d_%H%M%S)
CKPT="s3://${S3_BUCKET}/training/runs/fs6_npreds4_ep20/checkpoint_best.pt"
DATA_S3="s3://${S3_BUCKET}/data/duckietown_100k.h5"
N_SAMPLES=100
BATCH_SIZE=32
MAX_HORIZON=5
N_ROLLOUT_STEPS=3

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)             CKPT=$2;             shift 2 ;;
        --n-samples)        N_SAMPLES=$2;        shift 2 ;;
        --batch-size)       BATCH_SIZE=$2;       shift 2 ;;
        --max-horizon)      MAX_HORIZON=$2;      shift 2 ;;
        --n-rollout-steps)  N_ROLLOUT_STEPS=$2;  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "==> Uploading t6_eval.py to S3 ..."
aws s3 cp "$(dirname "$0")/../src/t6_eval.py" \
    "s3://${S3_BUCKET}/evals/t6_eval.py" --region "$REGION"

USER_DATA=$(cat <<USERDATA
#!/bin/bash
LOG=/var/log/t6-eval.log
exec >>"\$LOG" 2>&1
set -x
echo "=== T6 eval bootstrap \$(date -u) run=${RUN_ID} ==="
export HOME=/root DEBIAN_FRONTEND=noninteractive

# Live log upload every 30s
(while true; do
    aws s3 cp "\$LOG" "s3://${S3_BUCKET}/evals/t6/${RUN_ID}/live.log" --quiet 2>/dev/null || true
    sleep 30
done) &

apt-get update -q
apt-get install -y -q libgl1 libglib2.0-0 python3-pip git
echo "apt done"

pip3 install --no-cache-dir boto3                        && echo "boto3 ok"
pip3 install --no-cache-dir "numpy<2.0.0"                && echo "numpy<2 ok"
pip3 install --no-cache-dir h5py                         && echo "h5py ok"
pip3 install --no-cache-dir "stable-worldmodel[train]"   && echo "stable-worldmodel ok"
pip3 install --no-cache-dir --force-reinstall torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu     && echo "torch cpu ok"
pip3 install --no-cache-dir "numpy<2.0.0"                && echo "numpy<2 re-pin ok"
pip3 install --no-cache-dir einops pillow                && echo "misc deps ok"

# Download eval script
python3 -c "
import boto3
s3 = boto3.client('s3', region_name='${REGION}')
s3.download_file('${S3_BUCKET}', 'evals/t6_eval.py', '/tmp/t6_eval.py')
print('t6_eval.py ok')
"

cd /tmp
python3 t6_eval.py \
    --ckpt ${CKPT} \
    --data-path ${DATA_S3} \
    --n-samples ${N_SAMPLES} \
    --batch-size ${BATCH_SIZE} \
    --max-horizon ${MAX_HORIZON} \
    --n-rollout-steps ${N_ROLLOUT_STEPS} \
    --out /tmp/t6_results.txt \
    2>&1 | tee /tmp/t6_stdout.txt
EXIT_CODE=\${PIPESTATUS[0]}
echo "python exit \${EXIT_CODE}"

# Upload results
python3 -c "
import boto3, os, sys
s3 = boto3.client('s3', region_name='us-east-1')
for src, key in [
    ('/tmp/t6_results.txt',   'evals/t6/${RUN_ID}/results.txt'),
    ('/tmp/t6_stdout.txt',    'evals/t6/${RUN_ID}/stdout.txt'),
    ('/var/log/t6-eval.log',  'evals/t6/${RUN_ID}/instance.log'),
]:
    if not os.path.exists(src):
        print(f'skip {key}')
        continue
    try:
        s3.upload_file(src, '${S3_BUCKET}', key)
        print(f'uploaded {key}')
    except Exception as e:
        print(f'upload failed {key}: {e}', file=sys.stderr)
" || true

echo "=== done, shutting down ==="
shutdown -h now
USERDATA
)

echo "==> Launching spot t3.medium (run_id=${RUN_ID}) ..."

INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --iam-instance-profile Name="$INSTANCE_PROFILE" \
    --security-group-ids "$SECURITY_GROUP" \
    --subnet-id "$SUBNET" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":30,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=lewm-t6-eval-${RUN_ID}},{Key=Project,Value=leworldduckie}]" \
    --user-data "$USER_DATA" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo ""
echo "==> Instance : ${INSTANCE_ID}"
echo "==> Run ID   : ${RUN_ID}"
echo "==> Checkpoint: ${CKPT}"
echo ""
echo "Monitor (bootstrap ~10 min, eval ~20 min):"
echo ""
echo "  Live log:"
echo "    aws s3 cp s3://${S3_BUCKET}/evals/t6/${RUN_ID}/live.log -"
echo ""
echo "  Results:"
echo "    aws s3 cp s3://${S3_BUCKET}/evals/t6/${RUN_ID}/results.txt -"
echo ""
echo "  Instance status:"
echo "    aws ec2 describe-instances --instance-ids ${INSTANCE_ID} --region ${REGION} --query 'Reservations[0].Instances[0].State.Name' --output text"

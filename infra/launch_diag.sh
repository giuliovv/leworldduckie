#!/bin/bash
# Launch a spot t3.medium (CPU-only, no GPU needed) to run:
#   T4 — rollout error vs horizon (action encoder + predictor only; no image re-encoding)
#   T5 — linear probe z → steering (no model needed)
# Results uploaded to s3://leworldduckie/evals/diagnostics/<run_id>/
# (uses evals/ prefix so lewm-ec2-training IAM role can read/write)
#
# Usage:
#   bash infra/launch_diag.sh [--ckpt s3://...] [--skip-t4]
#
# Monitor:
#   aws s3 cp s3://leworldduckie/evals/diagnostics/<run_id>/model_diagnostic_report.txt -
#   aws s3 cp s3://leworldduckie/evals/diagnostics/<run_id>/live.log -

set -euo pipefail

REGION=us-east-1
AMI_ID=ami-05e86b3611c60b0b4   # Ubuntu 22.04 (same as mpc eval)
INSTANCE_TYPE=t3.medium         # CPU-only; T4 uses pre-computed z so no GPU needed
INSTANCE_PROFILE=lewm-ec2-training
SECURITY_GROUP=sg-03bbca875466eb52a
SUBNET=subnet-01497e4f428a93b98
S3_BUCKET=leworldduckie

RUN_ID=$(date -u +%Y%m%d_%H%M%S)
CKPT="s3://${S3_BUCKET}/training/runs/colab_v1/checkpoint_best.pt"
SKIP_T4_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)    CKPT=$2; shift 2 ;;
        --skip-t4) SKIP_T4_FLAG="--skip-t4"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "==> Uploading diagnostic_model.py to S3 ..."
aws s3 cp "$(dirname "$0")/../src/diagnostic_model.py" \
    "s3://${S3_BUCKET}/evals/diagnostic_model.py" --region "$REGION"

USER_DATA=$(cat <<USERDATA
#!/bin/bash
LOG=/var/log/diag.log
exec >>"\$LOG" 2>&1
set -x
echo "=== LeWM diagnostic bootstrap \$(date -u) run=${RUN_ID} ==="
export HOME=/root DEBIAN_FRONTEND=noninteractive

# Live log upload every 30s
(while true; do
    aws s3 cp "\$LOG" "s3://${S3_BUCKET}/evals/diagnostics/${RUN_ID}/live.log" --quiet 2>/dev/null || true
    sleep 30
done) &

apt-get update -q
apt-get install -y -q python3-pip
echo "apt done"

pip3 install --no-cache-dir boto3                                          && echo "boto3 ok"
pip3 install --no-cache-dir h5py einops matplotlib scikit-learn            && echo "deps ok"
pip3 install --no-cache-dir "stable-worldmodel[train]"                     && echo "stable-worldmodel ok"
pip3 install --no-cache-dir --force-reinstall torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu                       && echo "torch cpu ok"

python3 -c "
import boto3
s3 = boto3.client('s3', region_name='${REGION}')
s3.download_file('${S3_BUCKET}', 'evals/diagnostic_model.py', '/tmp/diagnostic_model.py')
print('diagnostic_model.py ok')
"

cd /tmp
python3 diagnostic_model.py \
    --ckpt ${CKPT} \
    --latent-index s3://${S3_BUCKET}/evals/latent_index.npz \
    --data-path s3://${S3_BUCKET}/data/duckietown_100k.h5 \
    --s3-output s3://${S3_BUCKET}/evals/diagnostics/${RUN_ID}/ \
    --max-horizon 15 \
    --n-seqs 300 \
    --probe-epochs 80 \
    ${SKIP_T4_FLAG}
EXIT_CODE=\$?

aws s3 cp "\$LOG" \
    s3://${S3_BUCKET}/evals/diagnostics/${RUN_ID}/instance.log \
    --region ${REGION} || true

echo "=== Diagnostic finished (exit \${EXIT_CODE}) — shutting down ==="
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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=lewm-diag-${RUN_ID}},{Key=Project,Value=leworldduckie}]" \
    --user-data "$USER_DATA" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo ""
echo "==> Instance : ${INSTANCE_ID}"
echo "==> Run ID   : ${RUN_ID}"
echo "==> Checkpoint: ${CKPT}"
echo ""
echo "Monitor (bootstrap ~10 min, diag ~20 min):"
echo ""
echo "  Live log:"
echo "    aws s3 cp s3://${S3_BUCKET}/evals/diagnostics/${RUN_ID}/live.log -"
echo ""
echo "  Report (after completion):"
echo "    aws s3 cp s3://${S3_BUCKET}/evals/diagnostics/${RUN_ID}/model_diagnostic_report.txt -"
echo ""
echo "  Rollout error plot:"
echo "    aws s3 presign s3://${S3_BUCKET}/evals/diagnostics/${RUN_ID}/rollout_error.png --expires-in 3600"
echo ""
echo "  Linear probe plot:"
echo "    aws s3 presign s3://${S3_BUCKET}/evals/diagnostics/${RUN_ID}/linear_probe.png --expires-in 3600"
echo ""
echo "  Instance status:"
echo "    aws ec2 describe-instances --instance-ids ${INSTANCE_ID} --region ${REGION} --query 'Reservations[0].Instances[0].State.Name' --output text"

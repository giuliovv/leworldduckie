#!/bin/bash
# Launch a spot g4dn.xlarge to train LeWM, auto-shuts-down when done.
# Usage: bash launch_training.sh [--epochs N] [--run-id my_run]
#
# Re-launch anytime to start a new run (or resume with same --run-id).
# Artifacts: s3://test-854656252703/lewm-training/runs/<run_id>/
#   loss_curve.png   — updated every epoch
#   metrics.jsonl    — loss per epoch
#   checkpoint_latest.pt / checkpoint_best.pt
#   summary.json     — written when training completes

set -euo pipefail

REGION=us-east-1
AMI_ID=ami-09d0a18beb02cc7d4   # Deep Learning OSS Nvidia PyTorch 2.7 Ubuntu 22.04 (2026-04-19)
INSTANCE_TYPE=g4dn.xlarge
S3_BUCKET=test-854656252703
INSTANCE_PROFILE=lewm-ec2-training
SECURITY_GROUP=sg-03bbca875466eb52a
SUBNET=subnet-0926809da94f51a24

EPOCHS=50
RUN_ID=$(date -u +%Y%m%d_%H%M%S)

# parse optional args
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs) EPOCHS=$2; shift 2 ;;
        --run-id) RUN_ID=$2; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "==> Uploading train.py to S3 ..."
aws s3 cp "$(dirname "$0")/train.py" "s3://${S3_BUCKET}/lewm-training/train.py" --region $REGION

USER_DATA=$(cat <<USERDATA
#!/bin/bash
set -euxo pipefail
exec > >(tee /var/log/lewm-train.log | logger -t lewm-train) 2>&1

echo "=== LeWM Training bootstrap ==="
export HOME=/root
export PATH="\$PATH:/usr/local/cuda/bin"

# Install Python deps (DLAMI already has PyTorch + CUDA)
pip install -q h5py einops matplotlib scikit-learn boto3 stable-worldmodel[train] 2>&1 | tail -5

# Download training script from S3
aws s3 cp s3://${S3_BUCKET}/lewm-training/train.py /tmp/train.py --region ${REGION}

# Run training
cd /tmp
python train.py --run-id ${RUN_ID} --epochs ${EPOCHS}
EXIT_CODE=\$?

# Upload logs to S3
aws s3 cp /var/log/lewm-train.log \
    s3://${S3_BUCKET}/lewm-training/runs/${RUN_ID}/instance.log \
    --region ${REGION} || true

echo "=== Training finished (exit \${EXIT_CODE}) — shutting down ==="
shutdown -h now
USERDATA
)

echo "==> Requesting spot g4dn.xlarge (run_id=${RUN_ID}, epochs=${EPOCHS}) ..."

INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --iam-instance-profile Name="$INSTANCE_PROFILE" \
    --security-group-ids "$SECURITY_GROUP" \
    --subnet-id "$SUBNET" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=lewm-training-${RUN_ID}},{Key=Project,Value=lewm}]" \
    --user-data "$USER_DATA" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "==> Instance launched: $INSTANCE_ID"
echo "==> Run ID: $RUN_ID"
echo ""
echo "Monitor training:"
echo "  Loss plot (refresh anytime):"
echo "    aws s3 presign s3://${S3_BUCKET}/lewm-training/runs/${RUN_ID}/loss_curve.png --expires-in 3600"
echo ""
echo "  Metrics JSONL:"
echo "    aws s3 cp s3://${S3_BUCKET}/lewm-training/runs/${RUN_ID}/metrics.jsonl - | tail -5"
echo ""
echo "  Instance log (after completion):"
echo "    aws s3 cp s3://${S3_BUCKET}/lewm-training/runs/${RUN_ID}/instance.log -"
echo ""
echo "  Instance status:"
echo "    aws ec2 describe-instances --instance-ids $INSTANCE_ID --region $REGION --query 'Reservations[0].Instances[0].State.Name' --output text"

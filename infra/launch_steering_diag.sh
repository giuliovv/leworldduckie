#!/bin/bash
# Launch a spot t3.medium to run T6 steering sensitivity diagnostic.
# Results → s3://leworldduckie/evals/steering_diag/<run_id>/results.txt
#
# Usage: bash infra/launch_steering_diag.sh [--ckpt s3://...] [--n 100] [--k 3]
#
# Monitor:
#   aws s3 cp s3://leworldduckie/evals/steering_diag/<run_id>/results.txt -
#   aws s3 cp s3://leworldduckie/evals/steering_diag/<run_id>/instance.log -

set -euo pipefail

REGION=us-east-1
AMI_ID=ami-05e86b3611c60b0b4
INSTANCE_TYPE=t3.medium
INSTANCE_PROFILE=lewm-ec2-training
SECURITY_GROUP=sg-03bbca875466eb52a
SUBNET=subnet-01497e4f428a93b98
S3_BUCKET=leworldduckie

RUN_ID=$(date -u +%Y%m%d_%H%M%S)
CKPT="s3://${S3_BUCKET}/training/runs/notebook/checkpoint_best.pt"
DATA_S3="s3://${S3_BUCKET}/data/duckietown_100k.h5"
LATENT_INDEX_S3="s3://${S3_BUCKET}/evals/latent_index.npz"
N=100
K=3
FRAMESKIP=1
ENCODE_FROM_HDF5=""   # set to "--encode-from-hdf5" to bypass latent index

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)             CKPT=$2;                 shift 2 ;;
        --n)                N=$2;                    shift 2 ;;
        --k)                K=$2;                    shift 2 ;;
        --frameskip)        FRAMESKIP=$2;             shift 2 ;;
        --encode-from-hdf5) ENCODE_FROM_HDF5="--encode-from-hdf5"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "==> Uploading steering_sensitivity.py to S3 ..."
aws s3 cp "$(dirname "$0")/../src/steering_sensitivity.py" \
    "s3://${S3_BUCKET}/evals/steering_sensitivity.py" --region "$REGION"

USER_DATA=$(cat <<USERDATA
#!/bin/bash
LOG=/var/log/steering-diag.log
exec >>"\$LOG" 2>&1
set -x
echo "=== Steering diag bootstrap \$(date -u) run=${RUN_ID} ==="
export HOME=/root DEBIAN_FRONTEND=noninteractive

# Live log upload every 30s
(while true; do
    aws s3 cp "\$LOG" "s3://${S3_BUCKET}/evals/steering_diag/${RUN_ID}/live.log" --quiet 2>/dev/null || true
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

# Download script
python3 -c "
import boto3
s3 = boto3.client('s3', region_name='${REGION}')
s3.download_file('${S3_BUCKET}', 'evals/steering_sensitivity.py', '/tmp/steering_sensitivity.py')
print('steering_sensitivity.py ok')
"

cd /tmp
python3 steering_sensitivity.py \
    --ckpt ${CKPT} \
    --data-path ${DATA_S3} \
    --latent-index ${LATENT_INDEX_S3} \
    --frameskip ${FRAMESKIP} \
    --n ${N} \
    --k ${K} \
    ${ENCODE_FROM_HDF5} \
    2>&1 | tee /tmp/steering_results.txt
EXIT_CODE=\${PIPESTATUS[0]}
echo "python exit \${EXIT_CODE}"

# Upload results
python3 -c "
import boto3, sys, os
s3 = boto3.client('s3', region_name='us-east-1')
for src, key in [
    ('/tmp/steering_results.txt', 'evals/steering_diag/${RUN_ID}/results.txt'),
    ('/var/log/steering-diag.log', 'evals/steering_diag/${RUN_ID}/instance.log'),
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
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":20,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=lewm-steering-diag-${RUN_ID}},{Key=Project,Value=leworldduckie}]" \
    --user-data "$USER_DATA" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo ""
echo "==> Instance : ${INSTANCE_ID}"
echo "==> Run ID   : ${RUN_ID}"
echo "==> Checkpoint: ${CKPT}"
echo ""
echo "Monitor (bootstrap ~8 min, diag ~5 min):"
echo ""
echo "  Live log:"
echo "    aws s3 cp s3://${S3_BUCKET}/evals/steering_diag/${RUN_ID}/live.log -"
echo ""
echo "  Results:"
echo "    aws s3 cp s3://${S3_BUCKET}/evals/steering_diag/${RUN_ID}/results.txt -"
echo ""
echo "  Instance status:"
echo "    aws ec2 describe-instances --instance-ids ${INSTANCE_ID} --region ${REGION} --query 'Reservations[0].Instances[0].State.Name' --output text"

#!/bin/bash
# Launch a spot t3.medium to run the identity-shortcut test (Step 1).
# Measures ||z_{t+fs} - z_t||^2 at fs=1,3,6 and compares to training best_val.
# Results → s3://leworldduckie/evals/frame_sim/<run_id>/results.txt
#
# Usage: bash infra/launch_frame_similarity.sh [--ckpt s3://...] [--best-val 0.168]
#   [--frameskips 1,3,6] [--try-pusht]

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
BEST_VAL="0.168"          # colab_v1 best_val; set to actual checkpoint val if different
FRAMESKIPS="1,3,6"
TRY_PUSHT=""              # set to "--try-pusht-download" to attempt Push-T download

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)        CKPT=$2;        shift 2 ;;
        --best-val)    BEST_VAL=$2;    shift 2 ;;
        --frameskips)  FRAMESKIPS=$2;  shift 2 ;;
        --try-pusht)   TRY_PUSHT="--try-pusht-download"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "==> Uploading frame_similarity.py to S3 ..."
aws s3 cp "$(dirname "$0")/../src/frame_similarity.py" \
    "s3://${S3_BUCKET}/evals/frame_similarity.py" --region "$REGION"

USER_DATA=$(cat <<USERDATA
#!/bin/bash
LOG=/var/log/frame-sim.log
exec >>"\$LOG" 2>&1
set -x
echo "=== Frame similarity bootstrap \$(date -u) run=${RUN_ID} ==="
export HOME=/root DEBIAN_FRONTEND=noninteractive

# Live log upload every 30s
(while true; do
    aws s3 cp "\$LOG" "s3://${S3_BUCKET}/evals/frame_sim/${RUN_ID}/live.log" --quiet 2>/dev/null || true
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
pip3 install --no-cache-dir huggingface_hub              && echo "huggingface_hub ok"

# Download script
python3 -c "
import boto3
s3 = boto3.client('s3', region_name='${REGION}')
s3.download_file('${S3_BUCKET}', 'evals/frame_similarity.py', '/tmp/frame_similarity.py')
print('frame_similarity.py ok')
"

cd /tmp
python3 frame_similarity.py \
    --ckpt ${CKPT} \
    --data-path ${DATA_S3} \
    --best-val ${BEST_VAL} \
    --frameskips ${FRAMESKIPS} \
    --n-pairs 300 \
    --batch-size 32 \
    ${TRY_PUSHT} \
    2>&1 | tee /tmp/frame_sim_results.txt
EXIT_CODE=\${PIPESTATUS[0]}
echo "python exit \${EXIT_CODE}"

# Upload results
python3 -c "
import boto3, sys, os
s3 = boto3.client('s3', region_name='us-east-1')
for src, key in [
    ('/tmp/frame_sim_results.txt', 'evals/frame_sim/${RUN_ID}/results.txt'),
    ('/var/log/frame-sim.log',     'evals/frame_sim/${RUN_ID}/instance.log'),
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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=lewm-frame-sim-${RUN_ID}},{Key=Project,Value=leworldduckie}]" \
    --user-data "$USER_DATA" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo ""
echo "==> Instance : ${INSTANCE_ID}"
echo "==> Run ID   : ${RUN_ID}"
echo "==> Checkpoint: ${CKPT}"
echo "==> Best val  : ${BEST_VAL}"
echo ""
echo "Monitor (bootstrap ~8 min, measurement ~15 min):"
echo ""
echo "  Live log:"
echo "    aws s3 cp s3://${S3_BUCKET}/evals/frame_sim/${RUN_ID}/live.log -"
echo ""
echo "  Results:"
echo "    aws s3 cp s3://${S3_BUCKET}/evals/frame_sim/${RUN_ID}/results.txt -"
echo ""
echo "  Instance status:"
echo "    aws ec2 describe-instances --instance-ids ${INSTANCE_ID} --region ${REGION} --query 'Reservations[0].Instances[0].State.Name' --output text"

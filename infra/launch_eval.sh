#!/bin/bash
# Launch a spot c5.xlarge to run LeWM VoE eval against live duckietown.
# Outputs: annotated GIF + summary figure + metrics JSON → s3://leworldduckie/evals/runs/<run_id>/
#
# Usage: bash launch_eval.sh [--ckpt s3://...] [--steps N] [--teleport-at N] [--map NAME] [--run-id ID]
#
# Monitor (after ~8 min bootstrap):
#   aws s3 cp s3://leworldduckie/evals/runs/<run_id>/metrics.json -
#   aws s3 presign s3://leworldduckie/evals/runs/<run_id>/voe_demo.gif --expires-in 3600

set -euo pipefail

REGION=us-east-1
AMI_ID=ami-05e86b3611c60b0b4          # Ubuntu 22.04 LTS (Python 3.10) — same as datagen
INSTANCE_TYPE=t3.medium               # 2 vCPU / 4 GB RAM — same as datagen, sufficient
S3_BUCKET=leworldduckie
INSTANCE_PROFILE=lewm-ec2-training
SECURITY_GROUP=sg-03bbca875466eb52a
SUBNET=subnet-00ef452a9147da192

RUN_ID=$(date -u +%Y%m%d_%H%M%S)
CKPT="s3://${S3_BUCKET}/training/runs/notebook/checkpoint_latest.pt"
STEPS=200
TELEPORT_AT=100
MAP=small_loop

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)         CKPT=$2;         shift 2 ;;
        --steps)        STEPS=$2;        shift 2 ;;
        --teleport-at)  TELEPORT_AT=$2;  shift 2 ;;
        --map)          MAP=$2;          shift 2 ;;
        --run-id)       RUN_ID=$2;       shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "==> Uploading run_eval.py to S3 ..."
aws s3 cp "$(dirname "$0")/../src/run_eval.py" \
    "s3://${S3_BUCKET}/evals/run_eval.py" --region "$REGION"

USER_DATA=$(cat <<USERDATA
#!/bin/bash
LOG=/var/log/lewm-eval.log
exec >>"\$LOG" 2>&1
set -x
echo "=== LeWM eval bootstrap \$(date -u) run=${RUN_ID} ==="
export HOME=/root DEBIAN_FRONTEND=noninteractive

# System deps (duckietown needs OpenGL + Xvfb)
apt-get update -q
apt-get install -y -q libgl1 libglu1-mesa libglib2.0-0 xvfb python3-pip \
    python3-opencv libegl1 libegl-mesa0 libglx-mesa0 git
echo "apt done"

# Python deps
# Install numpy<2 first; re-pin after stable-worldmodel since it upgrades numpy back to 2.x
pip3 install --no-cache-dir boto3                           && echo "boto3 ok"
pip3 install --no-cache-dir "numpy<2.0.0"                   && echo "numpy<2 ok"
pip3 install --no-cache-dir "pyglet==1.5.27"                && echo "pyglet ok"
pip3 install --no-cache-dir "duckietown-gym-daffy"          && echo "daffy ok"
pip3 install --no-cache-dir h5py                            && echo "h5py ok"
pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && echo "torch ok"
pip3 install --no-cache-dir "stable-worldmodel[train]"      && echo "stable-worldmodel ok"
pip3 install --no-cache-dir "numpy<2.0.0"                   && echo "numpy<2 re-pinned ok"
pip3 install --no-cache-dir einops imageio pillow           && echo "misc deps ok"

# Patch duckietown_world pwm_dynamics (numpy array / float inhomogeneous list)
python3 -c "
path = '/usr/local/lib/python3.10/dist-packages/duckietown_world/world_duckietown/pwm_dynamics.py'
with open(path) as f: txt = f.read()
old = '        linear = [longitudinal, lateral]'
new = '        linear = [float(longitudinal), lateral]'
if old in txt:
    with open(path, 'w') as f: f.write(txt.replace(old, new))
    print('pwm_dynamics patch applied')
else:
    print('pwm_dynamics patch: already applied or not needed')
" && echo "patch ok"

# Verify duckietown import
python3 -c "import gym_duckietown; print('gym_duckietown ok')" || echo "import FAILED"

# Start virtual display
Xvfb :99 -screen 0 1024x768x24 &
sleep 2
echo "Xvfb started"
export DISPLAY=:99

# Download eval script via boto3 (aws CLI not available on this AMI)
python3 -c "import boto3; boto3.client('s3', region_name='${REGION}').download_file('${S3_BUCKET}', 'evals/run_eval.py', '/tmp/run_eval.py'); print('script download ok')"
cd /tmp
python3 run_eval.py \
    --run-id "${RUN_ID}" \
    --ckpt "${CKPT}" \
    --steps ${STEPS} \
    --teleport-at ${TELEPORT_AT} \
    --map "${MAP}"
EXIT_CODE=\$?
echo "python exit \${EXIT_CODE}"

# Upload log to S3
python3 -c "
import boto3, sys
try:
    boto3.client('s3', region_name='us-east-1').upload_file(
        '/var/log/lewm-eval.log', '${S3_BUCKET}', 'evals/runs/${RUN_ID}/instance.log')
    print('log upload ok')
except Exception as e:
    print('log upload failed:', e, file=sys.stderr)
" || true

echo "=== eval done, shutting down ==="
shutdown -h now
USERDATA
)

echo "==> Requesting spot t3.medium (run_id=${RUN_ID}) ..."

INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --iam-instance-profile Name="$INSTANCE_PROFILE" \
    --security-group-ids "$SECURITY_GROUP" \
    --subnet-id "$SUBNET" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":30,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=lewm-eval-${RUN_ID}},{Key=Project,Value=leworldduckie}]" \
    --user-data "$USER_DATA" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo ""
echo "==> Instance: ${INSTANCE_ID}"
echo "==> Run ID:   ${RUN_ID}"
echo "==> Config:   steps=${STEPS}  teleport_at=${TELEPORT_AT}  map=${MAP}"
echo "==> Checkpoint: ${CKPT}"
echo ""
echo "Monitor (bootstrap ~8 min, eval ~10 min):"
echo ""
echo "  Instance log (after completion):"
echo "    aws s3 cp s3://${S3_BUCKET}/evals/runs/${RUN_ID}/instance.log -"
echo ""
echo "  Metrics:"
echo "    aws s3 cp s3://${S3_BUCKET}/evals/runs/${RUN_ID}/metrics.json -"
echo ""
echo "  GIF (presigned URL):"
echo "    aws s3 presign s3://${S3_BUCKET}/evals/runs/${RUN_ID}/voe_demo.gif --expires-in 3600"
echo ""
echo "  Summary figure:"
echo "    aws s3 presign s3://${S3_BUCKET}/evals/runs/${RUN_ID}/voe_summary.png --expires-in 3600"
echo ""
echo "  Instance status:"
echo "    aws ec2 describe-instances --instance-ids ${INSTANCE_ID} --region ${REGION} --query 'Reservations[0].Instances[0].State.Name' --output text"

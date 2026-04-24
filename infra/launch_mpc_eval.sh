#!/bin/bash
# Launch a spot t3.medium to run the LeWM MPC lane-following eval (10 episodes).
# Results (stdout summary + instance log) → s3://leworldduckie/evals/mpc/<run_id>/
#
# Usage: bash infra/launch_mpc_eval.sh [--ckpt s3://...] [--steps N] [--map NAME]
#
# Monitor (after ~10 min bootstrap + ~20 min eval):
#   aws s3 cp s3://leworldduckie/evals/mpc/<run_id>/results.txt -
#   aws s3 cp s3://leworldduckie/evals/mpc/<run_id>/instance.log -

set -euo pipefail

REGION=us-east-1
AMI_ID=ami-05e86b3611c60b0b4
INSTANCE_TYPE=t3.medium
INSTANCE_PROFILE=lewm-ec2-training
SECURITY_GROUP=sg-03bbca875466eb52a
SUBNET=subnet-00ef452a9147da192
S3_BUCKET=leworldduckie

RUN_ID=$(date -u +%Y%m%d_%H%M%S)
CKPT="s3://${S3_BUCKET}/training/runs/notebook/checkpoint_latest.pt"
GOAL_S3="s3://${S3_BUCKET}/evals/mpc_goal.png"
STEPS=300
MAP_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)  CKPT=$2;      shift 2 ;;
        --goal)  GOAL_S3=$2;   shift 2 ;;
        --steps) STEPS=$2;     shift 2 ;;
        --map)   MAP_ARG="--map $2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "==> Uploading mpc_controller.py to S3 ..."
aws s3 cp "$(dirname "$0")/../src/mpc_controller.py" \
    "s3://${S3_BUCKET}/evals/mpc_controller.py" --region "$REGION"

USER_DATA=$(cat <<USERDATA
#!/bin/bash
LOG=/var/log/mpc-eval.log
exec >>"\$LOG" 2>&1
set -x
echo "=== MPC eval bootstrap \$(date -u) run=${RUN_ID} ==="
export HOME=/root DEBIAN_FRONTEND=noninteractive

apt-get update -q
apt-get install -y -q libgl1 libglu1-mesa libglib2.0-0 xvfb python3-pip \
    python3-opencv libegl1 libegl-mesa0 libglx-mesa0 git
echo "apt done"

pip3 install --no-cache-dir boto3                        && echo "boto3 ok"
pip3 install --no-cache-dir "numpy<2.0.0"                && echo "numpy<2 ok"
pip3 install --no-cache-dir "pyglet==1.5.27"             && echo "pyglet ok"
pip3 install --no-cache-dir "duckietown-gym-daffy"       && echo "daffy ok"
pip3 install --no-cache-dir h5py                         && echo "h5py ok"
pip3 install --no-cache-dir "stable-worldmodel[train]"   && echo "stable-worldmodel ok"
pip3 install --no-cache-dir --force-reinstall torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu     && echo "torch cpu ok"
pip3 install --no-cache-dir "numpy<2.0.0"                && echo "numpy<2 re-pin ok"
pip3 install --no-cache-dir einops imageio pillow        && echo "misc deps ok"

# Patch pwm_dynamics (numpy array / float inhomogeneous list)
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

python3 -c "import gym_duckietown; print('gym_duckietown ok')" || echo "import FAILED"

# Map symlink
DT_WORLD=\$(python3 -c "import duckietown_world,os;print(os.path.dirname(duckietown_world.__file__))" 2>/dev/null | tail -1)
DT_GYM=\$(python3 -c "import gym_duckietown,os;print(os.path.dirname(gym_duckietown.__file__))" 2>/dev/null | tail -1)
ln -sf "\${DT_WORLD}/data/gd1/maps" "\${DT_GYM}/maps" 2>/dev/null || true

# Xvfb
Xvfb :99 -screen 0 1024x768x24 &
sleep 2
export DISPLAY=:99

# Download script, checkpoint, and goal frame
python3 -c "
import boto3
s3 = boto3.client('s3', region_name='${REGION}')
s3.download_file('${S3_BUCKET}', 'evals/mpc_controller.py', '/tmp/mpc_controller.py')
print('mpc_controller.py ok')
s3.download_file('${S3_BUCKET}', '${CKPT#s3://${S3_BUCKET}/}', '/tmp/lewm_best.pt')
print('checkpoint ok')
s3.download_file('${S3_BUCKET}', '${GOAL_S3#s3://${S3_BUCKET}/}', '/tmp/mpc_goal.png')
print('goal ok')
"

cd /tmp
DISPLAY=:99 python3 mpc_controller.py \
    --ckpt /tmp/lewm_best.pt \
    --goal /tmp/mpc_goal.png \
    --steps ${STEPS} \
    --episodes 10 \
    --frameskip 3 \
    --lag-frames 4 \
    --verbose \
    ${MAP_ARG} 2>&1 | tee /tmp/mpc_results.txt
EXIT_CODE=\${PIPESTATUS[0]}
echo "python exit \${EXIT_CODE}"

# Upload results and log
python3 -c "
import boto3, sys
s3 = boto3.client('s3', region_name='us-east-1')
for src, key in [
    ('/tmp/mpc_results.txt', 'evals/mpc/${RUN_ID}/results.txt'),
    ('/var/log/mpc-eval.log', 'evals/mpc/${RUN_ID}/instance.log'),
]:
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
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=lewm-mpc-eval-${RUN_ID}},{Key=Project,Value=leworldduckie}]" \
    --user-data "$USER_DATA" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo ""
echo "==> Instance : ${INSTANCE_ID}"
echo "==> Run ID   : ${RUN_ID}"
echo "==> Checkpoint: ${CKPT}"
echo "==> Steps    : ${STEPS}"
echo ""
echo "Monitor (bootstrap ~10 min, eval ~20 min):"
echo ""
echo "  Results (stdout summary):"
echo "    aws s3 cp s3://${S3_BUCKET}/evals/mpc/${RUN_ID}/results.txt -"
echo ""
echo "  Instance log:"
echo "    aws s3 cp s3://${S3_BUCKET}/evals/mpc/${RUN_ID}/instance.log -"
echo ""
echo "  Instance status:"
echo "    aws ec2 describe-instances --instance-ids ${INSTANCE_ID} --region ${REGION} --query 'Reservations[0].Instances[0].State.Name' --output text"

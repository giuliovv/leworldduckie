#!/bin/bash
# Launch a spot t3.medium to train a BC baseline and run 10-episode eval.
# Training: MLP z→action on latent_index + HDF5 (frozen encoder)
# Eval: same Duckietown setup as MPC eval
# Results → s3://leworldduckie/evals/mpc/bc_<run_id>/  (uses evals/mpc/ prefix — IAM-accessible)

set -euo pipefail

REGION=us-east-1
AMI_ID=ami-05e86b3611c60b0b4
INSTANCE_TYPE=t3.medium
INSTANCE_PROFILE=lewm-ec2-training
SECURITY_GROUP=sg-03bbca875466eb52a
SUBNET=subnet-01497e4f428a93b98
S3_BUCKET=leworldduckie

RUN_ID=$(date -u +%Y%m%d_%H%M%S)
CKPT="s3://${S3_BUCKET}/training/runs/colab_v1/checkpoint_best.pt"
STEPS=300
MAP_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)  CKPT=$2;              shift 2 ;;
        --steps) STEPS=$2;             shift 2 ;;
        --map)   MAP_ARG="--map $2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "==> Uploading bc_controller.py to S3 ..."
aws s3 cp "$(dirname "$0")/../src/bc_controller.py" \
    "s3://${S3_BUCKET}/evals/bc_controller.py" --region "$REGION"

USER_DATA=$(cat <<USERDATA
#!/bin/bash
LOG=/var/log/bc-eval.log
exec >>"\$LOG" 2>&1
set -x
echo "=== BC eval bootstrap \$(date -u) run=${RUN_ID} ==="
export HOME=/root DEBIAN_FRONTEND=noninteractive

(while true; do
    aws s3 cp "\$LOG" "s3://${S3_BUCKET}/evals/mpc/bc_${RUN_ID}/live.log" --quiet 2>/dev/null || true
    sleep 30
done) &

apt-get update -q
apt-get install -y -q python3-pip libgl1 libglu1-mesa libglib2.0-0 xvfb python3-opencv \
    libegl1 libegl-mesa0 libglx-mesa0 git
echo "apt done"

pip3 install --no-cache-dir boto3                                          && echo "boto3 ok"
python3 -c "import boto3; boto3.client('s3',region_name='${REGION}').put_object(Bucket='${S3_BUCKET}',Key='evals/mpc/bc_${RUN_ID}/started.txt',Body=b'bootstrap_running'); print('ping ok')" || echo "ping failed"
pip3 install --no-cache-dir "numpy<2.0.0"                                  && echo "numpy ok"
pip3 install --no-cache-dir "pyglet==1.5.27"                               && echo "pyglet ok"
pip3 install --no-cache-dir "duckietown-gym-daffy"                         && echo "daffy ok"
pip3 install --no-cache-dir h5py einops                                    && echo "h5py ok"
pip3 install --no-cache-dir "stable-worldmodel[train]"                     && echo "stable-worldmodel ok"
pip3 install --no-cache-dir --force-reinstall torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu                       && echo "torch cpu ok"
pip3 install --no-cache-dir "numpy<2.0.0"                                  && echo "numpy re-pin ok"
pip3 install --no-cache-dir imageio pillow                                 && echo "misc ok"

python3 -c "
path = '/usr/local/lib/python3.10/dist-packages/duckietown_world/world_duckietown/pwm_dynamics.py'
import os
if os.path.exists(path):
    with open(path) as f: txt = f.read()
    old = '        linear = [longitudinal, lateral]'
    new = '        linear = [float(longitudinal), lateral]'
    if old in txt:
        with open(path, 'w') as f: f.write(txt.replace(old, new))
        print('pwm patch applied')
"

python3 -c "
import boto3
s3 = boto3.client('s3', region_name='${REGION}')
s3.download_file('${S3_BUCKET}', 'evals/bc_controller.py', '/tmp/bc_controller.py')
print('bc_controller.py ok')
"

DT_WORLD=\$(python3 -c "import duckietown_world,os;print(os.path.dirname(duckietown_world.__file__))" 2>/dev/null | tail -1)
DT_GYM=\$(python3 -c "import gym_duckietown,os;print(os.path.dirname(gym_duckietown.__file__))" 2>/dev/null | tail -1)
ln -sf "\${DT_WORLD}/data/gd1/maps" "\${DT_GYM}/maps" 2>/dev/null || true

Xvfb :99 -screen 0 1024x768x24 &
sleep 2
export DISPLAY=:99

cd /tmp
DISPLAY=:99 python3 bc_controller.py \
    --ckpt ${CKPT} \
    --latent-index s3://${S3_BUCKET}/evals/latent_index.npz \
    --data-path s3://${S3_BUCKET}/data/duckietown_100k.h5 \
    --s3-output s3://${S3_BUCKET}/evals/mpc/bc_${RUN_ID}/ \
    --episodes 10 \
    --steps ${STEPS} \
    --train-epochs 50 \
    --gif-dir /tmp/bc_gifs \
    ${MAP_ARG} 2>&1 | tee /tmp/bc_results.txt
EXIT_CODE=\${PIPESTATUS[0]}
echo "python exit \${EXIT_CODE}"

aws s3 cp "\$LOG" "s3://${S3_BUCKET}/evals/mpc/bc_${RUN_ID}/instance.log" --region ${REGION} || true

echo "=== BC eval done (exit \${EXIT_CODE}) — shutting down ==="
shutdown -h now
USERDATA
)

echo "==> Launching on-demand t3.medium (run_id=${RUN_ID}) ..."

INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --iam-instance-profile Name="$INSTANCE_PROFILE" \
    --security-group-ids "$SECURITY_GROUP" \
    --subnet-id "$SUBNET" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":30,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=lewm-bc-${RUN_ID}},{Key=Project,Value=leworldduckie}]" \
    --user-data "$USER_DATA" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo ""
echo "==> Instance : ${INSTANCE_ID}"
echo "==> Run ID   : ${RUN_ID}"
echo ""
echo "Monitor:"
echo "  aws s3 cp s3://${S3_BUCKET}/evals/mpc/bc_${RUN_ID}/live.log -"
echo "  aws s3 cp s3://${S3_BUCKET}/evals/mpc/bc_${RUN_ID}/bc_results.txt -"
echo "  aws ec2 describe-instances --instance-ids ${INSTANCE_ID} --region ${REGION} --query 'Reservations[0].Instances[0].State.Name' --output text"

#!/bin/bash
# Launch a t3.medium to collect 100k duckietown transitions and upload to S3.
# Usage: bash launch_datagen.sh [--n-transitions N]

set -euo pipefail

REGION=us-east-1
AMI_ID=ami-05e86b3611c60b0b4
INSTANCE_TYPE=t3.medium
INSTANCE_PROFILE=lewm-ec2-training
SECURITY_GROUP=sg-03bbca875466eb52a
SUBNET=subnet-00ef452a9147da192
DATA_BUCKET=leworldduckie

N_TRANSITIONS=100000
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-transitions) N_TRANSITIONS=$2; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

RUN_ID=$(date -u +%Y%m%d_%H%M%S)

echo "==> Uploading generate_data.py to S3 ..."
aws s3 cp "$(dirname "$0")/../src/generate_data.py" \
    "s3://${DATA_BUCKET}/datagen/generate_data.py" --region "$REGION"

USER_DATA=$(cat << USERDATA
#!/bin/bash
LOG=/var/log/datagen.log
exec >>"\$LOG" 2>&1
set -x
echo "=== datagen start \$(date -u) run=${RUN_ID} ==="
export HOME=/root DEBIAN_FRONTEND=noninteractive

apt-get update -q
# NOTE: python3-h5py is intentionally excluded — it's compiled against older numpy
# and causes binary incompatibility. h5py is installed via pip instead.
apt-get install -y -q libgl1 libglu1-mesa libglib2.0-0 xvfb python3-pip python3-opencv \
    libegl1 libegl-mesa0 libglx-mesa0
echo "apt done"

pip3 install --no-cache-dir boto3 && echo "boto3 ok" || echo "boto3 FAILED"
# Pin numpy<2 — duckietown_world geometry lib is incompatible with numpy 2.x
pip3 install --no-cache-dir "numpy<2.0.0" && echo "numpy ok" || echo "numpy FAILED"
pip3 install --no-cache-dir "pyglet==1.5.27" && echo "pyglet ok" || echo "pyglet FAILED"
pip3 install --no-cache-dir "duckietown-gym-daffy" && echo "daffy ok" || echo "daffy FAILED"
pip3 install --no-cache-dir h5py && echo "h5py ok" || echo "h5py FAILED"
python3 -c "import h5py; print('h5py', h5py.__version__, 'from', h5py.__file__)" || echo "h5py import FAILED"

# Patch duckietown_world pwm_dynamics: longitudinal is shape (1,), lateral is 0.0 → inhomogeneous.
# Fix: cast longitudinal to float before building linear list.
python3 -c "
path = '/usr/local/lib/python3.10/dist-packages/duckietown_world/world_duckietown/pwm_dynamics.py'
with open(path) as f: txt = f.read()
old = '        linear = [longitudinal, lateral]'
new = '        linear = [float(longitudinal), lateral]'
if old in txt:
    with open(path, 'w') as f: f.write(txt.replace(old, new))
    print('pwm_dynamics patch applied')
else:
    print('pwm_dynamics patch: already applied or line not found')
" && echo "patch ok" || echo "patch FAILED"

python3 -c "import gym_duckietown; print('import ok')" || echo "import FAILED"

# Use tail -1 to strip pyglet options dict that gym_duckietown prints to stdout
DT_WORLD=\$(python3 -c "import duckietown_world,os;print(os.path.dirname(duckietown_world.__file__))" 2>/dev/null | tail -1)
DT_GYM=\$(python3 -c "import gym_duckietown,os;print(os.path.dirname(gym_duckietown.__file__))" 2>/dev/null | tail -1)
echo "DT_WORLD=\${DT_WORLD}"
echo "DT_GYM=\${DT_GYM}"
ln -sf "\${DT_WORLD}/data/gd1/maps" "\${DT_GYM}/maps" 2>/dev/null && echo "symlink ok" || echo "symlink skipped"

# Download generate_data.py from S3 (always uses the latest version)
python3 -c "import boto3; boto3.client('s3', region_name='${REGION}').download_file('${DATA_BUCKET}', 'datagen/generate_data.py', '/tmp/generate_data.py'); print('script download ok')"

sleep 3
Xvfb :99 -screen 0 1024x768x24 &
sleep 3
echo "xvfb started"

cd /tmp
DISPLAY=:99 python3 generate_data.py --n-transitions ${N_TRANSITIONS} --out /tmp/duckietown_100k.h5 --upload
EXIT_CODE=\$?
echo "python exit \${EXIT_CODE}"

python3 -c "
import boto3, sys
try:
    s3 = boto3.client('s3', region_name='us-east-1')
    s3.upload_file('/var/log/datagen.log', '${DATA_BUCKET}', 'logs/datagen_${RUN_ID}.log')
    print('log upload ok')
except Exception as e:
    print('log upload failed:', e, file=sys.stderr)
" || true
echo "=== done, shutting down ==="
shutdown -h now
USERDATA
)

echo "==> Launching t3.medium spot (run_id=${RUN_ID}) ..."

INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --iam-instance-profile Name="$INSTANCE_PROFILE" \
    --security-group-ids "$SECURITY_GROUP" \
    --subnet-id "$SUBNET" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":20,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=lewm-datagen-${RUN_ID}},{Key=Project,Value=leworldduckie}]" \
    --user-data "$USER_DATA" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "==> Instance: $INSTANCE_ID  run_id: $RUN_ID"
echo ""
echo "Log (after ~5 min bootstrap + ~2.5h run):"
echo "  aws s3 cp s3://${DATA_BUCKET}/logs/datagen_${RUN_ID}.log -"
echo ""
echo "Status: aws ec2 describe-instances --instance-ids $INSTANCE_ID --region $REGION --query 'Reservations[0].Instances[0].State.Name' --output text"

#!/bin/bash
# Launch a t3.medium to collect 100k duckietown transitions and upload to S3.
# Usage: bash launch_datagen.sh [--n-transitions N]
#
# Output: s3://leworldduckie/data/duckietown_100k.h5
# Logs:   s3://leworldduckie/logs/datagen_<run_id>.log

set -euo pipefail

REGION=us-east-1
AMI_ID=ami-05e86b3611c60b0b4   # Ubuntu 22.04 LTS amd64 (2026-04-10)
INSTANCE_TYPE=t3.medium          # 4 GB RAM — enough headroom for gym-duckietown GL
INSTANCE_PROFILE=lewm-ec2-training
SECURITY_GROUP=sg-03bbca875466eb52a
SUBNET=subnet-00ef452a9147da192  # us-east-1a (t3 not available in us-east-1e)
S3_BUCKET=leworldduckie

N_TRANSITIONS=100000
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-transitions) N_TRANSITIONS=$2; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

RUN_ID=$(date -u +%Y%m%d_%H%M%S)
SCRIPT_S3_KEY="scripts/generate_data_${RUN_ID}.py"

echo "==> Uploading generate_data.py to s3://${S3_BUCKET}/${SCRIPT_S3_KEY} ..."
aws s3 cp "$(dirname "$0")/generate_data.py" "s3://${S3_BUCKET}/${SCRIPT_S3_KEY}" --region $REGION

USER_DATA=$(cat <<USERDATA
#!/bin/bash
set -euxo pipefail
exec > >(tee /var/log/datagen.log | logger -t datagen) 2>&1

echo "=== Duckietown datagen bootstrap (run ${RUN_ID}) ==="
export HOME=/root
export DEBIAN_FRONTEND=noninteractive

# System deps
apt-get update -q
apt-get install -y -q libgl1 libglu1-mesa libglib2.0-0 xvfb python3-pip python3-opencv

# Python deps (pyglet 1.5.27 required by gym-duckietown)
pip3 install -q "duckietown-gym-daffy" "pyglet==1.5.27" h5py boto3 opencv-python

# Symlink maps (duckietown_world -> gym_duckietown)
DT_WORLD=\$(python3 -c "import duckietown_world,os;print(os.path.dirname(duckietown_world.__file__))")
DT_GYM=\$(python3 -c "import gym_duckietown,os;print(os.path.dirname(gym_duckietown.__file__))")
ln -sf "\${DT_WORLD}/data/gd1/maps" "\${DT_GYM}/maps" || true

# Download collection script
aws s3 cp s3://${S3_BUCKET}/${SCRIPT_S3_KEY} /tmp/generate_data.py --region ${REGION}

# Start virtual display
Xvfb :99 -screen 0 1024x768x24 &
sleep 2

# Collect and upload
cd /tmp
DISPLAY=:99 python3 generate_data.py \
    --n-transitions ${N_TRANSITIONS} \
    --out /tmp/duckietown_100k.h5 \
    --upload
EXIT_CODE=\$?

# Upload log
aws s3 cp /var/log/datagen.log \
    s3://${S3_BUCKET}/logs/datagen_${RUN_ID}.log \
    --region ${REGION} || true

echo "=== Datagen finished (exit \${EXIT_CODE}) — shutting down ==="
shutdown -h now
USERDATA
)

echo "==> Launching t3.medium spot instance (run_id=${RUN_ID}, n_transitions=${N_TRANSITIONS}) ..."

INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --iam-instance-profile Name="$INSTANCE_PROFILE" \
    --security-group-ids "$SECURITY_GROUP" \
    --subnet-id "$SUBNET" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":20,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=lewm-datagen-${RUN_ID}},{Key=Project,Value=lewm}]" \
    --user-data "$USER_DATA" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo ""
echo "==> Instance launched: $INSTANCE_ID"
echo "==> Run ID: $RUN_ID"
echo ""
echo "Monitor:"
echo "  Log (after ~5 min):"
echo "    aws s3 cp s3://${S3_BUCKET}/logs/datagen_${RUN_ID}.log -"
echo ""
echo "  Dataset (when complete):"
echo "    aws s3 ls s3://${S3_BUCKET}/data/duckietown_100k.h5"
echo ""
echo "  Instance status:"
echo "    aws ec2 describe-instances --instance-ids $INSTANCE_ID --region $REGION --query 'Reservations[0].Instances[0].State.Name' --output text"

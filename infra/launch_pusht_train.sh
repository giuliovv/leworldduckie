#!/bin/bash
# Launch a spot g4dn.xlarge to train LeWM on Push-T (official le-wm repo).
#
# Steps on the instance:
#   1. Install stable-worldmodel[train,env] + swig + le-wm deps
#   2. Clone le-wm
#   3. Download pusht_expert_train.h5 from HuggingFace (quentinll/lewm-pusht)
#   4. python3 train.py data=pusht subdir=pusht ...  (~4h on T4 / ~1h on A100)
#   5. Upload checkpoint(s) + logs to S3
#
# Artifacts:
#   training/pusht/pusht/lewm_object.ckpt        — canonical checkpoint (latest epoch)
#   training/pusht/<run_id>/lewm_epoch_N_object.ckpt — per-epoch snapshots
#   training/pusht/<run_id>/train_stdout.txt     — full training log
#   training/pusht/<run_id>/live.log             — rolling instance log (updated every 30s)
#   training/pusht/<run_id>/instance.log         — final instance log
#
# Usage:
#   bash infra/launch_pusht_train.sh [--epochs 100] [--batch 128] [--run-id ID]

set -euo pipefail

REGION=us-east-1
AMI_ID=ami-09d0a18beb02cc7d4   # Deep Learning OSS Nvidia PyTorch 2.7 Ubuntu 22.04
INSTANCE_TYPE=g4dn.xlarge       # T4 16 GB → batch=128, ~4h / 100 epochs
INSTANCE_PROFILE=lewm-ec2-training
SECURITY_GROUP=sg-03bbca875466eb52a
SUBNET=subnet-0b799a4832af70f5b
S3_BUCKET=leworldduckie

EPOCHS=100
BATCH_SIZE=128
PRECISION=bf16-mixed
NUM_WORKERS=4
RUN_ID=$(date -u +%Y%m%d_%H%M%S)

while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)  EPOCHS=$2;     shift 2 ;;
        --batch)   BATCH_SIZE=$2; shift 2 ;;
        --run-id)  RUN_ID=$2;     shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

USER_DATA=$(cat <<USERDATA
#!/bin/bash
LOG=/var/log/pusht-train.log
exec >>"\$LOG" 2>&1
set -x
echo "=== Push-T training bootstrap \$(date -u) run=${RUN_ID} ==="
export HOME=/root DEBIAN_FRONTEND=noninteractive MUJOCO_GL=egl

# Live log upload every 30s
(while true; do
    aws s3 cp "\$LOG" "s3://${S3_BUCKET}/training/pusht/${RUN_ID}/live.log" --quiet 2>/dev/null || true
    sleep 30
done) &

export PATH="\$PATH:/usr/local/cuda/bin"

# Install deps (DLAMI has PyTorch + CUDA; install le-wm extras)
apt-get install -y -q swig build-essential
pip3 install -q 'stable-worldmodel[train,env]' einops 'numpy<2.0.0' zstandard huggingface_hub && echo "deps ok"

# Clone le-wm
git clone --depth 1 https://github.com/lucas-maes/le-wm.git /tmp/le-wm && echo "le-wm cloned"

# Download Push-T dataset from HuggingFace (~13 GB compressed → ~22 GB decompressed)
mkdir -p /root/.stable-wm
python3 -c "
import os, zstandard
from pathlib import Path
from huggingface_hub import hf_hub_download

home = Path('/root/.stable-wm')
h5   = home / 'pusht_expert_train.h5'
zst  = home / 'pusht_expert_train.h5.zst'

if h5.exists():
    print(f'Dataset cached: {h5} ({h5.stat().st_size / 1e9:.1f} GB)')
else:
    if not zst.exists():
        print('Downloading pusht_expert_train.h5.zst from HuggingFace (~13 GB)...')
        dl = hf_hub_download(
            repo_id='quentinll/lewm-pusht',
            filename='pusht_expert_train.h5.zst',
            repo_type='dataset',
            local_dir=str(home),
        )
        from pathlib import Path as P
        dl_p = P(dl)
        if dl_p.resolve() != zst.resolve():
            import shutil
            shutil.copy2(dl_p, zst)

    print('Decompressing...')
    dctx = zstandard.ZstdDecompressor()
    with open(zst, 'rb') as fin, open(h5, 'wb') as fout:
        dctx.copy_stream(fin, fout)
    zst.unlink()
    print(f'Dataset ready: {h5.stat().st_size / 1e9:.1f} GB')
"
echo "dataset ok"

# Run training
export STABLEWM_HOME=/root/.stable-wm
cd /tmp/le-wm
python3 train.py \
    data=pusht \
    subdir=pusht \
    wandb.enabled=false \
    trainer.max_epochs=${EPOCHS} \
    loader.batch_size=${BATCH_SIZE} \
    loader.num_workers=${NUM_WORKERS} \
    trainer.precision=${PRECISION} \
    2>&1 | tee /tmp/train_stdout.txt
TRAIN_EXIT=\${PIPESTATUS[0]}
echo "training exit: \${TRAIN_EXIT}"

# Upload checkpoint(s) and logs to S3
python3 -c "
import glob, boto3
from pathlib import Path

s3      = boto3.client('s3', region_name='us-east-1')
bucket  = '${S3_BUCKET}'
run_id  = '${RUN_ID}'
run_dir = Path('/root/.stable-wm/pusht')

# Per-epoch checkpoints
ckpts = sorted(glob.glob(str(run_dir / 'lewm_epoch_*_object.ckpt')))
for ckpt in ckpts:
    key = f'training/pusht/{run_id}/{Path(ckpt).name}'
    s3.upload_file(ckpt, bucket, key)
    print(f'uploaded {key}')

# Canonical checkpoint (latest epoch overwrites the shared path for eval)
canonical_src = ckpts[-1] if ckpts else str(run_dir / 'lewm_object.ckpt')
if Path(canonical_src).exists():
    s3.upload_file(canonical_src, bucket, 'training/pusht/pusht/lewm_object.ckpt')
    print('uploaded training/pusht/pusht/lewm_object.ckpt')
else:
    print('WARNING: no checkpoint found')

for src, key in [
    ('/tmp/train_stdout.txt',       f'training/pusht/{run_id}/train_stdout.txt'),
    ('/var/log/pusht-train.log',    f'training/pusht/{run_id}/instance.log'),
]:
    try:
        if Path(src).exists():
            s3.upload_file(src, bucket, key)
            print(f'uploaded {key}')
    except Exception as e:
        print(f'upload failed {key}: {e}')
" || true

echo "=== done, shutting down ==="
shutdown -h now
USERDATA
)

echo "==> Launching spot g4dn.xlarge for Push-T training (run_id=${RUN_ID}) ..."

INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --iam-instance-profile Name="$INSTANCE_PROFILE" \
    --security-group-ids "$SECURITY_GROUP" \
    --subnet-id "$SUBNET" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":80,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=lewm-pusht-train-${RUN_ID}},{Key=Project,Value=leworldduckie}]" \
    --user-data "$USER_DATA" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo ""
echo "==> Instance  : ${INSTANCE_ID}"
echo "==> Run ID    : ${RUN_ID}"
echo ""
echo "Monitor (bootstrap ~15 min, training ~4h on T4 / ~1h on A100):"
echo ""
echo "  Live log:"
echo "    aws s3 cp s3://${S3_BUCKET}/training/pusht/${RUN_ID}/live.log -"
echo ""
echo "  Checkpoint (per-epoch, updated as training progresses):"
echo "    aws s3 ls s3://${S3_BUCKET}/training/pusht/${RUN_ID}/"
echo ""
echo "  Canonical checkpoint (for eval):"
echo "    aws s3 ls s3://${S3_BUCKET}/training/pusht/pusht/lewm_object.ckpt"
echo ""
echo "  Instance status:"
echo "    aws ec2 describe-instances --instance-ids ${INSTANCE_ID} --region ${REGION} --query 'Reservations[0].Instances[0].State.Name' --output text"
echo ""
echo "  Run eval after training completes:"
echo "    bash infra/launch_pusht_eval.sh"

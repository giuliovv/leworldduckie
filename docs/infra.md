# Infrastructure

## S3 Bucket: `leworldduckie` (us-east-1)

```
s3://leworldduckie/
├── data/
│   └── duckietown_100k.h5          # 1.87 GB, 100k transitions, 738 episodes (2026-04-24)
├── scripts/
│   └── generate_data.py            # uploaded by launch_datagen.sh before EC2 launch
├── training/
│   └── runs/
│       └── <run_id>/
│           ├── checkpoint_latest.pt
│           ├── checkpoint_best.pt
│           ├── loss_curve.png
│           ├── metrics.jsonl
│           └── summary.json
├── evals/
│   ├── mpc_goal.png                # goal frame for MPC
│   ├── mpc_controller.py           # uploaded by launch_mpc_eval.sh
│   └── mpc/
│       └── <run_id>/
│           ├── results.txt         # stdout summary (all episodes)
│           ├── instance.log        # full EC2 bootstrap + eval log
│           ├── best_episode.gif    # best episode GIF
│           ├── ep0.gif ... ep9.gif # all episode GIFs
│           └── progress.txt        # updated live after each episode
└── logs/
    └── datagen_<run_id>.log        # datagen instance logs
```

## EC2 Configuration

| Parameter | Value |
|-----------|-------|
| Region | us-east-1 |
| AMI | ami-05e86b3611c60b0b4 (Ubuntu 22.04) |
| Datagen instance | t3.medium (spot) |
| Training instance | g4dn.xlarge (spot, 16 GB T4 GPU) |
| MPC eval instance | t3.medium (spot) |
| IAM profile | lewm-ec2-training |
| Security group | sg-03bbca875466eb52a |
| Subnet | subnet-01497e4f428a93b98 (us-east-1c) |

**Note**: us-east-1a and us-east-1b have been out of t3.medium spot capacity. Always use us-east-1c (subnet-01497e4f428a93b98).

## Scripts

### `infra/launch_datagen.sh`

Launches t3.medium spot to collect 100k transitions → S3.

```bash
bash infra/launch_datagen.sh [--n-transitions 100000]
```

Pattern:
1. Uploads `src/generate_data.py` to `s3://leworldduckie/scripts/generate_data.py`
2. EC2 installs deps, downloads script via boto3, runs it
3. HDF5 dataset streamed to S3 during collection
4. Instance terminates and shuts down

### `infra/launch_training.sh`

Launches g4dn.xlarge spot to train LeWM → S3.

```bash
bash infra/launch_training.sh [--epochs 50] [--run-id my_run]
```

Artifacts uploaded each epoch: `checkpoint_latest.pt`, `checkpoint_best.pt`, `loss_curve.png`, `metrics.jsonl`.

Monitor:
```bash
aws s3 cp s3://leworldduckie/training/runs/<run_id>/metrics.jsonl - | tail -5
aws s3 presign s3://leworldduckie/training/runs/<run_id>/loss_curve.png --expires-in 3600
```

### `infra/launch_mpc_eval.sh`

Launches t3.medium spot to run 10-episode MPC eval → S3.

```bash
bash infra/launch_mpc_eval.sh [--ckpt s3://...] [--steps N] [--map NAME]
```

Default checkpoint: `s3://leworldduckie/training/runs/notebook/checkpoint_latest.pt`

Always specify `--ckpt` explicitly to avoid using a stale default:
```bash
bash infra/launch_mpc_eval.sh --ckpt s3://leworldduckie/training/runs/notebook/checkpoint_best.pt
```

Timeline: ~10 min bootstrap, ~20 min eval (300 steps × 10 episodes × ~5ms/step on CPU).

## Colab Training

Main notebook: `lewm_duckie_run.ipynb` (or `lewm_duckie.ipynb`)

1. Open in Colab, connect to A100 GPU runtime
2. `DATA_URL` in cell 5 is a presigned S3 URL (7-day TTL, expires 2026-05-01)
   - Regenerate: `aws s3 presign s3://leworldduckie/data/duckietown_100k.h5 --expires-in 604800 --region us-east-1`
3. Run all cells
4. Checkpoint saved to `s3://leworldduckie/training/runs/notebook/checkpoint_best.pt` after each epoch (requires boto3 in pip installs — already added)

Key Colab config:
```python
FRAMESKIP  = 1
BATCH_SIZE = 512   # fills ~30 GB of 40 GB A100
N_EPOCHS   = 50
```

## Monitoring Live MPC Eval

```bash
# Check progress (updated after each episode)
aws s3 cp s3://leworldduckie/evals/mpc/<run_id>/progress.txt -

# Check if instance is still running
aws ec2 describe-instances \
  --instance-ids <instance_id> \
  --region us-east-1 \
  --query 'Reservations[0].Instances[0].State.Name' \
  --output text

# Get final results
aws s3 cp s3://leworldduckie/evals/mpc/<run_id>/results.txt -

# Download best episode GIF
aws s3 cp s3://leworldduckie/evals/mpc/<run_id>/best_episode.gif .

# Download all episode GIFs
aws s3 sync s3://leworldduckie/evals/mpc/<run_id>/ ./evals/<run_id>/ \
  --exclude '*' --include '*.gif'
```

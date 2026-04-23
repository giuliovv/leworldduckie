# LeWorldModel on Gym-Duckietown

Train [LeWM](https://arxiv.org/abs/2603.19312) (a JEPA-based world model) on duckietown lane-following data collected from the real `gym-duckietown` simulator.

## Repo layout

```
leworldduckie/
├── lewm_duckie.ipynb        # main notebook — data, training, visualisation, VoE eval
├── requirements.txt
├── src/
│   ├── generate_data.py     # EC2 data collection (real gym-duckietown, streams to HDF5)
│   ├── collect_duckietown.py# local data collection (real env or synthetic mock fallback)
│   ├── train.py             # LeWM training loop
│   └── validate_pipeline.py # end-to-end validator (no Jupyter needed)
├── infra/
│   ├── launch_datagen.sh    # launch spot t3.medium to collect 100k transitions → S3
│   └── launch_training.sh   # launch spot g4dn.xlarge to train → S3
├── checkpoints/
│   └── lewm_best.pt.zip     # latest best model weights
└── data/                    # local outputs: HDF5 dataset, plots (gitignored)
```

## Dataset

100k transitions collected on EC2 from the real `gym-duckietown` simulator (6 maps, PD lane-follower with Gaussian noise). Stored at:

```
s3://leworldduckie/data/duckietown_100k.h5   (1.87 GB)
```

Each transition: `pixels` (120×160×3 uint8) + `action` ([vel, steer] float32) + `episode_idx` + `step_idx`.

To re-collect:

```bash
bash infra/launch_datagen.sh [--n-transitions 100000]
# logs → s3://leworldduckie/logs/datagen_<run_id>.log
```

To download locally:

```bash
aws s3 cp s3://leworldduckie/data/duckietown_100k.h5 data/
```

## Quick start (Google Colab)

1. Open `lewm_duckie.ipynb` in Colab
2. Set `IS_COLAB_OVERRIDE = True` in the first cell (or it auto-detects)
3. Run all cells — the data cell downloads the dataset from S3 (~1.87 GB), then training runs for 50 epochs on GPU

Key config variables:

```python
N_TRANSITIONS = 100_000   # transitions in dataset
N_EPOCHS      = 50        # training epochs
EMBED_DIM     = 192       # ViT-Tiny hidden size
```

## Architecture

- **Encoder**: ViT-Tiny (Colab) or lightweight CNN (local)
- **Predictor**: autoregressive transformer over latent embeddings
- **Loss**: MSE prediction loss + SIGReg signature regularisation
- **Data**: PD lane-follower + Gaussian noise → (120×160×3) RGB observations, (2,) actions

## Training on EC2

```bash
bash infra/launch_training.sh [--epochs 50] [--run-id my_run]
# artifacts → s3://leworldduckie/training/runs/<run_id>/
```

Artifacts written each epoch: `loss_curve.png`, `metrics.jsonl`, `checkpoint_latest.pt`, `checkpoint_best.pt`. `summary.json` written on completion.

Monitor mid-run:

```bash
aws s3 cp s3://leworldduckie/training/runs/<run_id>/metrics.jsonl - | tail -5
aws s3 presign s3://leworldduckie/training/runs/<run_id>/loss_curve.png --expires-in 3600
```

## Local validation

```bash
python src/validate_pipeline.py
# runs 500 transitions + 2 epochs, saves plots to data/
```

## Notebook sections

1. **Setup & installs** — Colab-only pip installs, virtual display, le-wm clone
2. **Data collection** — downloads from S3 (Colab) or runs env locally
3. **Dataset & model** — HDF5 dataloader, encoder, full JEPA model
4. **Training loop** — AdamW + gradient clipping + AMP on GPU
5. **Latent space** — t-SNE (Colab) or PCA (local) on learned embeddings
6. **VoE evaluation** — inject random teleport, plot surprise signal

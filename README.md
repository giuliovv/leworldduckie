# LeWorldModel on Gym-Duckietown

Train [LeWM](https://arxiv.org/abs/2603.19312) (a JEPA-based world model) on duckietown lane-following data.

## What's in this repo

| File | Description |
|---|---|
| `lewm_duckie.ipynb` | Main notebook — data, training, visualisation, VoE eval |
| `collect_duckietown.py` | Standalone data collection script (local use) |
| `validate_pipeline.py` | End-to-end pipeline validator (runs without Jupyter) |
| `requirements.txt` | Python dependencies |
| `data/` | Local outputs: HDF5 dataset, checkpoints, plots |

## Quick start (Google Colab)

1. Open `lewm_duckie.ipynb` in Colab
2. Set `IS_COLAB_OVERRIDE = True` in the first code cell (or it auto-detects)
3. Run all cells — the data cell downloads the pre-built 100k dataset from S3 (~97 MB), then training runs for 50 epochs on GPU

Key config variables in the Config cell:

```python
N_TRANSITIONS = 100_000   # transitions in dataset
N_EPOCHS      = 50        # training epochs
EMBED_DIM     = 192       # ViT-Tiny hidden size
```

## Architecture

- **Encoder**: ViT-Tiny (Colab) or lightweight CNN (local)
- **Predictor**: Autoregressive transformer over latent embeddings
- **Loss**: MSE prediction loss + SIGReg signature regularisation
- **Data**: PD lane-follower + Gaussian noise → (120×160×3) RGB observations, (2,) actions

## Dataset

100k transitions pre-generated on EC2 using a procedural mock environment (no OpenGL needed). Stored on S3 and downloaded automatically by the notebook on Colab.

**Why no real gym-duckietown?** `duckietown-gym-daffy` depends on `zuper_typing`, which crashes on Python 3.12 (`TypeError: cannot set 'repr' attribute of immutable type 'typing.TypeVar'`). The mock environment produces equivalent synthetic observations.

### Refreshing the pre-signed URL

The S3 download URL embedded in the notebook expires after 7 days. To regenerate it:

```bash
aws s3 presign s3://test-854656252703/lewm-duckietown/duckietown_100k.h5 --expires-in 604800
```

Then update `DATA_URL` in the Config cell of `lewm_duckie.ipynb`.

To re-upload a new dataset:

```bash
aws s3 cp /path/to/duckietown_100k.h5 s3://test-854656252703/lewm-duckietown/duckietown_100k.h5
```

## Local validation (EC2)

```bash
cd ~/leworldduckie
source .venv/bin/activate
python validate_pipeline.py
```

Runs 500 transitions + 2 epochs and saves plots to `data/`. Expects `le-wm` cloned at `/home/ubuntu/le-wm`.

## Notebook sections

1. **Setup & installs** — Colab-only pip installs, virtual display, le-wm clone
2. **Data collection** — downloads from S3 (Colab) or runs mock env locally
3. **Dataset & model** — HDF5 dataloader, encoder, full JEPA model
4. **Training loop** — AdamW + gradient clipping + AMP on GPU
5. **Latent space** — t-SNE (Colab) or PCA (local) on learned embeddings
6. **VoE evaluation** — inject random teleport, plot surprise signal

## Local disk note

The `.venv` directory (~840 MB) can be deleted after use and recreated with:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Project files (notebook, scripts, data) are only ~200 KB.

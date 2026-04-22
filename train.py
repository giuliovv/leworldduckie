"""
Standalone LeWM training script for EC2 spot instance.
- Downloads dataset from S3, trains, logs metrics + plots per epoch to S3
- Resumes from latest checkpoint if found on S3
- Uploads final artifacts and shuts down when done
"""

import os, sys, time, json, argparse, subprocess
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import boto3
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from einops import rearrange
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
S3_BUCKET     = 'test-854656252703'
S3_DATA_KEY   = 'lewm-duckietown/duckietown_100k.h5'
S3_RUNS_PREFIX = 'lewm-training/runs'
S3_SCRIPT_KEY  = 'lewm-training/train.py'

LEWM_DIR  = Path('/tmp/le-wm')
DATA_PATH = Path('/tmp/duckietown_100k.h5')

IMG_H, IMG_W  = 120, 160
IMG_C         = 3
ACTION_DIM    = 2
EMBED_DIM     = 192
HISTORY       = 3
N_PREDS       = 1
SEQ_LEN       = HISTORY + N_PREDS
FRAMESKIP     = 3
N_EPOCHS      = 50
BATCH_SIZE    = 128
LR            = 5e-4
SIGREG_W      = 0.09
SEED          = 42
IMG_SIZE      = 224   # ViT-Tiny input size
CKPT_EVERY    = 5     # save checkpoint every N epochs


def log(msg):
    ts = datetime.now(timezone.utc).strftime('%H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)


# ── S3 helpers ────────────────────────────────────────────────────────────────
s3 = boto3.client('s3', region_name='us-east-1')

def s3_upload(local_path, s3_key):
    s3.upload_file(str(local_path), S3_BUCKET, s3_key)
    log(f's3 upload: {s3_key}')

def s3_download(s3_key, local_path, show_progress=False):
    size = s3.head_object(Bucket=S3_BUCKET, Key=s3_key)['ContentLength']
    log(f'Downloading s3://{S3_BUCKET}/{s3_key} ({size/1e6:.1f} MB)')
    s3.download_file(S3_BUCKET, s3_key, str(local_path))

def s3_exists(s3_key):
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
        return True
    except Exception:
        return False

def s3_put_text(s3_key, text):
    s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=text.encode())

def s3_append_jsonl(s3_key, obj):
    try:
        existing = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)['Body'].read().decode()
    except Exception:
        existing = ''
    s3.put_object(Bucket=S3_BUCKET, Key=s3_key,
                  Body=(existing + json.dumps(obj) + '\n').encode())


# ── Dataset ───────────────────────────────────────────────────────────────────
class DuckietownH5Dataset(Dataset):
    def __init__(self, path, num_steps=4, frameskip=1, img_size=None):
        self.path      = path
        self.num_steps = num_steps
        self.frameskip = frameskip
        self.img_size  = img_size

        with h5py.File(path, 'r') as f:
            self.ep_idx  = f['episode_idx'][:]
            self.actions = f['action'][:]
            self.n       = len(self.ep_idx)

        window = num_steps * frameskip
        self.valid = []
        for ep in np.unique(self.ep_idx):
            ep_inds = np.where(self.ep_idx == ep)[0]
            for start in ep_inds[:max(1, len(ep_inds) - window + 1)]:
                if start + window <= ep_inds[-1] + 1:
                    self.valid.append(start)
        self.valid = np.array(self.valid)

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        start   = self.valid[idx]
        indices = np.arange(start, start + self.num_steps * self.frameskip, self.frameskip)
        indices = np.clip(indices, 0, self.n - 1)
        with h5py.File(self.path, 'r') as f:
            frames = f['pixels'][indices]
        actions = self.actions[indices]
        pixels  = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        pixels  = pixels * 2.0 - 1.0
        if self.img_size is not None:
            pixels = F.interpolate(pixels, size=(self.img_size, self.img_size),
                                   mode='bilinear', align_corners=False)
        return {'pixels': pixels, 'action': torch.from_numpy(actions)}


# ── Encoder ───────────────────────────────────────────────────────────────────
def make_encoder(embed_dim):
    try:
        import stable_pretraining as spt
        enc = spt.backbone.utils.vit_hf(
            'tiny', patch_size=14, image_size=IMG_SIZE,
            pretrained=False, use_mask_token=False,
        )
        log('Using ViT-Tiny encoder')
        return enc, IMG_SIZE
    except Exception as e:
        log(f'ViT unavailable ({e}), using CNN encoder')

    class CNNEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(64, embed_dim, 3, stride=2, padding=1), nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
            )
        def forward(self, pixel_values, **kw):
            b = pixel_values.size(0)
            feat = self.net(pixel_values).view(b, -1)
            return type('Out', (), {'last_hidden_state': feat.unsqueeze(1)})()
    return CNNEncoder(), None


# ── Training step ─────────────────────────────────────────────────────────────
def step_fn(batch, model, sigreg, device, dtype, lam=SIGREG_W):
    batch = {k: v.to(device, dtype=dtype if v.is_floating_point() else v.dtype)
             for k, v in batch.items()}
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)
    out     = model.encode(batch)
    emb     = out['emb']
    act_emb = out['act_emb']
    ctx_emb = emb[:, :HISTORY]
    ctx_act = act_emb[:, :HISTORY]
    tgt_emb = emb[:, N_PREDS:]
    pred    = model.predict(ctx_emb, ctx_act)
    pred_loss = (pred - tgt_emb).pow(2).mean()
    sig_loss  = sigreg(emb.transpose(0, 1))
    loss      = pred_loss + lam * sig_loss
    return loss, pred_loss.item(), sig_loss.item()


# ── Plot helpers ──────────────────────────────────────────────────────────────
def save_loss_plot(train_losses, val_losses, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ep = range(1, len(train_losses) + 1)
    ax.plot(ep, train_losses, label='train', marker='o', markersize=3)
    ax.plot(ep, val_losses,   label='val',   marker='s', markersize=3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('LeWM Training Loss')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--epochs', type=int, default=N_EPOCHS)
    args = parser.parse_args()

    run_id = args.run_id
    run_prefix = f'{S3_RUNS_PREFIX}/{run_id}'
    local_run  = Path(f'/tmp/run_{run_id}')
    local_run.mkdir(parents=True, exist_ok=True)

    log(f'Run ID: {run_id}')
    log(f'S3 prefix: s3://{S3_BUCKET}/{run_prefix}/')

    # ── Clone le-wm ──────────────────────────────────────────────────────────
    if not LEWM_DIR.exists():
        log('Cloning le-wm ...')
        r = subprocess.run(['git', 'clone', '--depth', '1',
                            'https://github.com/lucas-maes/le-wm.git', str(LEWM_DIR)])
        if r.returncode != 0:
            raise RuntimeError('git clone le-wm failed')
    if str(LEWM_DIR) not in sys.path:
        sys.path.insert(0, str(LEWM_DIR))
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP, SIGReg

    # ── Download dataset ──────────────────────────────────────────────────────
    if not DATA_PATH.exists():
        s3_download(S3_DATA_KEY, DATA_PATH)
    else:
        log(f'Dataset already at {DATA_PATH}')

    # ── Dataset / dataloaders ─────────────────────────────────────────────────
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.bfloat16 if device.type == 'cuda' else torch.float32
    log(f'Device: {device}  dtype: {dtype}')

    encoder, img_size = make_encoder(EMBED_DIM)
    full_ds   = DuckietownH5Dataset(DATA_PATH, num_steps=SEQ_LEN,
                                     frameskip=FRAMESKIP, img_size=img_size)
    n_train   = int(0.9 * len(full_ds))
    n_val     = len(full_ds) - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               drop_last=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                               drop_last=False, num_workers=4, pin_memory=True)
    log(f'Dataset: {len(train_ds)} train / {len(val_ds)} val samples')

    # ── Build model ───────────────────────────────────────────────────────────
    hidden_dim     = EMBED_DIM
    projector      = MLP(EMBED_DIM, hidden_dim, EMBED_DIM)
    pred_proj      = MLP(hidden_dim, hidden_dim, EMBED_DIM)
    action_encoder = Embedder(input_dim=ACTION_DIM, smoothed_dim=ACTION_DIM,
                               emb_dim=EMBED_DIM, mlp_scale=4)
    predictor      = ARPredictor(num_frames=HISTORY, input_dim=EMBED_DIM,
                                  hidden_dim=hidden_dim, output_dim=EMBED_DIM,
                                  depth=2, heads=4, dim_head=16,
                                  mlp_dim=hidden_dim * 2, dropout=0.1)
    sigreg = SIGReg(knots=17, num_proj=512)
    model  = JEPA(encoder, predictor, action_encoder, projector, pred_proj)
    model  = model.to(device)
    sigreg = sigreg.to(device)
    log(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(sigreg.parameters()),
        lr=LR, weight_decay=1e-3,
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch   = 1
    train_losses  = []
    val_losses    = []
    best_val      = float('inf')
    ckpt_s3_key   = f'{run_prefix}/checkpoint_latest.pt'
    metrics_key   = f'{run_prefix}/metrics.jsonl'

    if s3_exists(ckpt_s3_key):
        log('Resuming from checkpoint ...')
        local_ckpt = local_run / 'checkpoint_latest.pt'
        s3_download(ckpt_s3_key, local_ckpt)
        ckpt = torch.load(local_ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
        sigreg.load_state_dict(ckpt['sigreg'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch  = ckpt['epoch'] + 1
        train_losses = ckpt.get('train_losses', [])
        val_losses   = ckpt.get('val_losses', [])
        best_val     = ckpt.get('best_val', float('inf'))
        log(f'Resumed from epoch {ckpt["epoch"]}  best_val={best_val:.4f}')

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep_train = []
        for batch in train_loader:
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    loss, pl, sl = step_fn(batch, model, sigreg, device, dtype)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, pl, sl = step_fn(batch, model, sigreg, device, dtype)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            ep_train.append(loss.item())

        model.eval()
        ep_val = []
        with torch.no_grad():
            for batch in val_loader:
                loss, _, _ = step_fn(batch, model, sigreg, device, dtype)
                ep_val.append(loss.item())

        t_loss = float(np.mean(ep_train))
        v_loss = float(np.mean(ep_val))
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        elapsed = time.time() - t0
        log(f'Epoch {epoch:3d}/{args.epochs}  train={t_loss:.4f}  val={v_loss:.4f}  {elapsed:.0f}s')

        # append metrics to S3
        s3_append_jsonl(metrics_key, {
            'epoch': epoch, 'train_loss': t_loss, 'val_loss': v_loss,
            'ts': datetime.now(timezone.utc).isoformat(),
        })

        # save loss plot to S3 every epoch
        plot_path = local_run / 'loss_curve.png'
        save_loss_plot(train_losses, val_losses, plot_path)
        s3_upload(plot_path, f'{run_prefix}/loss_curve.png')

        # save checkpoint periodically and on best val
        is_best = v_loss < best_val
        if is_best:
            best_val = v_loss

        if epoch % CKPT_EVERY == 0 or is_best or epoch == args.epochs:
            ckpt_data = {
                'epoch': epoch, 'model': model.state_dict(),
                'sigreg': sigreg.state_dict(), 'optimizer': optimizer.state_dict(),
                'train_losses': train_losses, 'val_losses': val_losses,
                'best_val': best_val,
            }
            local_ckpt = local_run / 'checkpoint_latest.pt'
            torch.save(ckpt_data, local_ckpt)
            s3_upload(local_ckpt, ckpt_s3_key)
            if is_best:
                s3_upload(local_ckpt, f'{run_prefix}/checkpoint_best.pt')

    # ── Final summary to S3 ───────────────────────────────────────────────────
    summary = {
        'run_id': run_id,
        'epochs': args.epochs,
        'best_val': best_val,
        'final_train': train_losses[-1],
        'final_val': val_losses[-1],
        'completed_at': datetime.now(timezone.utc).isoformat(),
    }
    s3_put_text(f'{run_prefix}/summary.json', json.dumps(summary, indent=2))
    log(f'Training complete. Best val loss: {best_val:.4f}')
    log(f'Artifacts: s3://{S3_BUCKET}/{run_prefix}/')


if __name__ == '__main__':
    main()

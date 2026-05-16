#!/usr/bin/env python3
"""Probe obs->action predictability for Duckietown datasets.

Default mode is encoder-latent (official gate):
  frozen LeWM encoder/projector -> z, then MLP(z)->action
Optional secondary mode:
  small CNN directly from pixels -> action
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def resolve_local_or_s3(path: str) -> str:
    if path.startswith('s3://'):
        import boto3
        from urllib.parse import urlparse
        u = urlparse(path)
        local = Path('/tmp') / Path(u.path).name
        if not local.exists():
            boto3.client('s3', region_name='us-east-1').download_file(
                u.netloc, u.path.lstrip('/'), str(local)
            )
        return str(local)
    return os.path.expanduser(path)


def ensure_lewm(lewm_dir: str) -> None:
    p = Path(lewm_dir)
    if not p.exists():
        raise FileNotFoundError(f'le-wm dir not found: {lewm_dir}')
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def extract_encoder_tensor(emb):
    if torch.is_tensor(emb):
        return emb
    if hasattr(emb, 'pooler_output') and emb.pooler_output is not None:
        return emb.pooler_output
    if hasattr(emb, 'last_hidden_state') and emb.last_hidden_state is not None:
        return emb.last_hidden_state[:, 0]
    if isinstance(emb, dict):
        if emb.get('pooler_output', None) is not None:
            return emb['pooler_output']
        if emb.get('last_hidden_state', None) is not None:
            return emb['last_hidden_state'][:, 0]
    if isinstance(emb, (tuple, list)) and len(emb) > 0 and torch.is_tensor(emb[0]):
        return emb[0][:, 0] if emb[0].dim() == 3 else emb[0]
    raise TypeError(f'Unsupported encoder output type: {type(emb)}')


def load_jepa(ckpt_path: str, lewm_dir: str, device: torch.device):
    ensure_lewm(lewm_dir)
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    jepa = getattr(obj, 'model', obj)
    jepa = jepa.to(device)
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)
    return jepa


def load_pixels_actions(h5_path: str, max_samples: int, seed: int):
    with h5py.File(h5_path, 'r') as f:
        px = f['pixels'][:]
        act = f['action'][:]

    n = len(px)
    if max_samples > 0 and max_samples < n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_samples, replace=False)
        px = px[idx]
        act = act[idx]

    x = torch.from_numpy(px.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    y = torch.from_numpy(act.astype(np.float32))
    return x, y


@torch.no_grad()
def encode_to_latent(jepa, x, device: torch.device, img_size: int, batch_size: int):
    zs = []
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i+batch_size].to(device)
        if xb.shape[-2:] != (img_size, img_size):
            xb = F.interpolate(xb, size=(img_size, img_size), mode='bilinear', align_corners=False)
        xb = (xb - mean) / std
        emb = jepa.encoder(xb, interpolate_pos_encoding=True)
        emb = extract_encoder_tensor(emb)
        z = jepa.projector(emb)
        zs.append(z.cpu())
    return torch.cat(zs, dim=0)


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


def split_indices(n: int, seed: int, train_frac: float = 0.8):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)
    split = int(train_frac * n)
    return idx[:split], idx[split:]


def fit_probe(model, xtr, ytr, xva, yva, epochs: int, batch_size: int, lr: float):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(xtr.shape[0])
        total = 0.0
        for i in range(0, xtr.shape[0], batch_size):
            b = perm[i:i+batch_size]
            pred = model(xtr[b])
            loss = loss_fn(pred, ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(b)
        print(f'epoch {ep+1}/{epochs} train_mse={total/xtr.shape[0]:.6f}')

    model.eval()
    with torch.no_grad():
        p_tr = model(xtr).cpu().numpy()
        p_va = model(xva).cpu().numpy()
    t_tr = ytr.cpu().numpy()
    t_va = yva.cpu().numpy()

    out = {
        'train_r2_vel': r2_score(t_tr[:, 0], p_tr[:, 0]),
        'train_r2_steer': r2_score(t_tr[:, 1], p_tr[:, 1]),
        'val_r2_vel': r2_score(t_va[:, 0], p_va[:, 0]),
        'val_r2_steer': r2_score(t_va[:, 1], p_va[:, 1]),
    }
    return out


def run_one_dataset(
    data_path: str,
    mode: str,
    seed: int,
    max_samples: int,
    epochs: int,
    batch_size: int,
    lr: float,
    ckpt_path: str,
    lewm_dir: str,
    img_size: int,
    encode_batch_size: int,
):
    print(f'\n=== Dataset: {data_path} ===')
    x, y = load_pixels_actions(data_path, max_samples=max_samples, seed=seed)
    tr, va = split_indices(x.shape[0], seed=seed)

    if mode == 'encoder':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        jepa = load_jepa(ckpt_path, lewm_dir, device)
        z = encode_to_latent(jepa, x, device=device, img_size=img_size, batch_size=encode_batch_size)
        xtr, xva = z[tr], z[va]
        ytr, yva = y[tr], y[va]
        model = MLPProbe(in_dim=xtr.shape[1])
        result = fit_probe(model, xtr, ytr, xva, yva, epochs=epochs, batch_size=batch_size, lr=lr)
    elif mode == 'cnn':
        model = SmallCNN()
        xtr, xva = x[tr], x[va]
        ytr, yva = y[tr], y[va]
        result = fit_probe(model, xtr, ytr, xva, yva, epochs=epochs, batch_size=batch_size, lr=lr)
    else:
        raise ValueError(f'Unknown mode: {mode}')

    print('obs -> action probe R²:')
    print(f"  train velocity: {result['train_r2_vel']:.4f}")
    print(f"  train steering: {result['train_r2_steer']:.4f}")
    print(f"  val velocity:   {result['val_r2_vel']:.4f}")
    print(f"  val steering:   {result['val_r2_steer']:.4f}")

    s = result['val_r2_steer']
    if s < 0.6:
        print('Decision: PASS (steering val R² < 0.6).')
    elif s <= 0.8:
        print('Decision: MARGINAL (0.6-0.8). Increase steer noise std to 0.45 and recollect.')
    else:
        print('Decision: FAIL (>0.8). Increase steer noise std to 0.6 and recollect.')

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='New exploratory dataset path (local or s3://...)')
    ap.add_argument('--baseline-data', default=None, help='Optional old dataset path to compare in same run')
    ap.add_argument('--mode', choices=['encoder', 'cnn'], default='encoder')
    ap.add_argument('--ckpt', default=None, help='LeWM checkpoint path for --mode encoder')
    ap.add_argument('--lewm-dir', default='/tmp/le-wm')
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--encode-batch-size', type=int, default=512)
    ap.add_argument('--max-samples', type=int, default=50000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()

    if args.mode == 'encoder' and not args.ckpt:
        raise SystemExit('--ckpt is required in --mode encoder')

    data_main = resolve_local_or_s3(args.data)
    ckpt = resolve_local_or_s3(args.ckpt) if args.ckpt else None
    run_one_dataset(
        data_main,
        mode=args.mode,
        seed=args.seed,
        max_samples=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        ckpt_path=ckpt,
        lewm_dir=args.lewm_dir,
        img_size=args.img_size,
        encode_batch_size=args.encode_batch_size,
    )

    if args.baseline_data:
        data_base = resolve_local_or_s3(args.baseline_data)
        run_one_dataset(
            data_base,
            mode=args.mode,
            seed=args.seed,
            max_samples=args.max_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            ckpt_path=ckpt,
            lewm_dir=args.lewm_dir,
            img_size=args.img_size,
            encode_batch_size=args.encode_batch_size,
        )


if __name__ == '__main__':
    main()

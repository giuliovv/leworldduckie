#!/usr/bin/env python3
"""Probe obs->action predictability to verify exploration decorrelation.

Default target gate (steering):
- < 0.6  : success, proceed to retraining
- 0.6-0.8: increase steer noise to 0.45 and recollect
- > 0.8  : increase steer noise to 0.6 and recollect
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score


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
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.head(self.net(x))


def load_data(h5_path: str, max_samples: int, seed: int):
    with h5py.File(h5_path, 'r') as f:
        px = f['pixels'][:]
        act = f['action'][:]
    n = len(px)
    if max_samples > 0 and max_samples < n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_samples, replace=False)
        px = px[idx]
        act = act[idx]

    # uint8 HWC -> float CHW
    x = torch.from_numpy(px.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    y = torch.from_numpy(act.astype(np.float32))
    return x, y


def train_probe(x, y, epochs=6, batch_size=256, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    n = x.shape[0]
    idx = torch.randperm(n)
    split = int(0.8 * n)
    tr, va = idx[:split], idx[split:]

    xtr, ytr = x[tr], y[tr]
    xva, yva = x[va], y[va]

    model = SmallCNN()
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        perm = torch.randperm(xtr.shape[0])
        total = 0.0
        for i in range(0, xtr.shape[0], batch_size):
            b = perm[i:i+batch_size]
            xb, yb = xtr[b], ytr[b]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(b)
        print(f'epoch {ep+1}/{epochs} train_mse={total/xtr.shape[0]:.5f}')

    model.eval()
    with torch.no_grad():
        p = model(xva).cpu().numpy()
    t = yva.cpu().numpy()
    r2_vel = r2_score(t[:, 0], p[:, 0])
    r2_steer = r2_score(t[:, 1], p[:, 1])
    return r2_vel, r2_steer, xtr.shape[0], xva.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--max-samples', type=int, default=50000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()

    x, y = load_data(args.data, args.max_samples, args.seed)
    r2_vel, r2_steer, ntr, nva = train_probe(
        x, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    print('\nobs -> action probe R²:')
    print(f'  velocity: {r2_vel:.4f}')
    print(f'  steering: {r2_steer:.4f}')
    print(f'  split: train={ntr}, val={nva}')

    if r2_steer < 0.6:
        print('Decision: PASS (steering R² < 0.6). Proceed to retraining.')
    elif r2_steer <= 0.8:
        print('Decision: MARGINAL (0.6-0.8). Increase steer noise std to 0.45 and recollect.')
    else:
        print('Decision: FAIL (>0.8). Increase steer noise std to 0.6 and recollect.')


if __name__ == '__main__':
    main()

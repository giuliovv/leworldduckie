#!/usr/bin/env python3
"""
Model-level diagnostics for LeWM.

T4  Rollout error vs horizon (k=1..15)
    For each k, compute mean L2(predictor_rollout[-1], z_{t+k}) using pre-computed
    latent index and true actions from HDF5. Shows at what horizon the predictor
    breaks down. Context is FRAMESKIP=1 (matching MPC operating condition).

T5  Linear probe: z → steering  (proxy for lane offset)
    Train a 2-layer MLP on z from the latent index with action[:, 1] (steering)
    as the regression target. Steering ≈ 10*lane_dist + 5*heading_angle, so R²
    here lower-bounds what R² on true lane_dist would be.
    Report R² on a 20% holdout.

Neither test changes any model weights.

Usage:
  # Full run (T4 + T5):
  python src/diagnostic_model.py \\
      --ckpt s3://leworldduckie/training/runs/colab_v1/checkpoint_best.pt \\
      --latent-index s3://leworldduckie/evals/latent_index.npz \\
      --data-path s3://leworldduckie/data/duckietown_100k.h5 \\
      --s3-output s3://leworldduckie/diagnostics/

  # T5 only (no checkpoint needed):
  python src/diagnostic_model.py --skip-t4 \\
      --latent-index s3://leworldduckie/evals/latent_index.npz \\
      --data-path s3://leworldduckie/data/duckietown_100k.h5 \\
      --s3-output s3://leworldduckie/diagnostics/

  # Synthetic test (no HDF5 or checkpoint needed — verifies code only):
  python src/diagnostic_model.py --test-mode
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Constants (must match train.py / mpc_controller.py) ───────────────────────
EMBED_DIM  = 192
ACTION_DIM = 2
HISTORY    = 3
IMG_SIZE   = 224
LEWM_DIR   = Path('/home/ubuntu/le-wm') if Path('/home/ubuntu/le-wm').exists() \
             else Path('/tmp/le-wm')


# ── S3 helpers ─────────────────────────────────────────────────────────────────

def _s3_download(s3_uri: str, local: str) -> str:
    import boto3
    u = urlparse(s3_uri)
    print(f'Downloading {s3_uri} ...')
    boto3.client('s3', region_name='us-east-1').download_file(
        u.netloc, u.path.lstrip('/'), local)
    print(f'  → {local}  ({os.path.getsize(local) / 1e6:.1f} MB)')
    return local


def _s3_upload(local: str, s3_uri: str):
    import boto3
    u = urlparse(s3_uri)
    boto3.client('s3', region_name='us-east-1').upload_file(
        local, u.netloc, u.path.lstrip('/'))
    print(f'Uploaded {local} → {s3_uri}')


def resolve_local(path: str, tmp_name: str) -> str:
    if path.startswith('s3://'):
        local = f'/tmp/{tmp_name}'
        if not os.path.exists(local):
            _s3_download(path, local)
        return local
    return path


# ── Latent index ───────────────────────────────────────────────────────────────

def load_index(path: str):
    local = resolve_local(path, 'latent_index.npz')
    d = np.load(local)
    all_z    = d['all_z'].astype(np.float32)
    ep_idx   = d['ep_idx'].astype(np.int32)
    step_idx = d['step_idx'].astype(np.int32)
    print(f'Latent index: {all_z.shape[0]:,} frames, D={all_z.shape[1]}, '
          f'{len(np.unique(ep_idx))} episodes')
    return all_z, ep_idx, step_idx


def build_ep_step_map(ep_idx, step_idx):
    """ep → {step → global_index}"""
    ep_map = defaultdict(dict)
    for gi, (ep, step) in enumerate(zip(ep_idx.tolist(), step_idx.tolist())):
        ep_map[ep][step] = gi
    return ep_map


# ── HDF5 actions (small: ~1.6 MB) ─────────────────────────────────────────────

def load_hdf5_actions(hdf5_path: str):
    """Returns (actions (N,2), ep_h5 (N,), step_h5 (N,))."""
    import h5py
    local = resolve_local(hdf5_path, 'duckietown_100k.h5')
    with h5py.File(local, 'r') as f:
        actions  = f['action'][:]      # (N, 2)
        ep_h5    = f['episode_idx'][:] # (N,)
        step_h5  = f['step_idx'][:]    # (N,)
    print(f'HDF5 actions: {len(actions):,} rows')
    return actions.astype(np.float32), ep_h5.astype(np.int32), step_h5.astype(np.int32)


def build_action_lookup(actions, ep_h5, step_h5):
    """(ep, step) → action (2,)"""
    lookup = {}
    for i, (ep, step) in enumerate(zip(ep_h5.tolist(), step_h5.tolist())):
        lookup[(ep, step)] = actions[i]
    return lookup


# ── Model loading (mirrors mpc_controller.py) ──────────────────────────────────

def _ensure_lewm():
    import subprocess
    if not LEWM_DIR.exists():
        print(f'Cloning le-wm → {LEWM_DIR}')
        r = subprocess.run(['git', 'clone', '--depth', '1',
                            'https://github.com/lucas-maes/le-wm.git', str(LEWM_DIR)])
        if r.returncode != 0:
            raise RuntimeError('git clone le-wm failed')
    if str(LEWM_DIR) not in sys.path:
        sys.path.insert(0, str(LEWM_DIR))


def load_model(ckpt_path: str, device):
    import torch
    import torch.nn as nn
    _ensure_lewm()
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    import stable_pretraining as spt
    encoder = spt.backbone.utils.vit_hf(
        'tiny', patch_size=14, image_size=IMG_SIZE,
        pretrained=False, use_mask_token=False)

    projector      = MLP(EMBED_DIM, 2048, EMBED_DIM, norm_fn=nn.BatchNorm1d)
    pred_proj      = MLP(EMBED_DIM, 2048, EMBED_DIM, norm_fn=nn.BatchNorm1d)
    action_encoder = Embedder(input_dim=ACTION_DIM, smoothed_dim=ACTION_DIM,
                               emb_dim=EMBED_DIM, mlp_scale=4)
    predictor      = ARPredictor(num_frames=HISTORY, input_dim=EMBED_DIM,
                                  hidden_dim=EMBED_DIM, output_dim=EMBED_DIM,
                                  depth=6, heads=16, dim_head=64,
                                  mlp_dim=2048, dropout=0.1)
    model = JEPA(encoder, predictor, action_encoder, projector, pred_proj).to(device)

    local = ckpt_path
    if ckpt_path.startswith('s3://'):
        import boto3
        u = urlparse(ckpt_path)
        local = f'/tmp/{Path(u.path).name}'
        if not os.path.exists(local):
            print(f'Downloading checkpoint {ckpt_path} ...')
            boto3.client('s3', region_name='us-east-1').download_file(
                u.netloc, u.path.lstrip('/'), local)

    ckpt = torch.load(local, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model', ckpt), strict=True)
    model.eval()
    print(f'Loaded checkpoint: {ckpt_path}')
    return model


# ── T4: Rollout error vs horizon ───────────────────────────────────────────────

def _sample_sequences(ep_map, total_len: int, n_seqs: int, rng):
    """Sample n_seqs tuples of global indices with total_len consecutive steps."""
    episodes = list(ep_map.keys())
    seqs = []
    attempts = 0
    while len(seqs) < n_seqs and attempts < n_seqs * 100:
        ep = int(rng.choice(episodes))
        steps = sorted(ep_map[ep].keys())
        if len(steps) < total_len:
            attempts += 1
            continue
        si = int(rng.integers(0, len(steps) - total_len + 1))
        base = steps[si]
        needed = [base + j for j in range(total_len)]
        if all(s in ep_map[ep] for s in needed):
            seqs.append([ep_map[ep][s] for s in needed])
        attempts += 1
    return seqs


def test_rollout_error(model, all_z, ep_idx, step_idx, action_lookup,
                       max_horizon: int = 15, n_seqs: int = 300, rng=None):
    """
    For k=1..max_horizon, roll out k steps from a HISTORY-frame context using true
    actions, then compare the predicted latent against the ground-truth latent.

    Context is FRAMESKIP=1 (every raw step), matching the MPC operating condition.
    Training used FRAMESKIP=3; this measures generalisation to FRAMESKIP=1.
    """
    import torch
    import torch.nn.functional as F

    if rng is None:
        rng = np.random.default_rng(42)

    device = next(model.parameters()).device
    ep_map = build_ep_step_map(ep_idx, step_idx)

    total_len = HISTORY + max_horizon
    seqs = _sample_sequences(ep_map, total_len, n_seqs, rng)
    print(f'T4: sampled {len(seqs)} sequences (len={total_len})')

    if not seqs:
        return 'T4: could not sample sequences — skipped.', {}

    errors_per_k = {k: [] for k in range(1, max_horizon + 1)}

    model.eval()
    with torch.no_grad():
        for gi_seq in seqs:
            ep   = int(ep_idx[gi_seq[0]])
            # Get actions for all positions in this sequence.
            # action_lookup[(ep, step)] → (2,) float32.
            steps_in_seq = step_idx[np.array(gi_seq)].tolist()
            acts_np = np.array([
                action_lookup.get((ep, int(s)), np.array([0.35, 0.0]))
                for s in steps_in_seq
            ], dtype=np.float32)  # (total_len, 2)

            acts_t = torch.from_numpy(acts_np).to(device)          # (total_len, 2)
            act_embs = model.action_encoder(acts_t.unsqueeze(0))[0] # (total_len, D)

            # Initial context window
            ctx_z = torch.from_numpy(
                all_z[np.array(gi_seq[:HISTORY])]
            ).to(device)  # (HISTORY, D)

            emb_win = ctx_z.unsqueeze(0)               # (1, HISTORY, D)
            act_win = act_embs[:HISTORY - 1].unsqueeze(0)  # (1, HISTORY-1, D)

            for k in range(1, max_horizon + 1):
                # Action at position HISTORY-1 + (k-1) drives the k-th prediction
                a_k = act_embs[HISTORY - 1 + k - 1].unsqueeze(0).unsqueeze(0)  # (1, 1, D)
                full_act = torch.cat([act_win, a_k], dim=1)     # (1, HISTORY, D)
                z_pred = model.predict(emb_win, full_act)[:, -1] # (1, D)

                # Ground truth from latent index
                gi_target = gi_seq[HISTORY + k - 1]
                z_true = torch.from_numpy(all_z[gi_target]).to(device).unsqueeze(0)

                err = (z_pred - z_true).norm(dim=-1).item()
                errors_per_k[k].append(err)

                # Slide windows for next iteration
                emb_win = torch.cat([emb_win[:, 1:], z_pred.unsqueeze(1)], dim=1)
                act_win = full_act[:, 1:]

    mean_err = [float(np.mean(errors_per_k[k])) for k in range(1, max_horizon + 1)]
    std_err  = [float(np.std(errors_per_k[k]))  for k in range(1, max_horizon + 1)]

    lines = [
        '── T4: Rollout Error vs Horizon ─────────────────────────────',
        f'  Sequences     : {len(seqs)}',
        f'  Context FRAMESKIP: 1 (MPC operating condition; training FRAMESKIP=3)',
        '',
        '  k   mean_L2   ±std',
    ]
    for k, (m, s) in enumerate(zip(mean_err, std_err), 1):
        lines.append(f'  {k:2d}   {m:.3f}    ±{s:.3f}')

    # Find horizon where error doubles vs k=1
    if mean_err[0] > 0:
        double_k = next((k for k, e in enumerate(mean_err, 1)
                         if e >= 2 * mean_err[0]), max_horizon)
    else:
        double_k = max_horizon

    lines += ['', f'  Error doubles at k ≈ {double_k}']

    return '\n'.join(lines), {
        'horizons': list(range(1, max_horizon + 1)),
        'mean_err': mean_err,
        'std_err':  std_err,
    }


def plot_rollout_error(data: dict, out_dir: str) -> str:
    horizons = data['horizons']
    mean_err = data['mean_err']
    std_err  = data['std_err']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(horizons, mean_err, yerr=std_err, fmt='o-', capsize=4,
                color='steelblue', label='mean L2 ± std')
    ax.axhline(mean_err[0], linestyle='--', color='gray', alpha=0.5, label='k=1 baseline')
    ax.axhline(2 * mean_err[0], linestyle=':', color='red', alpha=0.5, label='2× baseline')
    ax.set_xlabel('Rollout horizon k (model steps)')
    ax.set_ylabel('L2(predicted z, true z)')
    ax.set_title('LeWM predictor rollout error vs horizon')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, 'rollout_error.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f'  Saved {out}')
    return out


# ── T5: Linear probe for lane offset ──────────────────────────────────────────

class _MLP(object):
    """2-layer PyTorch MLP with training loop."""

    def __init__(self, in_dim: int, hidden: int = 256):
        import torch
        import torch.nn as nn
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 1),
        ).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 80,
            batch: int = 512, val_frac: float = 0.2):
        import torch
        import torch.nn.functional as F

        rng = np.random.default_rng(0)
        n = len(X)
        idx = rng.permutation(n)
        n_val = int(n * val_frac)
        tr_idx, val_idx = idx[n_val:], idx[:n_val]

        X_tr = torch.from_numpy(X[tr_idx]).float().to(self.device)
        y_tr = torch.from_numpy(y[tr_idx]).float().to(self.device)
        X_val = torch.from_numpy(X[val_idx]).float().to(self.device)
        y_val = torch.from_numpy(y[val_idx]).float().to(self.device)

        best_val_r2 = -np.inf
        for ep in range(1, epochs + 1):
            self.net.train()
            perm = torch.randperm(len(X_tr), device=self.device)
            for i in range(0, len(X_tr), batch):
                bi = perm[i:i + batch]
                pred = self.net(X_tr[bi]).squeeze(-1)
                loss = F.mse_loss(pred, y_tr[bi])
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            if ep % 20 == 0 or ep == epochs:
                self.net.eval()
                with torch.no_grad():
                    pv = self.net(X_val).squeeze(-1)
                ss_res = ((pv - y_val) ** 2).sum().item()
                ss_tot = ((y_val - y_val.mean()) ** 2).sum().item()
                r2 = 1 - ss_res / max(ss_tot, 1e-8)
                best_val_r2 = max(best_val_r2, r2)
                print(f'  epoch {ep:3d}/{epochs}  val R² = {r2:.4f}')

        # Final val R²
        self.net.eval()
        with torch.no_grad():
            pv = self.net(X_val).squeeze(-1)
        ss_res = ((pv - y_val) ** 2).sum().item()
        ss_tot = ((y_val - y_val.mean()) ** 2).sum().item()
        final_r2 = 1 - ss_res / max(ss_tot, 1e-8)
        return final_r2, X_val.cpu().numpy(), y_val.cpu().numpy(), pv.cpu().numpy()


def test_linear_probe(all_z, ep_idx, step_idx, action_lookup,
                      epochs: int = 80, rng=None):
    """
    Regress z → steering.  Steering = 10*lane_dist + 5*angle + noise, so R² here
    lower-bounds R² for true lane offset.  We use action[:, 1] as the label.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Build (z, steering) pairs for all latent index frames that have an action.
    z_list, y_list = [], []
    for gi in range(len(all_z)):
        ep   = int(ep_idx[gi])
        step = int(step_idx[gi])
        act  = action_lookup.get((ep, step))
        if act is not None:
            z_list.append(all_z[gi])
            y_list.append(float(act[1]))  # steering

    X = np.array(z_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f'T5: {len(X):,} (z, steering) pairs')

    mlp = _MLP(in_dim=X.shape[1])
    r2, X_val, y_val, y_pred = mlp.fit(X, y, epochs=epochs)

    lines = [
        '── T5: Linear Probe (z → steering as lane-offset proxy) ─────',
        f'  Samples          : {len(X):,}',
        f'  Target           : action[:, 1] (steering)',
        f'  Target label     : steer = 10·dist + 5·angle + noise',
        f'  Architecture     : 192 → 256 → 64 → 1, Adam lr=1e-3',
        f'  Epochs           : {epochs}',
        '',
        f'  Val R²           : {r2:.4f}',
        '',
        '  Interpretation:',
    ]

    if r2 > 0.5:
        lines.append(f'  → R² = {r2:.2f}  Encoder captures steering/lane-offset signal. Good.')
    elif r2 > 0.2:
        lines.append(f'  → R² = {r2:.2f}  Weak but non-trivial correlation with lane offset.')
    else:
        lines.append(f'  → R² = {r2:.2f}  Encoder does not represent lane offset.')

    return '\n'.join(lines), {'r2': r2, 'y_val': y_val, 'y_pred': y_pred}


def plot_probe(data: dict, out_dir: str) -> str:
    y_val  = data['y_val']
    y_pred = data['y_pred']
    r2     = data['r2']

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_val, y_pred, s=3, alpha=0.3, color='steelblue')
    lo, hi = min(y_val.min(), y_pred.min()), max(y_val.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1, label='perfect')
    ax.set_xlabel('True steering (proxy for lane offset)')
    ax.set_ylabel('MLP prediction from z')
    ax.set_title(f'Linear probe: z → steering  R² = {r2:.4f}')
    ax.legend()
    fig.tight_layout()
    out = os.path.join(out_dir, 'linear_probe.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f'  Saved {out}')
    return out


# ── Synthetic test mode (no data needed) ──────────────────────────────────────

def _synthetic_test():
    """Smoke-test T5 MLP code with random z and sinusoidal labels."""
    print('=== Synthetic test mode (no HDF5 / checkpoint needed) ===')
    rng = np.random.default_rng(0)
    N = 5000
    all_z = rng.standard_normal((N, EMBED_DIM)).astype(np.float32)
    # Fake label: linear function of first 3 dims (verifies the MLP can learn it)
    y_fake = 0.5 * all_z[:, 0] + 0.3 * all_z[:, 1] - 0.2 * all_z[:, 2]
    action_lookup = {(0, i): np.array([0.35, float(y_fake[i])], dtype=np.float32)
                     for i in range(N)}
    ep_idx   = np.zeros(N, dtype=np.int32)
    step_idx = np.arange(N, dtype=np.int32)

    out_dir = '/tmp/diag_test'
    os.makedirs(out_dir, exist_ok=True)

    t5_text, t5_data = test_linear_probe(
        all_z, ep_idx, step_idx, action_lookup, epochs=60)
    print(t5_text)

    if t5_data:
        plot_probe(t5_data, out_dir)
        print(f'Plot saved to {out_dir}/linear_probe.png')

    expected_r2 = 0.9
    actual_r2   = t5_data['r2']
    assert actual_r2 > 0.85, \
        f'Synthetic test FAILED: R² = {actual_r2:.3f} < 0.85 (expected > {expected_r2})'
    print(f'\nSynthetic test PASSED — R² = {actual_r2:.4f} on linear target')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',
                    default='s3://leworldduckie/training/runs/colab_v1/checkpoint_best.pt')
    ap.add_argument('--latent-index',
                    default='s3://leworldduckie/evals/latent_index.npz')
    ap.add_argument('--data-path',
                    default='s3://leworldduckie/data/duckietown_100k.h5')
    ap.add_argument('--max-horizon',  type=int, default=15)
    ap.add_argument('--n-seqs',       type=int, default=300)
    ap.add_argument('--probe-epochs', type=int, default=80)
    ap.add_argument('--output-dir',   default='/tmp/diag_model')
    ap.add_argument('--s3-output',    default=None,
                    help='s3://bucket/prefix/ to upload results')
    ap.add_argument('--skip-t4',      action='store_true',
                    help='Skip rollout error test (no checkpoint needed)')
    ap.add_argument('--test-mode',    action='store_true',
                    help='Run on synthetic data only — no HDF5 or checkpoint needed')
    args = ap.parse_args()

    if args.test_mode:
        _synthetic_test()
        return

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    # Load latent index
    all_z, ep_idx, step_idx = load_index(args.latent_index)

    # Load HDF5 actions (small: ~1.6 MB)
    actions_h5, ep_h5, step_h5 = load_hdf5_actions(args.data_path)
    action_lookup = build_action_lookup(actions_h5, ep_h5, step_h5)
    print(f'Action lookup: {len(action_lookup):,} entries')

    report_lines = [
        '═' * 62,
        'LeWM Model Diagnostic  (T4: rollout  +  T5: linear probe)',
        f'Checkpoint  : {args.ckpt}',
        f'Latent index: {args.latent_index}',
        f'HDF5        : {args.data_path}',
        '═' * 62,
        '',
    ]

    plots = []

    # T4
    if not args.skip_t4:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'T4: loading model on {device} ...')
        model = load_model(args.ckpt, device)

        t4_text, t4_data = test_rollout_error(
            model, all_z, ep_idx, step_idx, action_lookup,
            max_horizon=args.max_horizon, n_seqs=args.n_seqs, rng=rng)
        print(t4_text)
        report_lines += [t4_text, '']
        if t4_data:
            plots.append(plot_rollout_error(t4_data, args.output_dir))
    else:
        report_lines += ['T4 skipped (--skip-t4)', '']

    # T5
    t5_text, t5_data = test_linear_probe(
        all_z, ep_idx, step_idx, action_lookup,
        epochs=args.probe_epochs, rng=rng)
    print(t5_text)
    report_lines += [t5_text, '']
    if t5_data:
        plots.append(plot_probe(t5_data, args.output_dir))

    # Save report
    report_text = '\n'.join(report_lines)
    report_path = os.path.join(args.output_dir, 'model_diagnostic_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f'\nReport saved → {report_path}')

    # Upload
    if args.s3_output:
        prefix = args.s3_output.rstrip('/') + '/'
        for fpath in [report_path] + plots:
            if fpath and os.path.exists(fpath):
                _s3_upload(fpath, prefix + os.path.basename(fpath))

    print('\nDone.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
T6 — Steering sensitivity diagnostic for the LeWM predictor.

Measures whether the predictor distinguishes left-vs-right steering at H=3.

For 100 context windows (z_{t-2fs}, z_{t-fs}, z_t) where |steering| < 0.2:
  Roll out k=3 steps with a_right = [0.4, +0.5] x 3  and  a_left = [0.4, -0.5] x 3
  Report Mean L2(z_right_3, z_left_3)

Interpretation vs T4 (k=3 rollout error ≈ 1.928 for fs=1 checkpoint):
  < 1.9  → predictor does NOT distinguish steering; more CEM won't help
  1.9–5  → weak signal; marginal benefit from more CEM
  > 5    → strong signal; predictor reliably separates; try N=1000

Modes:
  Default: uses pre-computed latent_index.npz (fast; only valid for fs=1 checkpoint)
  --encode-from-hdf5: encodes context frames from HDF5 pixels (correct for any checkpoint)
"""

import argparse, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LEWM_DIR   = Path('/home/ubuntu/le-wm') if Path('/home/ubuntu/le-wm').exists() \
             else Path('/tmp/le-wm')
IMG_SIZE   = 224
EMBED_DIM  = 192
ACTION_DIM = 2
HISTORY    = 3
LAG_FRAMES = 4


def _ensure_lewm():
    if not LEWM_DIR.exists():
        import subprocess
        r = subprocess.run(['git', 'clone', '--depth', '1',
                            'https://github.com/lucas-maes/le-wm.git', str(LEWM_DIR)])
        if r.returncode != 0:
            raise RuntimeError('git clone le-wm failed')
    if str(LEWM_DIR) not in sys.path:
        sys.path.insert(0, str(LEWM_DIR))


def _build_model(device):
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
    return JEPA(encoder, predictor, action_encoder, projector, pred_proj).to(device)


def load_model(ckpt_path, device):
    if ckpt_path.startswith('s3://'):
        import boto3
        from urllib.parse import urlparse
        u = urlparse(ckpt_path)
        local = Path('/tmp') / Path(u.path).name
        print(f'Downloading {ckpt_path} ...')
        boto3.client('s3', region_name='us-east-1').download_file(
            u.netloc, u.path.lstrip('/'), str(local))
        ckpt_path = str(local)
    model = _build_model(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model', ckpt), strict=True)
    model.eval()
    print(f'Loaded checkpoint ({sum(p.numel() for p in model.parameters()):,} params)')
    return model


def encode_frames(model, pixels_batch, device):
    """
    pixels_batch: (B, 3, H, W) float [0,1]
    Returns (B, D) embeddings.
    """
    px = (pixels_batch * 2.0 - 1.0).to(device)
    with torch.no_grad():
        return model.encode({'pixels': px.unsqueeze(1)})['emb'][:, 0]  # (B, D)


@torch.no_grad()
def rollout_k(model, ctx_embs_batch, ctx_act_past_batch, actions, device, k=3):
    """
    ctx_embs_batch:    (B, HISTORY, D)   — pre-encoded frame embeddings
    ctx_act_past_batch:(B, HISTORY-1, D) — pre-encoded past action embeddings
    actions:           (k, 2) numpy      — action sequence applied to all B windows
    Returns (B, D) final embeddings after k prediction steps.
    """
    B = ctx_embs_batch.shape[0]
    emb_win = ctx_embs_batch.clone()
    act_win = ctx_act_past_batch.clone()

    act_t    = torch.from_numpy(actions.astype(np.float32)).to(device)       # (k, 2)
    act_embs = model.action_encoder(act_t.unsqueeze(0).expand(B, -1, -1))    # (B, k, D)

    for step in range(k):
        full_act = torch.cat([act_win, act_embs[:, step:step + 1]], dim=1)   # (B, H, D)
        next_emb = model.predict(emb_win, full_act)[:, -1]                   # (B, D)
        emb_win  = torch.cat([emb_win[:, 1:], next_emb.unsqueeze(1)], dim=1)
        act_win  = full_act[:, 1:]

    return next_emb  # (B, D)


def _s3_download(s3_path, local_path):
    import boto3
    from urllib.parse import urlparse
    u = urlparse(s3_path)
    boto3.client('s3', region_name='us-east-1').download_file(
        u.netloc, u.path.lstrip('/'), local_path)
    print(f'Downloaded {s3_path} → {local_path}')


def _load_hdf5(data_path):
    """Download HDF5 from S3 if needed, return local path."""
    if data_path.startswith('s3://'):
        local = Path('/tmp') / Path(data_path.split('/')[-1])
        if not local.exists():
            _s3_download(data_path, str(local))
        return str(local)
    return data_path


def build_windows_from_latent_index(args):
    """Use pre-computed latent_index.npz (only valid for the checkpoint it was built with)."""
    li_path = args.latent_index
    if li_path.startswith('s3://'):
        li_path = '/tmp/latent_index.npz'
        _s3_download(args.latent_index, li_path)
    d = np.load(li_path)
    all_z    = d['all_z']    # (N, D)
    ep_idx   = d['ep_idx']
    step_idx = d['step_idx']
    print(f'Latent index: {len(all_z):,} frames')

    # (ep, step) → global index in latent index
    li_ep_step = {}
    for gi, (ep, step) in enumerate(zip(ep_idx, step_idx)):
        li_ep_step.setdefault(int(ep), {})[int(step)] = gi

    # Actions from HDF5
    hdf5_path = _load_hdf5(args.data_path)
    import h5py
    print(f'Loading HDF5 metadata from {hdf5_path} ...')
    with h5py.File(hdf5_path, 'r') as f:
        actions_all   = f['action'][:]
        ep_all_hdf5   = f['episode_idx'][:]
        step_all_hdf5 = f['step_idx'][:]
    hdf5_ep_step = {}
    for gi, (ep, step) in enumerate(zip(ep_all_hdf5, step_all_hdf5)):
        hdf5_ep_step.setdefault(int(ep), {})[int(step)] = gi

    fs = args.frameskip
    rng = np.random.default_rng(args.seed)
    valid_gis = []

    for gi in range(len(all_z)):
        ep   = int(ep_idx[gi])
        step = int(step_idx[gi])
        ep_map = li_ep_step.get(ep, {})

        # Need HISTORY consecutive frameskip-spaced frames in latent index
        if not all((step - h * fs) in ep_map for h in range(HISTORY)):
            continue

        hdf5_gi = hdf5_ep_step.get(ep, {}).get(step)
        if hdf5_gi is None:
            continue
        if abs(float(actions_all[hdf5_gi, 1])) >= 0.2:
            continue
        valid_gis.append(gi)

    print(f'Valid context windows (|steering|<0.2): {len(valid_gis):,}')
    if not valid_gis:
        raise RuntimeError('No valid windows — check latent index / HDF5')

    N = min(args.n, len(valid_gis))
    selected = rng.choice(valid_gis, size=N, replace=False)

    ctx_embs_list, ctx_act_list = [], []
    for gi in selected:
        ep   = int(ep_idx[gi])
        step = int(step_idx[gi])
        ep_map = li_ep_step[ep]

        hist_gis = [ep_map[step - h * fs] for h in reversed(range(HISTORY))]
        ctx_embs_list.append(all_z[hist_gis])

        past_acts = []
        for h in reversed(range(1, HISTORY)):
            hgi = hdf5_ep_step.get(ep, {}).get(step - h * fs)
            past_acts.append(actions_all[hgi] if hgi is not None
                             else np.array([0.35, 0.0], dtype=np.float32))
        ctx_act_list.append(np.array(past_acts, dtype=np.float32))

    ctx_embs = np.stack(ctx_embs_list).astype(np.float32)
    ctx_acts = np.stack(ctx_act_list)
    print(f'Selected {N} windows (latent-index mode)')
    return ctx_embs, ctx_acts, N


def build_windows_from_hdf5(model, args, device):
    """Encode context frames directly from HDF5 pixels — works for any checkpoint."""
    hdf5_path = _load_hdf5(args.data_path)
    import h5py
    print(f'Loading HDF5 from {hdf5_path} ...')
    with h5py.File(hdf5_path, 'r') as f:
        ep_all    = f['episode_idx'][:]
        step_all  = f['step_idx'][:]
        acts_all  = f['action'][:]
        n_total   = len(ep_all)
        # Pixel shape for slicing later
        px_shape = f['pixels'].shape  # e.g. (N, H, W, 3)

    print(f'HDF5: {n_total:,} frames, pixel shape {px_shape}')

    hdf5_ep_step = {}
    for gi, (ep, step) in enumerate(zip(ep_all, step_all)):
        hdf5_ep_step.setdefault(int(ep), {})[int(step)] = gi

    fs = args.frameskip
    rng = np.random.default_rng(args.seed)
    valid_items = []  # list of (ep, step) anchor

    for gi in range(n_total):
        ep   = int(ep_all[gi])
        step = int(step_all[gi])

        # Must be frameskip-aligned
        if step < LAG_FRAMES:
            continue
        if (step - LAG_FRAMES) % fs != 0:
            continue

        # Need HISTORY-1 prior frameskip-spaced frames
        ep_map = hdf5_ep_step.get(ep, {})
        if not all((step - h * fs) in ep_map for h in range(1, HISTORY)):
            continue

        if abs(float(acts_all[gi, 1])) >= 0.2:
            continue

        valid_items.append((ep, step))

    print(f'Valid context windows (|steering|<0.2, fs={fs}): {len(valid_items):,}')
    if not valid_items:
        raise RuntimeError('No valid context windows — check frameskip / HDF5')

    N = min(args.n, len(valid_items))
    idxs = rng.choice(len(valid_items), size=N, replace=False)
    selected = [valid_items[i] for i in idxs]

    # Encode context frames in batches
    print(f'Encoding {N * HISTORY} context frames ...')
    BATCH = 32

    all_anchors = []
    all_ctx_embs = []
    all_ctx_acts = []

    with h5py.File(hdf5_path, 'r') as f:
        pixels_ds = f['pixels']

        # Gather all global indices needed
        for ep, step in selected:
            ep_map = hdf5_ep_step[ep]
            hist_gis = [ep_map[step - h * fs] for h in reversed(range(HISTORY))]
            past_act_gis = [ep_map.get(step - h * fs) for h in reversed(range(1, HISTORY))]
            all_anchors.append((hist_gis, past_act_gis))

        # Batch-encode all context triplets
        for i in range(0, N, BATCH):
            batch_anchors = all_anchors[i:i + BATCH]
            B = len(batch_anchors)

            # Flatten all needed pixel indices
            flat_gis = [gi for (hgis, _) in batch_anchors for gi in hgis]
            # HDF5 fancy indexing requires sorted unique; load + reindex
            unique_gis = sorted(set(flat_gis))
            gi_to_local = {gi: j for j, gi in enumerate(unique_gis)}
            px_raw = pixels_ds[unique_gis]  # (U, H, W, 3) uint8

            ctx_embs_batch = []
            ctx_acts_batch = []

            for hist_gis, past_act_gis in batch_anchors:
                local_idxs = [gi_to_local[gi] for gi in hist_gis]
                px = torch.from_numpy(
                    px_raw[local_idxs].astype(np.float32) / 255.0
                ).permute(0, 3, 1, 2)  # (HISTORY, 3, H, W)
                px = F.interpolate(px, size=(IMG_SIZE, IMG_SIZE),
                                   mode='bilinear', align_corners=False)
                ctx_embs_batch.append(px)

                past_acts = []
                for pgi in past_act_gis:
                    past_acts.append(acts_all[pgi] if pgi is not None
                                     else np.array([0.35, 0.0], dtype=np.float32))
                ctx_acts_batch.append(np.array(past_acts, dtype=np.float32))

            ctx_px = torch.stack(ctx_embs_batch)  # (B, HISTORY, 3, IMG, IMG)
            # Encode: reshape to (B*HISTORY, 3, IMG, IMG), then reshape back
            px_flat = ctx_px.view(B * HISTORY, 3, IMG_SIZE, IMG_SIZE)
            z_flat = encode_frames(model, px_flat, device)           # (B*H, D)
            z = z_flat.view(B, HISTORY, -1).cpu().numpy()            # (B, H, D)

            all_ctx_embs.append(z)
            all_ctx_acts.append(np.stack(ctx_acts_batch))

    ctx_embs = np.concatenate(all_ctx_embs).astype(np.float32)  # (N, HISTORY, D)
    ctx_acts = np.concatenate(all_ctx_acts)                      # (N, HISTORY-1, 2)
    print(f'Encoded {N} context windows (hdf5 mode)')
    return ctx_embs, ctx_acts, N


def main():
    ap = argparse.ArgumentParser(description='T6 steering sensitivity diagnostic')
    ap.add_argument('--ckpt',         default='s3://leworldduckie/training/runs/notebook/checkpoint_best.pt')
    ap.add_argument('--data-path',    default='s3://leworldduckie/data/duckietown_100k.h5')
    ap.add_argument('--latent-index', default='s3://leworldduckie/evals/latent_index.npz')
    ap.add_argument('--encode-from-hdf5', action='store_true',
                    help='Encode context frames from HDF5 pixels (required when latent index '
                         'was built with a different checkpoint)')
    ap.add_argument('--frameskip', type=int, default=1,
                    help='Training frameskip — determines context window step spacing (default 1)')
    ap.add_argument('--n',   type=int,   default=100,  help='Number of context windows')
    ap.add_argument('--k',   type=int,   default=3,    help='Rollout horizon')
    ap.add_argument('--steer-mag', type=float, default=0.5, help='Steering magnitude ±')
    ap.add_argument('--seed', type=int,  default=42)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  frameskip={args.frameskip}  '
          f'mode={"hdf5-encode" if args.encode_from_hdf5 else "latent-index"}')

    model = load_model(args.ckpt, device)

    if args.encode_from_hdf5:
        ctx_embs_np, ctx_acts_np, N = build_windows_from_hdf5(model, args, device)
    else:
        ctx_embs_np, ctx_acts_np, N = build_windows_from_latent_index(args)

    ctx_embs_t  = torch.from_numpy(ctx_embs_np).to(device)
    ctx_acts_raw = torch.from_numpy(ctx_acts_np).to(device)
    with torch.no_grad():
        ctx_act_past = model.action_encoder(ctx_acts_raw)  # (N, HISTORY-1, D)

    vel_base = 0.4
    a_right = np.array([[vel_base,  args.steer_mag]] * args.k, dtype=np.float32)
    a_left  = np.array([[vel_base, -args.steer_mag]] * args.k, dtype=np.float32)

    BATCH = 50
    z_right_list, z_left_list = [], []
    with torch.no_grad():
        for i in range(0, N, BATCH):
            embs_b = ctx_embs_t[i:i + BATCH]
            acts_b = ctx_act_past[i:i + BATCH]
            z_right_list.append(rollout_k(model, embs_b, acts_b, a_right, device, args.k))
            z_left_list.append( rollout_k(model, embs_b, acts_b, a_left,  device, args.k))

    z_right = torch.cat(z_right_list, dim=0)
    z_left  = torch.cat(z_left_list,  dim=0)
    diffs   = (z_right - z_left).norm(dim=-1).cpu().numpy()

    T4_NOISE = 1.928  # k=3 rollout error for fs=1 checkpoint; re-run T4 for exact ratio

    print()
    print('─── T6 Steering Sensitivity ────────────────────────────────────')
    print(f'  k={args.k}  N={N}  steer=±{args.steer_mag}  vel={vel_base}  frameskip={args.frameskip}')
    print()
    print(f'  Mean L2(z_right_{args.k}, z_left_{args.k}) = {diffs.mean():.3f} ± {diffs.std():.3f}')
    print(f'  Median: {np.median(diffs):.3f}')
    print(f'  Min / Max: {diffs.min():.3f} / {diffs.max():.3f}')
    print()
    print(f'  Reference T4 noise floor at k=3 (fs=1 checkpoint): {T4_NOISE}')
    ratio = diffs.mean() / T4_NOISE
    print(f'  Ratio (signal / T4 noise): {ratio:.2f}x')
    print()
    if diffs.mean() < 1.9:
        verdict = ('FAIL — steering effect < T4 noise floor.\n'
                   '  Predictor does NOT distinguish steering.\n'
                   '  Wiring audit required before any further MPC work.')
    elif diffs.mean() > 5.0:
        verdict = (f'STRONG — steering effect >> noise floor ({ratio:.1f}x).\n'
                   '  Predictor reliably separates left/right trajectories.\n'
                   '  Proceed with MPC eval using this checkpoint.')
    else:
        verdict = (f'MARGINAL — steering effect in [1.9, 5.0] ({ratio:.1f}x).\n'
                   '  Predictor weakly distinguishes steering.\n'
                   '  MPC may work partially; Bezier goals still worthwhile.')
    print(f'  Verdict: {verdict}')
    print('────────────────────────────────────────────────────────────────')


if __name__ == '__main__':
    main()

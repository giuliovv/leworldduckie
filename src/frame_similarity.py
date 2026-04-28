#!/usr/bin/env python3
"""
Step 1 — Identity-shortcut test.

Measures ||z_{t+fs} - z_t||^2 in latent space at multiple frameskips.
If this equals the training best_val, the predictor found the identity shortcut.

Also measures relative frame change (||delta_z|| / ||z||) and compares
Duckietown vs Push-T (optional, requires --pusht-data).

Usage (Duckietown only):
  python3 frame_similarity.py \
      --ckpt s3://leworldduckie/training/runs/notebook/checkpoint_best.pt \
      --data-path s3://leworldduckie/data/duckietown_100k.h5 \
      --best-val 0.168

Usage (with Push-T comparison):
  python3 frame_similarity.py ... --pusht-data /path/to/pusht_expert_train.h5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

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
        if not local.exists():
            print(f'Downloading {ckpt_path} ...')
            boto3.client('s3', region_name='us-east-1').download_file(
                u.netloc, u.path.lstrip('/'), str(local))
        ckpt_path = str(local)
    model = _build_model(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    best_val = ckpt.get('best_val', None)
    epoch    = ckpt.get('epoch', None)
    print(f'Loaded checkpoint (epoch={epoch}, best_val={best_val}, '
          f'{sum(p.numel() for p in model.parameters()):,} params)')
    return model, best_val


def _s3_download(s3_path, local_path):
    import boto3
    from urllib.parse import urlparse
    u = urlparse(s3_path)
    boto3.client('s3', region_name='us-east-1').download_file(
        u.netloc, u.path.lstrip('/'), local_path)
    print(f'Downloaded {s3_path} → {local_path}')


def _get_local_hdf5(data_path):
    if data_path.startswith('s3://'):
        local = Path('/tmp') / Path(data_path.split('/')[-1])
        if not local.exists():
            _s3_download(data_path, str(local))
        return str(local)
    return data_path


@torch.no_grad()
def encode_pixel_batch(model, px_uint8_batch, device):
    """
    px_uint8_batch: (B, H, W, 3) uint8
    Returns (B, D) embeddings.
    """
    px = torch.from_numpy(px_uint8_batch.astype(np.float32) / 255.0)
    px = px.permute(0, 3, 1, 2)  # (B, 3, H, W)
    px = F.interpolate(px, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
    px = (px * 2.0 - 1.0).to(device)
    return model.encode({'pixels': px.unsqueeze(1)})['emb'][:, 0]  # (B, D)


def measure_frame_similarity(model, hdf5_path, device, frameskip, n_pairs=1000,
                              seed=42, batch_size=64, tag=""):
    """
    Measures ||z_{t+fs} - z_t||^2 for n_pairs random consecutive pairs
    at the given frameskip.
    Returns (delta_sq_mean, delta_sq_std, z_norm_sq_mean, n_actual)
    """
    rng = np.random.default_rng(seed)

    print(f'Loading HDF5 metadata from {hdf5_path} ...')
    with h5py.File(hdf5_path, 'r') as f:
        ep_all   = f['episode_idx'][:]
        step_all = f['step_idx'][:]
        n_total  = len(ep_all)

    # Build (ep, step) → global index
    ep_step_to_gi = {}
    for gi, (ep, step) in enumerate(zip(ep_all, step_all)):
        ep_step_to_gi[(int(ep), int(step))] = gi

    # Valid pairs: (gi_t, gi_t+fs) in same episode, step_t >= LAG_FRAMES
    valid_pairs = []
    for gi in range(n_total):
        ep   = int(ep_all[gi])
        step = int(step_all[gi])
        if step < LAG_FRAMES:
            continue
        gi_next = ep_step_to_gi.get((ep, step + frameskip))
        if gi_next is None:
            continue
        valid_pairs.append((gi, gi_next))

    print(f'  Valid pairs (fs={frameskip}): {len(valid_pairs):,}')
    n_actual = min(n_pairs, len(valid_pairs))
    chosen = rng.choice(len(valid_pairs), size=n_actual, replace=False)
    pairs  = [valid_pairs[i] for i in chosen]

    delta_sq_list = []
    z_norm_sq_list = []

    with h5py.File(hdf5_path, 'r') as f:
        pixels_ds = f['pixels']

        for start in range(0, n_actual, batch_size):
            batch_pairs = pairs[start:start + batch_size]
            gis_t      = [p[0] for p in batch_pairs]
            gis_tnext  = [p[1] for p in batch_pairs]

            all_gis = sorted(set(gis_t + gis_tnext))
            gi_to_local = {gi: j for j, gi in enumerate(all_gis)}
            px_all = pixels_ds[all_gis]  # (U, H, W, 3)

            px_t    = px_all[[gi_to_local[gi] for gi in gis_t]]
            px_next = px_all[[gi_to_local[gi] for gi in gis_tnext]]

            z_t    = encode_pixel_batch(model, px_t,    device).cpu()
            z_next = encode_pixel_batch(model, px_next, device).cpu()

            delta_sq = (z_next - z_t).pow(2).sum(dim=-1)   # (B,)
            z_norm_sq = z_t.pow(2).sum(dim=-1)              # (B,)

            delta_sq_list.append(delta_sq.numpy())
            z_norm_sq_list.append(z_norm_sq.numpy())

    delta_sq  = np.concatenate(delta_sq_list)
    z_norm_sq = np.concatenate(z_norm_sq_list)

    label = f" [{tag}]" if tag else ""
    print(f'  fs={frameskip:2d}{label}  '
          f'||delta_z||^2: mean={delta_sq.mean():.4f} ± {delta_sq.std():.4f}  '
          f'||z||^2: mean={z_norm_sq.mean():.2f}  '
          f'relative: {delta_sq.mean()/z_norm_sq.mean()*100:.2f}%')

    return delta_sq.mean(), delta_sq.std(), z_norm_sq.mean(), n_actual


def try_download_pusht(pusht_path):
    """Try to download Push-T HDF5 from HuggingFace. Returns local path or None."""
    if pusht_path and Path(pusht_path).exists():
        return pusht_path

    try:
        from huggingface_hub import hf_hub_download
        import tarfile
        print('Downloading pusht_expert_train from HuggingFace (quentinll/lewm) ...')
        archive = hf_hub_download(
            repo_id='quentinll/lewm',
            filename='pusht_expert_train.h5.tar.zst',
            repo_type='dataset',
            local_dir='/tmp',
        )
        out_path = '/tmp/pusht_expert_train.h5'
        import subprocess
        subprocess.run(['tar', '--zstd', '-xvf', archive, '-C', '/tmp'], check=True)
        if Path(out_path).exists():
            print(f'Push-T data ready: {out_path}')
            return out_path
    except Exception as e:
        print(f'Push-T download failed ({e}) — skipping comparison')

    return None


def main():
    ap = argparse.ArgumentParser(description='Identity shortcut test')
    ap.add_argument('--ckpt',       default='s3://leworldduckie/training/runs/notebook/checkpoint_best.pt')
    ap.add_argument('--data-path',  default='s3://leworldduckie/data/duckietown_100k.h5')
    ap.add_argument('--best-val',   type=float, default=None,
                    help='Override best_val from checkpoint (training convergence loss)')
    ap.add_argument('--frameskips', default='1,3,6',
                    help='Comma-separated frameskips to measure (default: 1,3,6)')
    ap.add_argument('--n-pairs',    type=int, default=1000)
    ap.add_argument('--seed',       type=int, default=42)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--pusht-data', default=None,
                    help='Path or None. If set, also runs Push-T comparison at fs=5.')
    ap.add_argument('--try-pusht-download', action='store_true',
                    help='Try to download Push-T from HuggingFace if --pusht-data not given')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model, ckpt_best_val = load_model(args.ckpt, device)
    best_val = args.best_val if args.best_val is not None else ckpt_best_val

    duckie_path = _get_local_hdf5(args.data_path)
    frameskips  = [int(x) for x in args.frameskips.split(',')]

    print()
    print('─── Frame similarity (Duckietown) ──────────────────────────────')
    results = {}
    for fs in frameskips:
        d_sq_mean, d_sq_std, z_norm_sq_mean, n = measure_frame_similarity(
            model, duckie_path, device,
            frameskip=fs, n_pairs=args.n_pairs, seed=args.seed,
            batch_size=args.batch_size, tag='duckie',
        )
        results[fs] = (d_sq_mean, d_sq_std, z_norm_sq_mean)

    # Push-T comparison
    pusht_path = args.pusht_data
    if pusht_path is None and args.try_pusht_download:
        pusht_path = try_download_pusht(None)

    pusht_result = None
    if pusht_path:
        print()
        print('─── Frame similarity (Push-T, fs=5) ────────────────────────────')
        pusht_d_sq, pusht_d_sq_std, pusht_z_sq, _ = measure_frame_similarity(
            model, pusht_path, device,
            frameskip=5, n_pairs=args.n_pairs, seed=args.seed,
            batch_size=args.batch_size, tag='pusht',
        )
        pusht_result = (pusht_d_sq, pusht_d_sq_std, pusht_z_sq)

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print('═'*68)
    print('  IDENTITY SHORTCUT TEST — SUMMARY')
    print('═'*68)
    if best_val is not None:
        print(f'  Training convergence loss (best_val): {best_val:.4f}')
        print(f'  (Identity shortcut = predictor copies previous frame.')
        print(f'   Identity loss ≈ ||z_{{t+fs}} - z_t||^2 at training frameskip.)')
    print()

    print(f'  {"Frameskip":>10}  {"||delta_z||^2":>14}  {"||z||^2":>10}  {"relative %":>10}  Diagnosis')
    print(f'  {"-"*10}  {"-"*14}  {"-"*10}  {"-"*10}  ---------')
    for fs, (d_sq, d_sq_std, z_sq) in sorted(results.items()):
        rel_pct = d_sq / z_sq * 100
        if best_val is not None:
            identity_ratio = d_sq / best_val
            ident_str = f'  {identity_ratio:.2f}× best_val'
        else:
            ident_str = ''
        print(f'  {fs:>10}  {d_sq:>12.4f}±{d_sq_std:5.3f}  {z_sq:>10.2f}  {rel_pct:>9.2f}%{ident_str}')

    if pusht_result:
        pd, pds, pz = pusht_result
        print(f'  {"pusht fs=5":>10}  {pd:>12.4f}±{pds:5.3f}  {pz:>10.2f}  {pd/pz*100:>9.2f}%')

    print()
    # Determine the training frameskip by checking which fs matches best_val
    if best_val is not None:
        print('  Key question: is ||delta_z||^2 at training frameskip ≈ best_val?')
        for fs, (d_sq, _, _) in results.items():
            ratio = d_sq / best_val if best_val > 0 else float('inf')
            if 0.5 < ratio < 2.0:
                print(f'  → fs={fs}: ||delta_z||^2 = {d_sq:.4f} ≈ best_val {best_val:.4f} (ratio {ratio:.2f}×)')
                print(f'    CONFIRMED: predictor learned the identity shortcut at frameskip={fs}')
            elif ratio < 0.5:
                print(f'  → fs={fs}: ||delta_z||^2 = {d_sq:.4f} << best_val {best_val:.4f} (ratio {ratio:.2f}×)')
                print(f'    At this fs, identity IS cheap — harder task needed')
            else:
                print(f'  → fs={fs}: ||delta_z||^2 = {d_sq:.4f} > best_val {best_val:.4f} (ratio {ratio:.2f}×)')
                print(f'    Identity would cost MORE than best_val — shortcut may not be the issue at fs={fs}')

    if pusht_result and best_val is not None:
        pd = pusht_result[0]
        rel_duckie = results.get(1, results[min(results)])[0] / results.get(1, results[min(results)])[2]
        rel_pusht  = pd / pusht_result[2]
        print(f'\n  Duckietown relative change (fs=1): {rel_duckie*100:.2f}%')
        print(f'  Push-T    relative change (fs=5): {rel_pusht*100:.2f}%')
        if rel_pusht > 3 * rel_duckie:
            print(f'  → Push-T frames change {rel_pusht/rel_duckie:.1f}× more per step → temporal scale mismatch confirmed')
        else:
            print(f'  → Similar relative change → temporal scale is not the main difference')

    print('═'*68)


if __name__ == '__main__':
    main()

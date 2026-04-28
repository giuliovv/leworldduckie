#!/usr/bin/env python3
"""
T6 / T4 evaluation on real Duckietown frames.

T6  – Action discriminability: L2(z_right_k, z_left_k) vs noise floor
       using real encoder embeddings. Pass: ratio > 2×.

T4  – Rollout prediction error vs horizon k=1..5:
       mean ||pred_z_{t+k} - enc_z_{t+k}|| using real actions.

ID  – Identity-shortcut check: mean ||z_{t+1} - z_t||^2 vs best_val.

Usage:
  python3 t6_eval.py \
      --ckpt s3://leworldduckie/training/runs/fs6_npreds4_ep20/checkpoint_best.pt \
      --data-path s3://leworldduckie/data/duckietown_100k.h5
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
LAG_FRAMES = 4   # same guard as frame_similarity.py


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
    print(f'Loaded checkpoint  epoch={epoch}  best_val={best_val}')
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
    """px_uint8_batch: (B, H, W, 3) uint8 → (B, D)"""
    px = torch.from_numpy(px_uint8_batch.astype(np.float32) / 255.0)
    px = px.permute(0, 3, 1, 2)
    px = F.interpolate(px, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
    px = (px * 2.0 - 1.0).to(device)
    return model.encode({'pixels': px.unsqueeze(1)})['emb'][:, 0]  # (B, D)


def _norm_act_like_training(act_emb, frame_emb):
    """Replicate per-batch action normalization used in step_fn during training."""
    frame_norm = frame_emb.norm(dim=-1).mean()
    act_norm   = act_emb.norm(dim=-1).mean()
    if act_norm > 1e-6:
        act_emb = act_emb * (frame_norm / act_norm)
    return act_emb


@torch.no_grad()
def ar_rollout(model, z_ctx, action_2d_seq, n_steps, device):
    """
    Autoregressive rollout of the predictor.

    z_ctx:          (B, HISTORY, D)  real encoder embeddings
    action_2d_seq:  (B, n_steps, 2)  raw 2D actions for each future step
                    The same action is broadcast to fill each HISTORY window.
    n_steps:        number of steps to roll forward

    Applies the same per-batch action normalization as training step_fn.

    Returns: (B, n_steps, D) predicted embeddings
    """
    from einops import rearrange

    B = z_ctx.size(0)
    emb_buf = z_ctx.clone()   # (B, HISTORY + step, D) grows as we rollout

    preds = []
    for step in range(n_steps):
        ctx_emb = emb_buf[:, -HISTORY:]     # (B, HISTORY, D)

        # Build action window: repeat the step's action HISTORY times
        a_step  = action_2d_seq[:, step:step+1].expand(B, HISTORY, -1)  # (B, H, 2)
        act_emb = model.action_encoder(a_step.to(device))               # (B, H, D)
        act_emb = _norm_act_like_training(act_emb, ctx_emb)

        pred = model.predictor(ctx_emb, act_emb)      # (B, HISTORY, D)
        next_z = model.pred_proj(
            rearrange(pred[:, -1:], 'b t d -> (b t) d'))
        next_z = rearrange(next_z, '(b t) d -> b t d', b=B)  # (B, 1, D)

        emb_buf = torch.cat([emb_buf, next_z], dim=1)
        preds.append(next_z)

    return torch.cat(preds, dim=1)  # (B, n_steps, D)


def build_episode_lookup(hdf5_path):
    with h5py.File(hdf5_path, 'r') as f:
        ep_all   = f['episode_idx'][:]
        step_all = f['step_idx'][:]
    ep_step_to_gi = {(int(ep), int(step)): gi
                     for gi, (ep, step) in enumerate(zip(ep_all, step_all))}
    return ep_all, step_all, ep_step_to_gi


# ─────────────────────────────────────────────────────────────────────────────
# T6: Action discriminability
# ─────────────────────────────────────────────────────────────────────────────

def t6_action_discriminability(model, hdf5_path, device,
                                n_samples=100, n_rollout_steps=3,
                                seed=42, batch_size=32):
    """
    For each sample: encode real z_history, rollout n_rollout_steps with
    right/left/noise action pairs, measure L2 differences.
    """
    rng = np.random.default_rng(seed)
    ep_all, step_all, ep_step_to_gi = build_episode_lookup(hdf5_path)
    n_total = len(ep_all)

    # Valid starting points: need HISTORY consecutive same-episode frames
    valid = []
    for gi in range(n_total):
        ep, step = int(ep_all[gi]), int(step_all[gi])
        if step < LAG_FRAMES + HISTORY - 1:
            continue
        if all(ep_step_to_gi.get((ep, step - h)) is not None for h in range(HISTORY)):
            valid.append(gi)

    chosen = rng.choice(len(valid), size=min(n_samples, len(valid)), replace=False)
    sample_gis = [valid[i] for i in chosen]
    n_actual = len(sample_gis)
    print(f'  T6: {n_actual} samples, {n_rollout_steps} rollout steps')

    # Fixed actions (same for all steps in the rollout window)
    a_right  = torch.tensor([[0.4, +0.5]], dtype=torch.float32).expand(1, n_rollout_steps, -1)
    a_left   = torch.tensor([[0.4, -0.5]], dtype=torch.float32).expand(1, n_rollout_steps, -1)
    a_str    = torch.tensor([[0.4,  0.0]], dtype=torch.float32).expand(1, n_rollout_steps, -1)

    diffs_rl, diffs_rs, diffs_ls, diffs_noise = [], [], [], []

    with h5py.File(hdf5_path, 'r') as f:
        pixels_ds = f['pixels']

        for start in range(0, n_actual, batch_size):
            batch_gis = sample_gis[start:start + batch_size]
            B = len(batch_gis)

            # Encode real z_history for each sample
            ctx_list = []
            for gi in batch_gis:
                ep, step = int(ep_all[gi]), int(step_all[gi])
                gis_hist = [ep_step_to_gi[(ep, step - (HISTORY - 1 - h))]
                            for h in range(HISTORY)]
                px = pixels_ds[gis_hist]                             # (H, h, w, 3)
                z  = encode_pixel_batch(model, px, device).cpu()     # (H, D)
                ctx_list.append(z.unsqueeze(0))

            z_ctx = torch.cat(ctx_list, dim=0).to(device)            # (B, H, D)

            # Broadcast fixed actions to batch
            a_r  = a_right.expand(B, -1, -1).clone().to(device)
            a_l  = a_left.expand(B, -1, -1).clone().to(device)
            a_s  = a_str.expand(B, -1, -1).clone().to(device)
            a_n1 = torch.from_numpy(
                rng.uniform(-1, 1, (B, n_rollout_steps, 2)).astype(np.float32)).to(device)
            a_n2 = torch.from_numpy(
                rng.uniform(-1, 1, (B, n_rollout_steps, 2)).astype(np.float32)).to(device)

            z_right_k = ar_rollout(model, z_ctx, a_r,  n_rollout_steps, device)[:, -1]
            z_left_k  = ar_rollout(model, z_ctx, a_l,  n_rollout_steps, device)[:, -1]
            z_str_k   = ar_rollout(model, z_ctx, a_s,  n_rollout_steps, device)[:, -1]
            z_n1_k    = ar_rollout(model, z_ctx, a_n1, n_rollout_steps, device)[:, -1]
            z_n2_k    = ar_rollout(model, z_ctx, a_n2, n_rollout_steps, device)[:, -1]

            diffs_rl.append((z_right_k - z_left_k).norm(dim=-1).cpu())
            diffs_rs.append((z_right_k - z_str_k).norm(dim=-1).cpu())
            diffs_ls.append((z_left_k  - z_str_k).norm(dim=-1).cpu())
            diffs_noise.append((z_n1_k - z_n2_k).norm(dim=-1).cpu())

    d_rl    = torch.cat(diffs_rl)
    d_rs    = torch.cat(diffs_rs)
    d_ls    = torch.cat(diffs_ls)
    d_noise = torch.cat(diffs_noise)

    ratio = d_rl.mean() / d_noise.mean() if d_noise.mean() > 0 else float('inf')

    print(f'  L2(right, left)     mean={d_rl.mean():.4f} ± {d_rl.std():.4f}')
    print(f'  L2(right, straight) mean={d_rs.mean():.4f} ± {d_rs.std():.4f}')
    print(f'  L2(left,  straight) mean={d_ls.mean():.4f} ± {d_ls.std():.4f}')
    print(f'  L2 noise floor      mean={d_noise.mean():.4f} ± {d_noise.std():.4f}')
    print(f'  Ratio L2(R,L)/noise = {ratio:.2f}×')

    return {'rl': d_rl, 'rs': d_rs, 'ls': d_ls, 'noise': d_noise, 'ratio': float(ratio)}


# ─────────────────────────────────────────────────────────────────────────────
# T6-RH: T6 with randomised history (hypothesis A vs B diagnostic)
# ─────────────────────────────────────────────────────────────────────────────

def t6_random_history(model, hdf5_path, device,
                      n_samples=100, n_rollout_steps=3,
                      seed=42, batch_size=32):
    """
    Same T6 protocol but the two history frames preceding z_now are replaced
    by random, unrelated frames drawn from anywhere in the dataset.
    history = [z_rand1, z_rand2, z_now]

    Interpretation vs real-history T6 ratio (0.86×):
      ratio >> 0.86×  → Hyp B: coherent history was suppressing action effects.
                         Fix: retrain with history=1 or random history masking.
      ratio ≈ 0.86×   → Hyp A: predictor ignores actions regardless of history.
                         Fix: fix action normalization and retrain.
      ratio << 0.86×  → predictor needs consistent temporal context; brittle to OOD.
    """
    rng = np.random.default_rng(seed + 999)
    ep_all, step_all, _ = build_episode_lookup(hdf5_path)
    n_total = len(ep_all)

    # Valid "current" frames: only need the LAG_FRAMES guard
    valid_now = [gi for gi in range(n_total) if int(step_all[gi]) >= LAG_FRAMES]
    chosen = rng.choice(len(valid_now), size=min(n_samples, len(valid_now)), replace=False)
    sample_gis = [valid_now[i] for i in chosen]
    n_actual = len(sample_gis)
    print(f'  T6-randHist: {n_actual} samples, {n_rollout_steps} rollout steps')

    # Pre-sample random history indices for all samples
    rand_hist_gis = [rng.choice(n_total, size=HISTORY - 1, replace=False).tolist()
                     for _ in range(n_actual)]

    # Collect all unique GIs needed, encode in one sorted pass
    all_needed = sorted(set(sample_gis + [gi for rh in rand_hist_gis for gi in rh]))
    gi_to_emb  = {}

    with h5py.File(hdf5_path, 'r') as f:
        pixels_ds = f['pixels']
        for bs in range(0, len(all_needed), batch_size * HISTORY):
            chunk = all_needed[bs:bs + batch_size * HISTORY]
            px    = pixels_ds[chunk]
            z     = encode_pixel_batch(model, px, device).cpu()
            for gi, emb in zip(chunk, z):
                gi_to_emb[gi] = emb

    a_right  = torch.tensor([[0.4, +0.5]], dtype=torch.float32).expand(1, n_rollout_steps, -1)
    a_left   = torch.tensor([[0.4, -0.5]], dtype=torch.float32).expand(1, n_rollout_steps, -1)

    diffs_rl, diffs_noise = [], []

    for start in range(0, n_actual, batch_size):
        batch_idx = list(range(start, min(start + batch_size, n_actual)))
        B = len(batch_idx)

        ctx_list = []
        for idx in batch_idx:
            z_rand = torch.stack([gi_to_emb[gi] for gi in rand_hist_gis[idx]])  # (H-1, D)
            z_now  = gi_to_emb[sample_gis[idx]].unsqueeze(0)                    # (1, D)
            ctx_list.append(torch.cat([z_rand, z_now], dim=0).unsqueeze(0))     # (1, H, D)

        z_ctx = torch.cat(ctx_list, dim=0).to(device)  # (B, H, D)

        a_r  = a_right.expand(B, -1, -1).clone().to(device)
        a_l  = a_left.expand(B, -1, -1).clone().to(device)
        a_n1 = torch.from_numpy(
            rng.uniform(-1, 1, (B, n_rollout_steps, 2)).astype(np.float32)).to(device)
        a_n2 = torch.from_numpy(
            rng.uniform(-1, 1, (B, n_rollout_steps, 2)).astype(np.float32)).to(device)

        z_right_k = ar_rollout(model, z_ctx, a_r,  n_rollout_steps, device)[:, -1]
        z_left_k  = ar_rollout(model, z_ctx, a_l,  n_rollout_steps, device)[:, -1]
        z_n1_k    = ar_rollout(model, z_ctx, a_n1, n_rollout_steps, device)[:, -1]
        z_n2_k    = ar_rollout(model, z_ctx, a_n2, n_rollout_steps, device)[:, -1]

        diffs_rl.append((z_right_k - z_left_k).norm(dim=-1).cpu())
        diffs_noise.append((z_n1_k - z_n2_k).norm(dim=-1).cpu())

    d_rl    = torch.cat(diffs_rl)
    d_noise = torch.cat(diffs_noise)
    ratio   = float(d_rl.mean() / d_noise.mean()) if d_noise.mean() > 0 else float('inf')

    print(f'  L2(right, left)  mean={d_rl.mean():.4f} ± {d_rl.std():.4f}')
    print(f'  L2 noise floor   mean={d_noise.mean():.4f} ± {d_noise.std():.4f}')
    print(f'  Ratio L2(R,L)/noise = {ratio:.2f}×')

    return {'rl': d_rl, 'noise': d_noise, 'ratio': ratio}


# ─────────────────────────────────────────────────────────────────────────────
# Input-vs-Action sensitivity (Hypothesis F diagnostic)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def t6_input_sensitivity(model, hdf5_path, device,
                          n_action_samples=200, n_state_samples=200,
                          seed=42, batch_size=64):
    """
    std_over_actions: fix 1 reference context, vary 200 random actions → per-dim std of 1-step outputs
    std_over_inputs:  fix reference action (straight 0.4), vary 200 real contexts → per-dim std
    sensitivity_ratio = std_over_inputs / std_over_actions

    > 10: input-dominated → Hyp F (encoder redundancy) confirmed
    1-3:  balanced → Hyp F not confirmed
    """
    rng = np.random.default_rng(seed + 777)
    ep_all, step_all, ep_step_to_gi = build_episode_lookup(hdf5_path)
    n_total = len(ep_all)

    valid = []
    for gi in range(n_total):
        ep, step = int(ep_all[gi]), int(step_all[gi])
        if step < LAG_FRAMES + HISTORY - 1:
            continue
        if all(ep_step_to_gi.get((ep, step - h)) is not None for h in range(HISTORY)):
            valid.append(gi)

    n_states = min(n_state_samples, len(valid))
    chosen = rng.choice(len(valid), size=n_states, replace=False)
    state_gis = [valid[i] for i in chosen]
    print(f'  Sensitivity: {n_states} states, {n_action_samples} actions')

    ctx_list = []
    with h5py.File(hdf5_path, 'r') as f:
        pixels_ds = f['pixels']
        for gi in state_gis:
            ep, step = int(ep_all[gi]), int(step_all[gi])
            gis_hist = [ep_step_to_gi[(ep, step - (HISTORY - 1 - h))] for h in range(HISTORY)]
            px = pixels_ds[gis_hist]
            z = encode_pixel_batch(model, px, device).cpu()
            ctx_list.append(z.unsqueeze(0))

    all_ctx = torch.cat(ctx_list)   # (N_state, H, D)
    z_ref   = all_ctx[0:1]          # (1, H, D) — reference for action sweep

    # Vary action, fix state
    a_varied = torch.from_numpy(
        rng.uniform(-1, 1, (n_action_samples, 1, ACTION_DIM)).astype(np.float32))
    out_a = []
    for bs in range(0, n_action_samples, batch_size):
        a_chunk = a_varied[bs:bs + batch_size].to(device)   # (B, 1, 2)
        B = a_chunk.size(0)
        z_ctx_b = z_ref.expand(B, -1, -1).to(device)
        pred = ar_rollout(model, z_ctx_b, a_chunk, n_steps=1, device=device)
        out_a.append(pred[:, 0].cpu())
    out_a = torch.cat(out_a)   # (N_action, D)
    std_action = out_a.std(dim=0).mean().item()

    # Vary state, fix action
    a_ref_t = torch.tensor([[0.4, 0.0]], dtype=torch.float32)  # straight
    out_i = []
    for bs in range(0, n_states, batch_size):
        ctx_chunk = all_ctx[bs:bs + batch_size].to(device)
        B = ctx_chunk.size(0)
        a_b = a_ref_t.expand(B, 1, -1).to(device)
        pred = ar_rollout(model, ctx_chunk, a_b, n_steps=1, device=device)
        out_i.append(pred[:, 0].cpu())
    out_i = torch.cat(out_i)   # (N_state, D)
    std_input = out_i.std(dim=0).mean().item()

    ratio = std_input / std_action if std_action > 1e-9 else float('inf')
    print(f'  std(output | varied actions, fixed state):  {std_action:.5f}')
    print(f'  std(output | varied states,  fixed action): {std_input:.5f}')
    print(f'  Sensitivity ratio (input / action):         {ratio:.1f}×')

    if ratio > 10:
        verdict = f'INPUT-DOMINATED ({ratio:.1f}×) — Hyp F confirmed'
    elif ratio > 3:
        verdict = f'Mildly input-dominated ({ratio:.1f}×) — partial Hyp F'
    else:
        verdict = f'Balanced ({ratio:.1f}×) — Hyp F not confirmed'
    print(f'  → {verdict}')
    return {'std_action': std_action, 'std_input': std_input,
            'sensitivity_ratio': ratio, 'verdict': verdict}


# ─────────────────────────────────────────────────────────────────────────────
# T4: Rollout prediction error vs horizon
# ─────────────────────────────────────────────────────────────────────────────

def t4_rollout_error(model, hdf5_path, device,
                     max_horizon=5, n_samples=100, seed=42, batch_size=32):
    """
    For k=1..max_horizon: predict z_{t+k} using real actions, compare to
    the actual encoder output enc(obs_{t+k}).
    """
    rng = np.random.default_rng(seed)
    ep_all, step_all, ep_step_to_gi = build_episode_lookup(hdf5_path)
    n_total = len(ep_all)

    # Need HISTORY frames as context + max_horizon future frames in same episode
    valid = []
    for gi in range(n_total):
        ep, step = int(ep_all[gi]), int(step_all[gi])
        if step < LAG_FRAMES + HISTORY - 1:
            continue
        ok_past   = all(ep_step_to_gi.get((ep, step - h)) is not None
                        for h in range(HISTORY))
        ok_future = all(ep_step_to_gi.get((ep, step + k)) is not None
                        for k in range(1, max_horizon + 1))
        if ok_past and ok_future:
            valid.append(gi)

    chosen = rng.choice(len(valid), size=min(n_samples, len(valid)), replace=False)
    sample_gis = [valid[i] for i in chosen]
    n_actual = len(sample_gis)
    print(f'  T4: {n_actual} samples, horizon 1..{max_horizon}')

    errors_by_k = {k: [] for k in range(1, max_horizon + 1)}

    with h5py.File(hdf5_path, 'r') as f:
        pixels_ds = f['pixels']
        actions_ds = f['action']

        for start in range(0, n_actual, batch_size):
            batch_gis = sample_gis[start:start + batch_size]
            B = len(batch_gis)

            ctx_list  = []
            act_list  = []   # real actions at t, t+1, ..., t+max_horizon-1
            fut_list  = []   # future frames t+1 .. t+max_horizon

            for gi in batch_gis:
                ep, step = int(ep_all[gi]), int(step_all[gi])
                gis_hist = [ep_step_to_gi[(ep, step - (HISTORY - 1 - h))]
                            for h in range(HISTORY)]
                gis_fut  = [ep_step_to_gi[(ep, step + k)]
                            for k in range(1, max_horizon + 1)]

                px_hist = pixels_ds[gis_hist]               # (H, h, w, 3)
                px_fut  = pixels_ds[gis_fut]                # (K, h, w, 3)
                acts    = actions_ds[gis_hist]              # (H, 2) actions at context frames

                z_hist = encode_pixel_batch(model, px_hist, device).cpu()
                z_fut  = encode_pixel_batch(model, px_fut,  device).cpu()

                ctx_list.append(z_hist.unsqueeze(0))
                act_list.append(torch.from_numpy(acts.astype(np.float32)).unsqueeze(0))
                fut_list.append(z_fut.unsqueeze(0))

            z_ctx   = torch.cat(ctx_list, dim=0).to(device)   # (B, H, D)
            acts    = torch.cat(act_list,  dim=0).to(device)   # (B, H, 2) — real context acts
            z_gt    = torch.cat(fut_list,  dim=0).to(device)   # (B, K, D)

            # Rollout with real actions (repeat context actions as best proxy for future)
            # For step k, action window = last HISTORY actions; we use the last context act repeated
            a_future = acts[:, -1:].expand(B, max_horizon, -1)   # (B, K, 2)
            z_pred   = ar_rollout(model, z_ctx, a_future, max_horizon, device)  # (B, K, D)

            for k in range(1, max_horizon + 1):
                err = (z_pred[:, k-1] - z_gt[:, k-1]).norm(dim=-1).cpu()
                errors_by_k[k].append(err)

    result = {}
    for k in range(1, max_horizon + 1):
        e = torch.cat(errors_by_k[k])
        result[k] = (float(e.mean()), float(e.std()))
        print(f'  k={k}  L2(pred, enc) = {e.mean():.4f} ± {e.std():.4f}')

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Identity shortcut check
# ─────────────────────────────────────────────────────────────────────────────

def identity_shortcut_check(model, hdf5_path, device,
                             frameskip=1, n_pairs=300, seed=42, batch_size=32):
    """Measure mean ||z_{t+fs} - z_t||^2. If ≈ best_val → identity shortcut."""
    rng = np.random.default_rng(seed)
    ep_all, step_all, ep_step_to_gi = build_episode_lookup(hdf5_path)
    n_total = len(ep_all)

    valid_pairs = []
    for gi in range(n_total):
        ep, step = int(ep_all[gi]), int(step_all[gi])
        if step < LAG_FRAMES:
            continue
        gi_next = ep_step_to_gi.get((ep, step + frameskip))
        if gi_next is not None:
            valid_pairs.append((gi, gi_next))

    n_actual = min(n_pairs, len(valid_pairs))
    chosen = rng.choice(len(valid_pairs), size=n_actual, replace=False)
    pairs  = [valid_pairs[i] for i in chosen]

    delta_sq_list = []
    with h5py.File(hdf5_path, 'r') as f:
        pixels_ds = f['pixels']
        for start in range(0, n_actual, batch_size):
            bp = pairs[start:start + batch_size]
            all_gis = sorted(set([p[0] for p in bp] + [p[1] for p in bp]))
            gi_to_local = {gi: j for j, gi in enumerate(all_gis)}
            px_all = pixels_ds[all_gis]
            z_t    = encode_pixel_batch(model, px_all[[gi_to_local[p[0]] for p in bp]], device).cpu()
            z_next = encode_pixel_batch(model, px_all[[gi_to_local[p[1]] for p in bp]], device).cpu()
            delta_sq_list.append((z_next - z_t).pow(2).sum(dim=-1).numpy())

    delta_sq = np.concatenate(delta_sq_list)
    return float(delta_sq.mean()), float(delta_sq.std())


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',
                    default='s3://leworldduckie/training/runs/fs6_npreds4_ep20/checkpoint_best.pt')
    ap.add_argument('--data-path', default='s3://leworldduckie/data/duckietown_100k.h5')
    ap.add_argument('--n-samples',  type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-horizon', type=int, default=5)
    ap.add_argument('--n-rollout-steps', type=int, default=3)
    ap.add_argument('--out', default='/tmp/t6_results.txt')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model, best_val = load_model(args.ckpt, device)
    hdf5_path = _get_local_hdf5(args.data_path)

    import io, contextlib

    lines = []
    def tee(*a, **kw):
        print(*a, **kw)
        lines.append(' '.join(str(x) for x in a))

    print()
    print('─── Identity shortcut check (fs=1, 6) ─────────────────────────────')
    for fs in [1, 6]:
        dsq_mean, dsq_std = identity_shortcut_check(
            model, hdf5_path, device, frameskip=fs,
            n_pairs=args.n_samples * 3, batch_size=args.batch_size)
        ratio = dsq_mean / best_val if best_val else None
        tag   = f'  fs={fs}  ||delta_z||^2 = {dsq_mean:.4f} ± {dsq_std:.4f}'
        if ratio is not None:
            tag += f'  ({ratio:.2f}× best_val={best_val:.4f})'
            if 0.5 < ratio < 2.0:
                tag += '  ← identity shortcut DETECTED'
            elif ratio < 0.5:
                tag += '  ← below identity cost (ok)'
            else:
                tag += '  ← well above identity cost (ok)'
        print(tag); lines.append(tag)

    print()
    print('─── T4: Rollout prediction error vs horizon ────────────────────────')
    lines.append('\nT4: Rollout prediction error vs horizon')
    t4 = t4_rollout_error(model, hdf5_path, device,
                           max_horizon=args.max_horizon,
                           n_samples=args.n_samples,
                           batch_size=args.batch_size)
    for k, (m, s) in t4.items():
        lines.append(f'  k={k}  {m:.4f} ± {s:.4f}')

    print()
    print('─── Input-vs-Action Sensitivity (Hyp F diagnostic) ─────────────────')
    lines.append('\nInput-vs-Action Sensitivity (Hyp F diagnostic)')
    sens = t6_input_sensitivity(model, hdf5_path, device,
                                 batch_size=args.batch_size)
    for k in ('std_action', 'std_input', 'sensitivity_ratio'):
        lines.append(f'  {k}: {sens[k]:.4f}')
    lines.append(f'  verdict: {sens["verdict"]}')

    print()
    print('─── T6: Action discriminability (real history) ─────────────────────')
    lines.append('\nT6: Action discriminability (real history)')
    t6 = t6_action_discriminability(model, hdf5_path, device,
                                     n_samples=args.n_samples,
                                     n_rollout_steps=args.n_rollout_steps,
                                     batch_size=args.batch_size)

    print()
    print('─── T6-RH: Action discriminability (random history) ────────────────')
    lines.append('\nT6-RH: Action discriminability (random history)')
    t6rh = t6_random_history(model, hdf5_path, device,
                              n_samples=args.n_samples,
                              n_rollout_steps=args.n_rollout_steps,
                              batch_size=args.batch_size)

    # Hypothesis interpretation
    rh_ratio     = t6rh['ratio']
    real_ratio   = t6['ratio']
    boost_factor = rh_ratio / real_ratio if real_ratio > 0 else float('inf')

    print()
    print('═' * 68)
    print('  T6 / T4 SUMMARY')
    print('═' * 68)
    print(f'  Checkpoint best_val      : {best_val:.4f}')
    print(f'  T6 ratio real-hist       : {real_ratio:.2f}×')
    print(f'  T6 ratio random-hist     : {rh_ratio:.2f}×')
    print(f'  Random-hist boost factor : {boost_factor:.2f}×')
    verdict = ('PASS' if real_ratio > 2.0
               else 'MARGINAL (consider more epochs)' if real_ratio > 1.0
               else 'FAIL')
    print(f'  T6 verdict               : {verdict}')
    print()
    if boost_factor > 3.0:
        hyp = 'B confirmed: coherent history suppresses action effects'
        fix = 'Retrain with history=1 OR random history masking p=0.5'
    elif boost_factor < 1.5:
        hyp = 'A confirmed: predictor ignores actions regardless of history'
        fix = 'Fix action normalization (constant ×5.3 scaling) and retrain'
    else:
        hyp = 'Ambiguous: partial history effect'
        fix = 'Consider both fixes: normalization + shorter history'
    print(f'  Hypothesis: {hyp}')
    print(f'  Suggested fix: {fix}')
    if real_ratio > 2.0:
        print('  → CLEARED for MPC eval')
    elif real_ratio > 1.0:
        print('  → Consider 30 more training epochs before MPC')
    else:
        print('  → DO NOT proceed to MPC — action signal too weak')
    print('═' * 68)

    lines += [
        '',
        '═' * 68,
        'T6 / T4 SUMMARY',
        f'  best_val = {best_val:.4f}',
        f'  T6 ratio (real-hist)   = {real_ratio:.2f}×  [{verdict}]',
        f'  T6 ratio (random-hist) = {rh_ratio:.2f}×',
        f'  Boost factor           = {boost_factor:.2f}×',
        f'  Hypothesis: {hyp}',
        f'  Suggested fix: {fix}',
        f'  Sensitivity ratio (input/action) = {sens["sensitivity_ratio"]:.1f}×  [{sens["verdict"]}]',
    ]

    with open(args.out, 'w') as fh:
        fh.write('\n'.join(lines))
    print(f'\nResults written to {args.out}')


if __name__ == '__main__':
    main()

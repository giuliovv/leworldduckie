#!/usr/bin/env python3
"""
Push-T diagnostics: T5 (state probe), T6 (action discriminability), sensitivity ratio.

Works on the official le-wm checkpoint format (torch.save of spt.Module or JEPA object).
Requires: stable-worldmodel[train], le-wm (auto-cloned), h5py, scikit-learn, boto3.

Usage:
  python3 pusht_diagnostics.py \
      --ckpt s3://leworldduckie/training/pusht/pusht/lewm_object.ckpt \
      --data s3://leworldduckie/data/pusht_expert_train.h5
  # or local paths
  python3 pusht_diagnostics.py \
      --ckpt ~/.stable-wm/pusht/lewm_object.ckpt \
      --data ~/.stable-wm/pusht_expert_train.h5

Outputs:
  T5: R² for linear probe of z → agent_x/y, block_x/y, block_theta
  T6: L2(right, left) / noise floor  (pass: > 2×)
  Sensitivity ratio: std_input / std_action  (>10× = encoder-dominated)
"""
import argparse
import sys
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import h5py

LEWM_DIR   = Path('/home/ubuntu/le-wm') if Path('/home/ubuntu/le-wm').exists() \
             else Path('/tmp/le-wm')

IMG_SIZE   = 224
EMBED_DIM  = 192
HISTORY    = 3      # history_size in training config
FRAMESKIP  = 5      # frameskip in push-T data config
ACTION_DIM = 2      # raw push-T action dim
EFF_A_DIM  = FRAMESKIP * ACTION_DIM   # 10: input_dim of action_encoder

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Setup helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_lewm():
    if not LEWM_DIR.exists():
        import subprocess
        r = subprocess.run(['git', 'clone', '--depth', '1',
                            'https://github.com/lucas-maes/le-wm.git', str(LEWM_DIR)])
        if r.returncode != 0:
            raise RuntimeError('git clone le-wm failed')
    if str(LEWM_DIR) not in sys.path:
        sys.path.insert(0, str(LEWM_DIR))


def _s3_download(s3_path, local_path):
    import boto3
    from urllib.parse import urlparse
    u = urlparse(s3_path)
    boto3.client('s3', region_name='us-east-1').download_file(
        u.netloc, u.path.lstrip('/'), local_path)
    print(f'Downloaded {s3_path} → {local_path}')


def _resolve_path(path):
    if path.startswith('s3://'):
        local = Path('/tmp') / Path(path.split('/')[-1])
        if not local.exists():
            _s3_download(path, str(local))
        return str(local)
    return os.path.expanduser(path)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def load_jepa(ckpt_path, device):
    """
    Load the JEPA world model from an official le-wm checkpoint.
    The official training saves the full spt.Module or JEPA object via torch.save.
    """
    _ensure_lewm()
    print(f'Loading checkpoint: {ckpt_path}')
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)

    # spt.Module wraps JEPA as .model; try both
    jepa = getattr(obj, 'model', obj)
    jepa = jepa.to(device)
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)
    print(f'  JEPA loaded — encoder: {type(jepa.encoder).__name__}, '
          f'predictor: {type(jepa.predictor).__name__}')
    return jepa


# ─────────────────────────────────────────────────────────────────────────────
# Image encoding
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_pixel_batch(jepa, px_uint8, device):
    """
    px_uint8: (B, H, W, 3) uint8
    Returns: (B, EMBED_DIM) projector-output embeddings
    """
    px = torch.from_numpy(px_uint8.astype(np.float32) / 255.0)  # (B, 3, H, W) after permute
    px = px.permute(0, 3, 1, 2)

    if px.shape[-2:] != (IMG_SIZE, IMG_SIZE):
        px = F.interpolate(px, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)

    mean = IMAGENET_MEAN.to(device)
    std  = IMAGENET_STD.to(device)
    px   = (px.to(device) - mean) / std

    emb = jepa.encoder(px, interpolate_pos_encoding=True)  # (B, D)
    return jepa.projector(emb)                              # (B, D)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_episode_lookup(h5_path):
    with h5py.File(h5_path, 'r') as f:
        ep_all   = f['episode_idx'][:]
        step_all = f['step_idx'][:]
    ep_step_to_gi = {(int(ep), int(s)): gi
                     for gi, (ep, s) in enumerate(zip(ep_all, step_all))}
    return ep_all, step_all, ep_step_to_gi


def sample_valid_starts(ep_all, step_all, ep_step_to_gi, n_samples, seed, lag=4):
    """Pick frame indices with HISTORY consecutive frames from the same episode."""
    rng = np.random.default_rng(seed)
    valid = [gi for gi in range(len(ep_all))
             if int(step_all[gi]) >= lag + HISTORY - 1
             and all(ep_step_to_gi.get((int(ep_all[gi]),
                                        int(step_all[gi]) - h)) is not None
                     for h in range(HISTORY))]
    chosen = rng.choice(len(valid), size=min(n_samples, len(valid)), replace=False)
    return [valid[i] for i in chosen]


# ─────────────────────────────────────────────────────────────────────────────
# T5: state probe
# ─────────────────────────────────────────────────────────────────────────────

def t5_state_probe(jepa, h5_path, device, n_samples=1000, seed=0, batch_size=64):
    """
    Linear probe: z_t → state features (agent_x/y, block_x/y, block_theta).
    Paper reports R² > 0.99 for block/agent positions on Push-T (Table 1).
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    rng = np.random.default_rng(seed)
    print(f'\n── T5: State probe ({n_samples} samples) ──')

    with h5py.File(h5_path, 'r') as f:
        ep_all   = f['episode_idx'][:]
        step_all = f['step_idx'][:]
        n_total  = len(ep_all)
        has_state = 'state' in f

        if not has_state:
            print('  WARNING: no "state" key in HDF5 — T5 skipped')
            return {}

        state_dim = f['state'].shape[1]
        state_labels = {
            5: ['agent_x', 'agent_y', 'block_x', 'block_y', 'block_theta'],
            4: ['agent_x', 'agent_y', 'block_x', 'block_y'],
            2: ['agent_x', 'agent_y'],
        }.get(state_dim, [f'state_{i}' for i in range(state_dim)])
        print(f'  State dim={state_dim}: {state_labels}')

        chosen = rng.choice(n_total, size=min(n_samples, n_total), replace=False)
        chosen.sort()

        all_z, all_s = [], []
        pixels_ds = f['pixels']
        state_ds  = f['state']

        for start in range(0, len(chosen), batch_size):
            idxs = chosen[start:start + batch_size]
            px   = pixels_ds[idxs.tolist()]           # (B, H, W, 3) or (B, C, H, W)
            if px.ndim == 4 and px.shape[1] in (1, 3):  # (B, C, H, W) → (B, H, W, C)
                px = np.transpose(px, (0, 2, 3, 1))
            s    = state_ds[idxs.tolist()]             # (B, D)
            z    = encode_pixel_batch(jepa, px, device).cpu().numpy()
            all_z.append(z)
            all_s.append(s)

    Z = np.concatenate(all_z, axis=0)   # (N, 192)
    S = np.concatenate(all_s, axis=0)   # (N, D_state)

    split = int(0.8 * len(Z))
    Z_tr, Z_te = Z[:split], Z[split:]
    S_tr, S_te = S[:split], S[split:]

    results = {}
    for i, name in enumerate(state_labels):
        probe = Ridge(alpha=1.0).fit(Z_tr, S_tr[:, i])
        r2 = r2_score(S_te[:, i], probe.predict(Z_te))
        results[name] = r2
        status = 'PASS' if r2 > 0.95 else ('WARN' if r2 > 0.80 else 'FAIL')
        print(f'  {name:<15}: R²={r2:.4f}  {status}')

    return results


# ─────────────────────────────────────────────────────────────────────────────
# T6: action discriminability (synthetic contexts)
# ─────────────────────────────────────────────────────────────────────────────

def t6_action_discriminability_synthetic(jepa, device, n_contexts=500, seed=42):
    """
    Synthetic T6 gate using random context embeddings.
    Uses Push-T action semantics: 10D input (frameskip=5 × action_dim=2).
    Tests if the model can distinguish push-right vs push-left.
    """
    print(f'\n── T6: Action discriminability — synthetic ({n_contexts} contexts) ──')
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1)

    sigma = 9.6 / (EMBED_DIM ** 0.5)  # empirical context embedding norm
    z_ctx = torch.randn(n_contexts, HISTORY, EMBED_DIM) * sigma

    # Actions: 10D (5 sub-steps × 2D); use ±1.5σ in x dimension
    def make_action(x_val, y_val=0.0):
        a = torch.zeros(n_contexts, HISTORY, EFF_A_DIM)
        # fill all 5 sub-step x-components
        for i in range(0, EFF_A_DIM, 2):
            a[:, :, i]   = x_val
            a[:, :, i+1] = y_val
        return a

    a_right = make_action(+1.5)
    a_left  = make_action(-1.5)
    a_up    = make_action(0.0,  +1.5)
    a_down  = make_action(0.0,  -1.5)

    a_noise1 = torch.from_numpy(rng.standard_normal((n_contexts, HISTORY, EFF_A_DIM)).astype(np.float32))
    a_noise2 = torch.from_numpy(rng.standard_normal((n_contexts, HISTORY, EFF_A_DIM)).astype(np.float32))

    @torch.no_grad()
    def predict(z, a):
        act_emb = jepa.action_encoder(a.to(device))
        pred = jepa.predict(z.to(device), act_emb)  # (B, H, D)
        return pred[:, -1]                           # (B, D) last position

    z_right  = predict(z_ctx, a_right)
    z_left   = predict(z_ctx, a_left)
    z_up     = predict(z_ctx, a_up)
    z_down   = predict(z_ctx, a_down)
    z_noise1 = predict(z_ctx, a_noise1)
    z_noise2 = predict(z_ctx, a_noise2)

    l2_rl    = (z_right  - z_left).norm(dim=-1)
    l2_ud    = (z_up     - z_down).norm(dim=-1)
    l2_noise = (z_noise1 - z_noise2).norm(dim=-1)

    ratio_rl = l2_rl.mean() / l2_noise.mean()
    ratio_ud = l2_ud.mean() / l2_noise.mean()

    print(f'  L2(right, left): mean={l2_rl.mean():.4f} ± {l2_rl.std():.4f}')
    print(f'  L2(up,    down): mean={l2_ud.mean():.4f} ± {l2_ud.std():.4f}')
    print(f'  L2 noise floor:  mean={l2_noise.mean():.4f} ± {l2_noise.std():.4f}')
    print(f'  Ratio L2(R,L)/noise = {ratio_rl:.2f}×  '
          f'{"PASS" if ratio_rl > 2.0 else "FAIL"} (threshold: >2×)')
    print(f'  Ratio L2(U,D)/noise = {ratio_ud:.2f}×')

    return {'ratio_rl': float(ratio_rl), 'ratio_ud': float(ratio_ud),
            'l2_rl': float(l2_rl.mean()), 'noise': float(l2_noise.mean())}


# ─────────────────────────────────────────────────────────────────────────────
# T6: action discriminability (real frames from dataset)
# ─────────────────────────────────────────────────────────────────────────────

def t6_action_discriminability_real(jepa, h5_path, device, n_samples=100,
                                    seed=42, batch_size=32):
    """
    T6 on real encoder embeddings (matches the Duckietown protocol exactly).
    Encodes real Push-T frames as context, then rolls forward with right/left actions.
    """
    print(f'\n── T6: Action discriminability — real frames ({n_samples} samples) ──')
    rng = np.random.default_rng(seed)
    ep_all, step_all, ep_step_to_gi = build_episode_lookup(h5_path)
    sample_gis = sample_valid_starts(ep_all, step_all, ep_step_to_gi, n_samples, seed)
    n_actual = len(sample_gis)

    def make_action(x_val, y_val=0.0, B=1):
        a = torch.zeros(B, HISTORY, EFF_A_DIM)
        for i in range(0, EFF_A_DIM, 2):
            a[:, :, i]   = x_val
            a[:, :, i+1] = y_val
        return a

    diffs_rl, diffs_noise = [], []

    with h5py.File(h5_path, 'r') as f:
        pixels_ds = f['pixels']

        for start in range(0, n_actual, batch_size):
            batch_gis = sample_gis[start:start + batch_size]
            B = len(batch_gis)

            ctx_list = []
            for gi in batch_gis:
                ep, step = int(ep_all[gi]), int(step_all[gi])
                gis_hist = [ep_step_to_gi[(ep, step - (HISTORY - 1 - h))]
                            for h in range(HISTORY)]
                px = pixels_ds[gis_hist]           # (H, ...)
                if px.ndim == 4 and px.shape[1] in (1, 3):
                    px = np.transpose(px, (0, 2, 3, 1))
                z = encode_pixel_batch(jepa, px, device).cpu()  # (H, D)
                ctx_list.append(z.unsqueeze(0))

            z_ctx = torch.cat(ctx_list, dim=0).to(device)  # (B, H, D)

            a_r = make_action(+1.5, B=B).to(device)
            a_l = make_action(-1.5, B=B).to(device)
            a_n1 = torch.from_numpy(
                rng.standard_normal((B, HISTORY, EFF_A_DIM)).astype(np.float32)).to(device)
            a_n2 = torch.from_numpy(
                rng.standard_normal((B, HISTORY, EFF_A_DIM)).astype(np.float32)).to(device)

            with torch.no_grad():
                act_r  = jepa.action_encoder(a_r)
                act_l  = jepa.action_encoder(a_l)
                act_n1 = jepa.action_encoder(a_n1)
                act_n2 = jepa.action_encoder(a_n2)

                z_r  = jepa.predict(z_ctx, act_r)[:, -1]
                z_l  = jepa.predict(z_ctx, act_l)[:, -1]
                z_n1 = jepa.predict(z_ctx, act_n1)[:, -1]
                z_n2 = jepa.predict(z_ctx, act_n2)[:, -1]

            diffs_rl.append((z_r - z_l).norm(dim=-1).cpu())
            diffs_noise.append((z_n1 - z_n2).norm(dim=-1).cpu())

    d_rl    = torch.cat(diffs_rl)
    d_noise = torch.cat(diffs_noise)
    ratio   = d_rl.mean() / d_noise.mean()

    print(f'  L2(right, left): mean={d_rl.mean():.4f} ± {d_rl.std():.4f}')
    print(f'  L2 noise floor:  mean={d_noise.mean():.4f} ± {d_noise.std():.4f}')
    print(f'  Ratio L2(R,L)/noise = {float(ratio):.2f}×  '
          f'{"PASS" if ratio > 2.0 else "FAIL"} (threshold: >2×)')
    return {'ratio': float(ratio), 'l2_rl': float(d_rl.mean()), 'noise': float(d_noise.mean())}


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity ratio (Hyp F diagnostic)
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_ratio(jepa, h5_path, device, n_samples=200, seed=99, batch_size=32):
    """
    std(output | varied actions, fixed state) vs std(output | varied states, fixed action).
    Ratio > 10× → INPUT-DOMINATED (Hyp F).
    Compare to Duckietown result: 50.6× (INPUT-DOMINATED).
    Push-T expected: much lower ratio (actions cause visible state changes).
    """
    print(f'\n── Sensitivity ratio (Hyp F) — {n_samples} samples ──')
    rng = np.random.default_rng(seed)
    ep_all, step_all, ep_step_to_gi = build_episode_lookup(h5_path)
    sample_gis = sample_valid_starts(ep_all, step_all, ep_step_to_gi, n_samples, seed)
    n_actual = len(sample_gis)

    all_z_ctx   = []
    all_z_fixed = []

    with h5py.File(h5_path, 'r') as f:
        pixels_ds = f['pixels']
        for gi in sample_gis:
            ep, step = int(ep_all[gi]), int(step_all[gi])
            gis_hist = [ep_step_to_gi[(ep, step - (HISTORY - 1 - h))]
                        for h in range(HISTORY)]
            px = pixels_ds[gis_hist]
            if px.ndim == 4 and px.shape[1] in (1, 3):
                px = np.transpose(px, (0, 2, 3, 1))
            z = encode_pixel_batch(jepa, px, device).cpu()
            all_z_ctx.append(z.unsqueeze(0))

    z_all = torch.cat(all_z_ctx, dim=0)  # (N, H, D)

    # Fixed action, vary states
    a_fixed = torch.zeros(1, HISTORY, EFF_A_DIM)
    a_fixed[:, :, 0] = 0.5  # mild push right

    preds_fixed_action = []
    for start in range(0, n_actual, batch_size):
        z_ctx = z_all[start:start + batch_size].to(device)
        B = z_ctx.size(0)
        a = a_fixed.expand(B, -1, -1).to(device)
        with torch.no_grad():
            act_emb = jepa.action_encoder(a)
            preds_fixed_action.append(jepa.predict(z_ctx, act_emb)[:, -1].cpu())

    preds_fixed_action = torch.cat(preds_fixed_action, dim=0)  # (N, D)

    # Fixed state (mean context), vary actions
    z_mean = z_all.mean(dim=0, keepdim=True)  # (1, H, D)
    n_action_samples = 200
    random_actions = torch.from_numpy(
        rng.standard_normal((n_action_samples, HISTORY, EFF_A_DIM)).astype(np.float32))

    preds_fixed_state = []
    for start in range(0, n_action_samples, batch_size):
        a_batch = random_actions[start:start + batch_size].to(device)
        B = a_batch.size(0)
        z_ctx = z_mean.expand(B, -1, -1).to(device)
        with torch.no_grad():
            act_emb = jepa.action_encoder(a_batch)
            preds_fixed_state.append(jepa.predict(z_ctx, act_emb)[:, -1].cpu())

    preds_fixed_state = torch.cat(preds_fixed_state, dim=0)  # (M, D)

    std_input  = preds_fixed_action.std(dim=0).mean().item()
    std_action = preds_fixed_state.std(dim=0).mean().item()
    ratio = std_input / std_action if std_action > 1e-8 else float('inf')

    print(f'  std(pred | varied states, fixed action):  {std_input:.4f}')
    print(f'  std(pred | varied actions, fixed state):  {std_action:.4f}')
    print(f'  Sensitivity ratio (input/action):         {ratio:.2f}×')
    verdict = 'INPUT-DOMINATED (Hyp F confirmed)' if ratio > 10 else \
              'BALANCED (Hyp F not confirmed — action signal present)'
    print(f'  Verdict: {verdict}')
    print(f'  [Duckietown baseline: 50.6× INPUT-DOMINATED]')
    return {'std_input': std_input, 'std_action': std_action, 'ratio': ratio}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True,
                        help='Path or s3:// URI for the le-wm _object.ckpt checkpoint')
    parser.add_argument('--data', required=True,
                        help='Path or s3:// URI for pusht_expert_train.h5')
    parser.add_argument('--n-samples', type=int, default=200,
                        help='Samples for T6/sensitivity (default 200)')
    parser.add_argument('--device', default='auto',
                        help='cpu / cuda / auto (default: auto)')
    parser.add_argument('--out', default='/tmp/pusht_diagnostics.txt')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Device: {device}')

    ckpt_path = _resolve_path(args.ckpt)
    data_path = _resolve_path(args.data)

    jepa = load_jepa(ckpt_path, device)

    t5_results   = t5_state_probe(jepa, data_path, device, n_samples=1000)
    t6_synth     = t6_action_discriminability_synthetic(jepa, device, n_contexts=500)
    t6_real      = t6_action_discriminability_real(jepa, data_path, device,
                                                    n_samples=args.n_samples)
    sens         = sensitivity_ratio(jepa, data_path, device, n_samples=args.n_samples)

    # Summary
    sep = '=' * 68
    lines = [
        sep,
        '  PUSH-T DIAGNOSTICS SUMMARY',
        sep,
        '',
        '  T5 — State probe R²',
    ]
    for name, r2 in t5_results.items():
        status = 'PASS' if r2 > 0.95 else ('WARN' if r2 > 0.80 else 'FAIL')
        lines.append(f'    {name:<15}: R²={r2:.4f}  {status}')
    lines += [
        '    [Paper target: > 0.99 for block/agent positions]',
        '',
        '  T6 — Action discriminability (synthetic contexts)',
        f'    ratio L2(right,left)/noise : {t6_synth["ratio_rl"]:.2f}×  '
        f'{"PASS" if t6_synth["ratio_rl"] > 2 else "FAIL"}',
        f'    ratio L2(up,down)/noise    : {t6_synth["ratio_ud"]:.2f}×',
        '',
        '  T6 — Action discriminability (real frames)',
        f'    ratio L2(right,left)/noise : {t6_real["ratio"]:.2f}×  '
        f'{"PASS" if t6_real["ratio"] > 2 else "FAIL"}',
        '    [Duckietown baseline: 0.86× FAIL]',
        '',
        '  Sensitivity ratio (input vs action)',
        f'    std_input  : {sens["std_input"]:.4f}',
        f'    std_action : {sens["std_action"]:.4f}',
        f'    ratio      : {sens["ratio"]:.2f}×',
        '    [Duckietown baseline: 50.6× INPUT-DOMINATED]',
        sep,
    ]
    summary = '\n'.join(lines)
    print('\n' + summary)

    with open(args.out, 'w') as f:
        f.write(summary + '\n')
    print(f'\nResults written to {args.out}')


if __name__ == '__main__':
    main()

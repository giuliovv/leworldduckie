#!/usr/bin/env python3
"""
Behavior-cloning baseline for Duckietown lane-following.

Training:  (z_t, a_t) pairs from latent_index + HDF5  →  MLP (frozen encoder)
Eval:      obs → encoder → z → MLP → action  (10 episodes, same setup as MPC eval)

Usage:
    python3 bc_controller.py \
        --ckpt s3://leworldduckie/training/runs/colab_v1/checkpoint_best.pt \
        --latent-index s3://leworldduckie/evals/latent_index.npz \
        --data-path s3://leworldduckie/data/duckietown_100k.h5 \
        --s3-output s3://leworldduckie/evals/bc/<run_id>/
"""

import argparse, os, subprocess, sys
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LEWM_DIR   = Path('/home/ubuntu/le-wm') if Path('/home/ubuntu/le-wm').exists() \
             else Path('/tmp/le-wm')
IMG_SIZE   = 224
EMBED_DIM  = 192
ACTION_DIM = 2
BC_HISTORY = 3   # frames to stack as input context
HISTORY    = 3
FRAMESKIP  = 1
LAG_FRAMES = 4

ACT_LO = np.array([0.1, -1.0], dtype=np.float32)
ACT_HI = np.array([0.6,  1.0], dtype=np.float32)


# ── model loading (mirrors mpc_controller.py) ─────────────────────────────────

def _ensure_lewm():
    if not LEWM_DIR.exists():
        subprocess.run(['git', 'clone', '--depth', '1',
                        'https://github.com/lucas-maes/le-wm.git', str(LEWM_DIR)],
                       check=True)
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
    for p in model.parameters():
        p.requires_grad_(False)
    print(f'Loaded encoder from {ckpt_path}')
    return model


def preprocess(frame, device):
    t = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
    return (t * 2.0 - 1.0).squeeze(0).to(device)


def encode_obs(model, frame, device):
    px = preprocess(frame, device).unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
    with torch.no_grad():
        return model.encode({'pixels': px})['emb'][0, 0]      # (D,)


# ── BC training ───────────────────────────────────────────────────────────────

class BCPolicy(nn.Module):
    def __init__(self, in_dim=EMBED_DIM * BC_HISTORY, out_dim=ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 64),    nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, z):
        return self.net(z)


def _s3_download(s3_uri, local_path):
    import boto3
    u = urlparse(s3_uri)
    boto3.client('s3', region_name='us-east-1').download_file(
        u.netloc, u.path.lstrip('/'), str(local_path))


def build_training_data(latent_index_path, data_path):
    if latent_index_path.startswith('s3://'):
        local = '/tmp/latent_index_bc.npz'
        print(f'Downloading latent index ...')
        _s3_download(latent_index_path, local)
        latent_index_path = local

    if data_path.startswith('s3://'):
        local = '/tmp/duckietown_bc.h5'
        print(f'Downloading HDF5 ...')
        _s3_download(data_path, local)
        data_path = local

    print('Loading latent index ...')
    d       = np.load(latent_index_path)
    all_z   = d['all_z']    # (N, 192)
    ep_idx  = d['ep_idx']   # (N,)
    st_idx  = d['step_idx'] # (N,)

    print('Building action lookup from HDF5 ...')
    import h5py
    with h5py.File(data_path, 'r') as f:
        h5_ep   = f['episode_idx'][:]  # (M,)
        h5_st   = f['step_idx'][:]     # (M,)
        h5_act  = f['action'][:]       # (M, 2)

    lookup = {(int(e), int(s)): h5_act[i]
              for i, (e, s) in enumerate(zip(h5_ep, h5_st))}

    print(f'Building {BC_HISTORY}-frame context windows ...')
    ep_to_frames = defaultdict(list)
    for i, (ep, step) in enumerate(zip(ep_idx, st_idx)):
        ep_to_frames[int(ep)].append((int(step), i))
    for ep in ep_to_frames:
        ep_to_frames[ep].sort()

    z_windows, act_list = [], []
    for ep, frames in ep_to_frames.items():
        for j in range(len(frames) - (BC_HISTORY - 1)):
            window = frames[j:j + BC_HISTORY]
            steps  = [s for s, _ in window]
            idxs   = [i for _, i in window]
            if all(steps[k+1] == steps[k] + 1 for k in range(len(steps) - 1)):
                act = lookup.get((ep, steps[-1]))
                if act is not None:
                    z_windows.append(all_z[idxs].reshape(-1))
                    act_list.append(act)

    print(f'Built {len(z_windows)} context windows from {len(all_z)} frames')
    return np.array(z_windows, dtype=np.float32), np.array(act_list, dtype=np.float32)


def train_bc(all_z, actions, epochs=50, device='cpu'):
    N = len(all_z)
    perm = np.random.permutation(N)
    split = int(0.8 * N)
    tr_idx, va_idx = perm[:split], perm[split:]

    z_tr = torch.tensor(all_z[tr_idx],    device=device)
    a_tr = torch.tensor(actions[tr_idx],  device=device)
    z_va = torch.tensor(all_z[va_idx],    device=device)
    a_va = torch.tensor(actions[va_idx],  device=device)

    policy = BCPolicy().to(device)
    opt    = torch.optim.Adam(policy.parameters(), lr=1e-3)
    bs     = 2048

    print(f'Training BC: {len(tr_idx)} train, {len(va_idx)} val, {epochs} epochs')
    for ep in range(1, epochs + 1):
        policy.train()
        perm2 = torch.randperm(len(z_tr), device=device)
        losses = []
        for i in range(0, len(z_tr), bs):
            idx = perm2[i:i+bs]
            loss = F.mse_loss(policy(z_tr[idx]), a_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if ep % 10 == 0 or ep == epochs:
            policy.eval()
            with torch.no_grad():
                val_loss = F.mse_loss(policy(z_va), a_va).item()
            print(f'  epoch {ep:3d}  train={np.mean(losses):.4f}  val={val_loss:.4f}')

    return policy


# ── Duckietown eval ───────────────────────────────────────────────────────────

def _patch_pwm_dynamics():
    try:
        import duckietown_world
        path = Path(duckietown_world.__file__).parent / \
               'world_duckietown/pwm_dynamics.py'
        txt = path.read_text()
        old = '        linear = [longitudinal, lateral]'
        new = '        linear = [float(longitudinal), lateral]'
        if old in txt:
            path.write_text(txt.replace(old, new))
    except Exception:
        pass


def setup_duckietown():
    _patch_pwm_dynamics()
    try:
        dt_world = subprocess.check_output(
            ['python3', '-c',
             'import duckietown_world,os;print(os.path.dirname(duckietown_world.__file__))'],
            stderr=subprocess.DEVNULL).decode().strip().splitlines()[-1]
        dt_gym = subprocess.check_output(
            ['python3', '-c',
             'import gym_duckietown,os;print(os.path.dirname(gym_duckietown.__file__))'],
            stderr=subprocess.DEVNULL).decode().strip().splitlines()[-1]
        maps_dst = Path(dt_gym) / 'maps'
        if not maps_dst.exists():
            maps_dst.symlink_to(Path(dt_world) / 'data/gd1/maps')
    except Exception:
        pass


def run_eval(model, policy, episodes, steps, map_name, device, seed=42, gif_dir=None):
    from gym_duckietown.envs import DuckietownEnv
    import imageio

    rng = np.random.default_rng(seed)
    results = []

    for ep in range(episodes):
        ep_map = map_name or rng.choice(['loop_empty', 'straight_road', 'udem1'])
        env = DuckietownEnv(map_name=ep_map, seed=int(rng.integers(1e6)),
                            draw_curve=False, draw_bbox=False,
                            domain_rand=True, distortion=False,
                            accept_start_angle_deg=4, full_transparency=True)
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        # Lag frames
        for _ in range(LAG_FRAMES):
            result = env.step([0.2, 0.0])
            obs, done = result[0], result[2]
            if done:
                break

        ep_reward, ep_steps = 0.0, 0
        frames = []
        done = False
        emb_buf = deque(maxlen=BC_HISTORY)

        while not done and ep_steps < steps:
            emb_buf.append(encode_obs(model, obs, device))
            buf_list = list(emb_buf)
            if len(buf_list) < BC_HISTORY:
                buf_list = [buf_list[0]] * (BC_HISTORY - len(buf_list)) + buf_list
            z_ctx = torch.cat(buf_list)  # (BC_HISTORY * EMBED_DIM,)
            with torch.no_grad():
                act_raw = policy(z_ctx).cpu().numpy()
            action = np.clip(act_raw, ACT_LO, ACT_HI)

            result = env.step(action)
            obs, reward, done = result[0], result[1], result[2]
            ep_reward += reward
            ep_steps  += 1
            if gif_dir:
                frames.append(obs)

        success = ep_steps >= steps and not done
        results.append({'ep': ep, 'steps': ep_steps, 'reward': ep_reward,
                        'success': success, 'map': ep_map})
        print(f'ep={ep}  map={ep_map}  steps={ep_steps}  '
              f'reward={ep_reward:.2f}  {"SUCCESS" if success else "fail"}')

        if gif_dir and frames:
            Path(gif_dir).mkdir(parents=True, exist_ok=True)
            imageio.mimsave(f'{gif_dir}/ep{ep}.gif',
                            [f[::2, ::2] for f in frames[::3]], fps=10)
        env.close()

    return results


def upload_results(results, s3_output, gif_dir=None):
    import boto3
    u = urlparse(s3_output)
    bucket = u.netloc
    prefix = u.path.lstrip('/')
    s3 = boto3.client('s3', region_name='us-east-1')

    n_success = sum(r['success'] for r in results)
    mean_steps = np.mean([r['steps'] for r in results])
    mean_reward = np.mean([r['reward'] for r in results])

    lines = [
        '══════════════════════════════════════════',
        'BC Baseline Eval Results',
        '══════════════════════════════════════════',
        f'Success: {n_success}/{len(results)}',
        f'Mean steps: {mean_steps:.1f}',
        f'Mean reward: {mean_reward:.3f}',
        '',
        'Per episode:',
    ]
    for r in results:
        lines.append(f"  ep={r['ep']}  map={r['map']}  steps={r['steps']}  "
                     f"reward={r['reward']:.2f}  {'SUCCESS' if r['success'] else 'fail'}")

    report = '\n'.join(lines)
    print(report)

    key = prefix + 'bc_results.txt'
    s3.put_object(Bucket=bucket, Key=key, Body=report.encode())
    print(f'Uploaded s3://{bucket}/{key}')

    if gif_dir:
        import glob
        for g in sorted(glob.glob(f'{gif_dir}/ep*.gif')):
            gkey = prefix + Path(g).name
            s3.upload_file(g, bucket, gkey)
            print(f'Uploaded s3://{bucket}/{gkey}')


def _s3_put(s3_uri, body: bytes):
    import boto3
    u = urlparse(s3_uri)
    boto3.client('s3', region_name='us-east-1').put_object(
        Bucket=u.netloc, Key=u.path.lstrip('/'), Body=body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',         required=True)
    ap.add_argument('--latent-index', required=True)
    ap.add_argument('--data-path',    required=True)
    ap.add_argument('--s3-output',    default=None)
    ap.add_argument('--episodes',     type=int, default=10)
    ap.add_argument('--steps',        type=int, default=300)
    ap.add_argument('--train-epochs', type=int, default=50)
    ap.add_argument('--map',          default=None)
    ap.add_argument('--gif-dir',      default='/tmp/bc_gifs')
    ap.add_argument('--seed',         type=int, default=42)
    args = ap.parse_args()

    device = torch.device('cpu')
    print(f'Device: {device}')

    try:
        print('=== Phase 1: build training data ===')
        all_z, actions = build_training_data(args.latent_index, args.data_path)
        if args.s3_output:
            _s3_put(args.s3_output + 'phase1_done.txt',
                    f'training_data_built n={len(all_z)}'.encode())

        print('=== Phase 2: train BC policy ===')
        policy = train_bc(all_z, actions, epochs=args.train_epochs, device=str(device))
        if args.s3_output:
            _s3_put(args.s3_output + 'phase2_done.txt', b'training_done')

        print('=== Phase 3: load encoder ===')
        model = load_model(args.ckpt, device)
        setup_duckietown()
        if args.s3_output:
            _s3_put(args.s3_output + 'phase3_done.txt', b'encoder_loaded')

        print('=== Phase 4: eval ===')
        results = run_eval(model, policy, episodes=args.episodes, steps=args.steps,
                           map_name=args.map, device=device, seed=args.seed,
                           gif_dir=args.gif_dir)

        if args.s3_output:
            upload_results(results, args.s3_output, gif_dir=args.gif_dir)
        else:
            n = sum(r['success'] for r in results)
            print(f'\nSuccess: {n}/{len(results)}  '
                  f'mean_steps={np.mean([r["steps"] for r in results]):.1f}')

    except Exception:
        import traceback
        tb = traceback.format_exc()
        print(f'FATAL ERROR:\n{tb}', file=sys.stderr)
        if args.s3_output:
            try:
                _s3_put(args.s3_output + 'error.txt', tb.encode())
                print('Error traceback uploaded to S3.')
            except Exception as e2:
                print(f'Also failed to upload error: {e2}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

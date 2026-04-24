#!/usr/bin/env python3
"""
MPC lane-following controller for gym-duckietown using a trained LeWM checkpoint.

Receding-horizon MPC: CEM as inner optimizer, LeWM predictor as dynamics model.
Cost = trajectory MSE in latent space to a goal embedding + action smoothness.

Usage:
    python src/mpc_controller.py --ckpt data/lewm_best.pt --episodes 10
    python src/mpc_controller.py --ckpt s3://leworldduckie/training/runs/notebook/checkpoint_best.pt
    python src/mpc_controller.py --goal /path/to/goal.png --video out.mp4
"""

import argparse, os, subprocess, sys, time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants ──────────────────────────────────────────────────────────────────
LEWM_DIR   = Path('/home/ubuntu/le-wm') if Path('/home/ubuntu/le-wm').exists() \
             else Path('/tmp/le-wm')
IMG_SIZE   = 224
EMBED_DIM  = 192
ACTION_DIM = 2
HISTORY    = 3   # predictor context window (= notebook HISTORY)
FRAMESKIP  = 1   # env steps per predictor step — must match notebook FRAMESKIP
LAG_FRAMES = 4   # duckietown PWM warm-up steps to discard at episode start

# Action bounds: [velocity, steering]
ACT_LO = np.array([0.1, -1.0], dtype=np.float32)
ACT_HI = np.array([0.6,  1.0], dtype=np.float32)


# ── Environment / le-wm setup ─────────────────────────────────────────────────
def _ensure_lewm():
    if not LEWM_DIR.exists():
        print(f'Cloning le-wm → {LEWM_DIR}')
        r = subprocess.run(['git', 'clone', '--depth', '1',
                            'https://github.com/lucas-maes/le-wm.git', str(LEWM_DIR)])
        if r.returncode != 0:
            raise RuntimeError('git clone le-wm failed')
    if str(LEWM_DIR) not in sys.path:
        sys.path.insert(0, str(LEWM_DIR))


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
             'import duckietown_world,os;'
             'print(os.path.dirname(duckietown_world.__file__))'],
            stderr=subprocess.DEVNULL).decode().strip().splitlines()[-1]
        dt_gym = subprocess.check_output(
            ['python3', '-c',
             'import gym_duckietown,os;'
             'print(os.path.dirname(gym_duckietown.__file__))'],
            stderr=subprocess.DEVNULL).decode().strip().splitlines()[-1]
        maps_dst = Path(dt_gym) / 'maps'
        if not maps_dst.exists():
            maps_dst.symlink_to(Path(dt_world) / 'data/gd1/maps')
    except Exception:
        pass


# ── Model ──────────────────────────────────────────────────────────────────────
def _build_model(device):
    _ensure_lewm()
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    try:
        import stable_pretraining as spt
        encoder = spt.backbone.utils.vit_hf(
            'tiny', patch_size=14, image_size=IMG_SIZE,
            pretrained=False, use_mask_token=False)
    except Exception as e:
        raise RuntimeError(f'ViT-Tiny unavailable: {e}')

    projector      = MLP(EMBED_DIM, 2048, EMBED_DIM, norm_fn=nn.BatchNorm1d)
    pred_proj      = MLP(EMBED_DIM, 2048, EMBED_DIM, norm_fn=nn.BatchNorm1d)
    action_encoder = Embedder(input_dim=ACTION_DIM, smoothed_dim=ACTION_DIM,
                               emb_dim=EMBED_DIM, mlp_scale=4)
    predictor      = ARPredictor(num_frames=HISTORY, input_dim=EMBED_DIM,
                                  hidden_dim=EMBED_DIM, output_dim=EMBED_DIM,
                                  depth=6, heads=16, dim_head=64,
                                  mlp_dim=2048, dropout=0.1)
    return JEPA(encoder, predictor, action_encoder, projector,
                pred_proj).to(device)


def load_model(ckpt_path: str, device) -> nn.Module:
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
    print(f'Loaded: {ckpt_path}  ({sum(p.numel() for p in model.parameters()):,} params)')
    return model


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(frame: np.ndarray, device) -> torch.Tensor:
    """(H, W, 3) uint8 → (3, IMG_SIZE, IMG_SIZE) float [-1, 1]."""
    t = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
    return (t * 2.0 - 1.0).squeeze(0).to(device)


# ── Goal ───────────────────────────────────────────────────────────────────────
def load_goal(goal_arg, data_path: str, model, device) -> torch.Tensor:
    """
    Returns z_goal: (D,) float tensor.
    Uses goal_arg (PNG path) if given; otherwise auto-selects the frame from
    the training HDF5 with the smallest |steering| (proxy for centered lane).
    """
    if goal_arg and Path(goal_arg).exists():
        from PIL import Image
        frame = np.array(Image.open(goal_arg).convert('RGB'))
        print(f'Goal: {goal_arg}')
    else:
        import h5py
        print(f'Auto-selecting goal from {data_path} ...')
        with h5py.File(data_path, 'r') as f:
            actions = f['action'][:]
            idx     = int(np.argmin(np.abs(actions[:, 1])))
            frame   = f['pixels'][idx]
        print(f'  idx={idx}  steering={actions[idx, 1]:.4f}')

    px = preprocess(frame, device).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, H, W)
    with torch.no_grad():
        z_goal = model.encode({'pixels': px})['emb'][0, 0]    # (D,)
    return z_goal


# ── CEM Planner ────────────────────────────────────────────────────────────────
@torch.no_grad()
def cem_plan(
    model,
    ctx_embs: torch.Tensor,       # (HISTORY, D)   pre-encoded frame embeddings
    ctx_act_past: torch.Tensor,   # (HISTORY-1, D) pre-encoded past action embeddings
    z_goal: torch.Tensor,         # (D,)
    device,
    horizon: int = 10,
    n_samples: int = 200,
    n_iters: int = 3,
    warm_start=None,              # (horizon, 2) | None
    vel_weight: float = 1.0,      # reward for forward velocity (breaks spinning local min)
    steer_weight: float = 0.1,    # penalise large steering
) -> torch.Tensor:
    """
    Cross-Entropy Method planning.

    At each CEM step k the predictor sees:
        emb_window:  [z_{t+k-H+1}, ..., z_{t+k}]         (HISTORY frames)
        act_window:  [a_{t+k-H+1}, ..., a_{t+k}^(s)]     (HISTORY actions)
    where the first HISTORY-1 actions in act_window shift from real history to
    sampled ones as k increases.  Returns (horizon, 2) best action sequence.
    """
    elite_k = max(3, n_samples // 10)
    lo = torch.tensor(ACT_LO, device=device)
    hi = torch.tensor(ACT_HI, device=device)

    # Initialise distribution — forward-biased velocity, neutral steering
    mu = torch.zeros(horizon, ACTION_DIM, device=device)
    mu[:, 0] = 0.35
    if warm_start is not None:
        mu = warm_start.clone().to(device)
    sigma = torch.tensor([0.10, 0.30], device=device).unsqueeze(0).expand(horizon, -1).clone()

    # Expand context to N parallel rollouts
    ctx_e = ctx_embs.unsqueeze(0).expand(n_samples, -1, -1)   # (N, H, D)
    ctx_a = ctx_act_past.unsqueeze(0).expand(n_samples, -1, -1)  # (N, H-1, D)

    z_goal_exp = z_goal.unsqueeze(0).unsqueeze(0)  # (1, 1, D) — broadcast over N and T

    for _ in range(n_iters):
        # Sample and clamp to action bounds: (N, horizon, 2)
        noise      = torch.randn(n_samples, horizon, ACTION_DIM, device=device)
        candidates = torch.clamp(mu + sigma * noise, lo, hi)

        # Encode all sampled future actions in one batch: (N, horizon, D)
        fut_act_embs = model.action_encoder(candidates)

        # Rolling-window rollout
        emb_win = ctx_e.clone()   # (N, H, D)
        act_win = ctx_a.clone()   # (N, H-1, D)

        pred_embs = []
        for k in range(horizon):
            # Append current sampled action to make full H-length context
            full_act = torch.cat(
                [act_win, fut_act_embs[:, k:k+1]], dim=1)      # (N, H, D)
            # predict()[:, -1] ≈ next embedding after applying full_act[-1]
            next_emb = model.predict(emb_win, full_act)[:, -1]  # (N, D)
            pred_embs.append(next_emb)
            # Slide both windows
            emb_win = torch.cat([emb_win[:, 1:], next_emb.unsqueeze(1)], dim=1)
            act_win = full_act[:, 1:]                            # drop oldest

        pred_embs = torch.stack(pred_embs, dim=1)  # (N, horizon, D)

        # Cost: latent MSE to goal + smoothness + action prior
        traj_cost   = (pred_embs - z_goal_exp).pow(2).mean(dim=-1).sum(dim=-1)  # (N,)
        smooth      = 0.01 * (candidates[:, 1:] - candidates[:, :-1]).pow(2)\
                            .sum(dim=(-1, -2))                                    # (N,)
        # vel_weight rewards forward progress; steer_weight penalises spinning
        vel_reward  = vel_weight  * candidates[:, :, 0].mean(dim=-1)             # (N,)
        steer_cost  = steer_weight * candidates[:, :, 1].abs().mean(dim=-1)      # (N,)
        costs = traj_cost + smooth - vel_reward + steer_cost

        # Refit distribution with top-K elite samples
        _, idx   = torch.topk(costs, elite_k, largest=False)
        elite    = candidates[idx]        # (K, horizon, 2)
        mu       = elite.mean(dim=0)
        sigma    = elite.std(dim=0).clamp(min=1e-3)

    return mu   # (horizon, 2)


# ── Episode runner ─────────────────────────────────────────────────────────────
def _lane_follow(env, rng: np.random.Generator) -> np.ndarray:
    """Ground-truth lane follower — used for warm-up frames."""
    try:
        lp    = env.get_lane_pos2(env.cur_pos, env.cur_angle)
        steer = 10.0 * lp.dist + 5.0 * lp.angle_rad
        vel   = float(np.clip(0.35 + rng.normal(0, 0.03), 0.1, 0.6))
        steer = float(np.clip(steer + rng.normal(0, 0.03), -1.0, 1.0))
        return np.array([vel, steer], dtype=np.float32)
    except Exception:
        return np.array([0.35, 0.0], dtype=np.float32)


def run_episode(model, z_goal, env, ep_idx: int, args, device,
                video_frames=None) -> dict:
    """
    Run one episode.  Returns dict with success, n_steps, mean_reward, mean_ms.

    Frameskip: each planning step executes the chosen action for args.frameskip
    raw env steps, matching the training dataset which sampled every 3rd frame.
    The context buffer is updated once per planning step (not per raw step).

    Lag frames: the first args.lag_frames raw steps use LaneFollower without
    filling the context buffer, matching the dataset's skip_initial_steps=LAG_FRAMES.
    """
    rng  = np.random.default_rng(args.seed + ep_idx)
    obs  = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    # --- Burn through PWM lag frames (not stored in context) ---
    early_done = False
    for _ in range(args.lag_frames):
        action = _lane_follow(env, rng)
        for _ in range(args.frameskip):
            result = env.step(action)
            obs, _, early_done = result[0], result[1], result[2]
            if early_done:
                break
        if early_done:
            break

    frame_buf  = deque(maxlen=HISTORY)     # preprocessed (3, 224, 224) tensors
    action_buf = deque(maxlen=HISTORY)     # raw (2,) numpy actions
    warm_start = None

    rewards    = []
    step_times = []
    z_dist     = 0.0
    done       = False

    for step in range(args.steps):
        t0 = time.time()

        if video_frames is not None:
            video_frames.append(obs.copy())

        px = preprocess(obs, device)
        frame_buf.append(px)

        if len(frame_buf) < HISTORY:
            # Warm-up: collect real context frames before MPC can run
            action = _lane_follow(env, rng)
        else:
            # Encode context frames once (reused across all CEM iterations)
            ctx_pixels = torch.stack(list(frame_buf))        # (H, 3, 224, 224)
            with torch.no_grad():
                ctx_embs = model.encode(
                    {'pixels': ctx_pixels.unsqueeze(0)}      # (1, H, 3, 224, 224)
                )['emb'][0]                                   # (H, D)

            # Past H-1 actions
            past = list(action_buf)[-(HISTORY - 1):]
            while len(past) < HISTORY - 1:
                past.insert(0, np.array([0.35, 0.0], dtype=np.float32))
            past_acts = torch.from_numpy(
                np.array(past, dtype=np.float32)).to(device)  # (H-1, 2)
            with torch.no_grad():
                ctx_act_past = model.action_encoder(
                    past_acts.unsqueeze(0))[0]               # (H-1, D)

            # Warm-start: shift previous plan by one step
            ws = None
            if warm_start is not None:
                fwd = torch.tensor([[0.35, 0.0]], device=device)
                ws  = torch.cat([warm_start[1:], fwd], dim=0)

            z_dist = (ctx_embs[-1] - z_goal).norm().item()

            plan = cem_plan(
                model, ctx_embs, ctx_act_past, z_goal, device,
                horizon=args.horizon, n_samples=args.n_samples,
                n_iters=args.n_iters, warm_start=ws,
                vel_weight=args.vel_weight, steer_weight=args.steer_weight)
            warm_start = plan
            action     = plan[0].cpu().numpy()

        action = np.clip(action, ACT_LO, ACT_HI).astype(np.float32)
        action_buf.append(action)

        # Execute action for FRAMESKIP raw env steps; observe after last step.
        # Matches training: dataset[i].action = action at raw step i*frameskip,
        # which caused the transition to the frame at raw step (i+1)*frameskip.
        total_reward = 0.0
        for _ in range(args.frameskip):
            result = env.step(action)
            obs, rew, done = result[0], result[1], result[2]
            total_reward += float(rew)
            if done:
                break
        reward = total_reward

        rewards.append(reward)
        step_times.append(time.time() - t0)

        if args.verbose:
            zdist_str = f' z_dist={z_dist:.3f}' if step >= HISTORY else ''
            try:
                lp = env.get_lane_pos2(env.cur_pos, env.cur_angle)
                print(f'  ep={ep_idx} t={step:3d} r={reward:.3f} '
                      f'lane_off={lp.dist:.3f} '
                      f'a=[{action[0]:.2f},{action[1]:+.2f}]'
                      f'{zdist_str} {step_times[-1]*1e3:.0f}ms')
            except Exception:
                print(f'  ep={ep_idx} t={step:3d} r={reward:.3f} '
                      f'a=[{action[0]:.2f},{action[1]:+.2f}]'
                      f'{zdist_str} {step_times[-1]*1e3:.0f}ms')

        if done:
            break

    success  = not done or step >= args.steps - 1
    n_steps  = len(rewards)
    mean_rew = float(np.mean(rewards)) if rewards else 0.0
    # Only count planning steps (skip warm-up) in timing
    plan_times = step_times[HISTORY:]
    mean_ms  = float(np.mean(plan_times) * 1000) if plan_times else 0.0

    return {
        'episode':     ep_idx,
        'success':     success,
        'n_steps':     n_steps,
        'mean_reward': mean_rew,
        'mean_ms':     mean_ms,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='LeWM MPC lane-following controller')
    ap.add_argument('--ckpt',       default='data/lewm_best.pt',
                    help='Checkpoint path or s3://bucket/key')
    ap.add_argument('--goal',       default=None,
                    help='Goal frame PNG. Auto-selected from HDF5 if omitted.')
    ap.add_argument('--data-path',  default='data/duckietown_100k.h5',
                    help='HDF5 dataset (used for auto goal selection)')
    ap.add_argument('--map',        default=None,
                    help='Duckietown map name. Randomized per episode if omitted.')
    ap.add_argument('--frameskip',   type=int, default=FRAMESKIP,
                    help='Raw env steps per planning step (must match training)')
    ap.add_argument('--lag-frames', type=int, default=LAG_FRAMES,
                    help='Initial PWM-lag env steps to discard (must match training)')
    ap.add_argument('--steps',      type=int, default=300,
                    help='Max planning steps per episode (each = frameskip raw steps)')
    ap.add_argument('--episodes',   type=int, default=1)
    ap.add_argument('--horizon',    type=int, default=10,
                    help='CEM planning horizon')
    ap.add_argument('--n-samples',  type=int, default=200,
                    help='CEM action samples per iteration')
    ap.add_argument('--n-iters',    type=int, default=3,
                    help='CEM refinement iterations')
    ap.add_argument('--vel-weight',   type=float, default=1.0,
                    help='Reward weight for forward velocity in CEM cost (breaks spinning)')
    ap.add_argument('--steer-weight', type=float, default=0.1,
                    help='Penalty weight for large steering in CEM cost')
    ap.add_argument('--seed',       type=int, default=42)
    ap.add_argument('--video',      default=None,
                    help='Save last episode video to this MP4 path')
    ap.add_argument('--gif',        default=None,
                    help='Save best episode (most steps, then reward) as GIF to this path')
    ap.add_argument('--gif-all',    default=None,
                    help='Directory to save a GIF for every episode (ep0.gif, ep1.gif, ...)')
    ap.add_argument('--s3-progress', default=None,
                    help='S3 URI (s3://bucket/key) to upload a progress summary after each episode')
    ap.add_argument('--verbose',    action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'CEM: horizon={args.horizon}  N={args.n_samples}  iters={args.n_iters}')

    _ensure_lewm()
    setup_duckietown()

    model  = load_model(args.ckpt, device)
    z_goal = load_goal(args.goal, args.data_path, model, device)

    from gym_duckietown.envs import DuckietownEnv
    rng_env = np.random.default_rng(args.seed)

    s3_progress_bucket = s3_progress_key = None
    if args.s3_progress:
        _parts = args.s3_progress[len('s3://'):].split('/', 1)
        s3_progress_bucket, s3_progress_key = _parts[0], _parts[1]

    def _upload_progress(stats_so_far):
        if not s3_progress_bucket:
            return
        try:
            import boto3, io
            lines = [f'ep={s["ep"]}  success={s["success"]}  steps={s["n_steps"]}  '
                     f'reward={s["mean_reward"]:.3f}  {s["mean_ms"]:.0f}ms/step'
                     for s in stats_so_far]
            done = len(stats_so_far)
            total = args.episodes
            lines.append(f'--- {done}/{total} episodes done ---')
            body = '\n'.join(lines) + '\n'
            boto3.client('s3').put_object(Bucket=s3_progress_bucket,
                                          Key=s3_progress_key,
                                          Body=body.encode())
        except Exception as e:
            print(f'progress upload failed: {e}')

    all_stats   = []
    best_frames = []   # frames of the best episode so far
    best_key    = (-1, -float('inf'))  # (n_steps, mean_reward)

    if args.gif_all:
        import os; os.makedirs(args.gif_all, exist_ok=True)

    for ep in range(args.episodes):
        ep_seed = int(rng_env.integers(0, 2**31))
        record  = args.video and (ep == args.episodes - 1)
        vframes = [] if (record or args.gif or args.gif_all) else None

        kw = dict(distortion=False, max_steps=args.steps + 20, seed=ep_seed)
        if args.map:
            kw['map_name'] = args.map
        else:
            kw['randomize_maps_on_reset'] = True

        env   = DuckietownEnv(**kw)
        stats = run_episode(model, z_goal, env, ep, args, device, vframes)
        stats['ep'] = ep
        env.close()
        all_stats.append(stats)

        print(f'Episode {ep}: success={stats["success"]}  '
              f'steps={stats["n_steps"]}  '
              f'reward={stats["mean_reward"]:.3f}  '
              f'{stats["mean_ms"]:.0f}ms/step')

        _upload_progress(all_stats)

        if args.gif_all and vframes:
            try:
                import imageio, os
                ep_path = os.path.join(args.gif_all, f'ep{ep}.gif')
                imageio.mimwrite(ep_path, vframes, fps=10, loop=0)
                print(f'  GIF → {ep_path}')
            except Exception as e:
                print(f'  ep GIF write failed: {e}')

        if record and vframes:
            try:
                import imageio
                imageio.mimwrite(args.video, vframes, fps=30, macro_block_size=1)
                print(f'Video saved → {args.video}')
            except Exception as e:
                print(f'Video write failed: {e}')

        if args.gif and vframes:
            ep_key = (stats['n_steps'], stats['mean_reward'])
            if ep_key > best_key:
                best_key    = ep_key
                best_frames = list(vframes)

    if args.gif and best_frames:
        try:
            import imageio
            imageio.mimwrite(args.gif, best_frames, fps=10, loop=0)
            print(f'GIF saved → {args.gif}  ({len(best_frames)} frames, best ep: steps={best_key[0]} reward={best_key[1]:.3f})')
        except Exception as e:
            print(f'GIF write failed: {e}')

    if args.episodes > 1:
        n_ok     = sum(s['success']     for s in all_stats)
        avg_step = np.mean([s['n_steps']     for s in all_stats])
        avg_rew  = np.mean([s['mean_reward'] for s in all_stats])
        avg_ms   = np.mean([s['mean_ms']     for s in all_stats])
        print()
        print('── Summary ──────────────────────────────────')
        print(f'Episodes     : {args.episodes}')
        print(f'Success rate : {n_ok}/{args.episodes} ({100*n_ok/args.episodes:.0f}%)')
        print(f'Mean steps   : {avg_step:.1f}')
        print(f'Mean reward  : {avg_rew:.3f}')
        print(f'Mean ms/step : {avg_ms:.0f}  ({1000/max(avg_ms,1):.1f} steps/s)')
        print('─────────────────────────────────────────────')


if __name__ == '__main__':
    main()

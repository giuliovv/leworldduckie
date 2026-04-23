"""
run_eval.py — LeWM VoE demo on live duckietown env.

Downloads checkpoint from S3, runs N_STEPS with PD controller,
injects a teleport at --teleport-at, computes VoE surprise signal,
saves annotated GIF + summary figure + metrics JSON to S3.

Usage (on EC2):
    python run_eval.py [--run-id ID] [--ckpt s3://...] [--steps N] [--teleport-at N] [--map NAME]
"""

import os, sys, time, json, argparse, subprocess
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import boto3
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────────
S3_BUCKET    = 'leworldduckie'
DEFAULT_CKPT = 'training/runs/notebook/checkpoint_latest.pt'
LEWM_DIR     = Path('/tmp/le-wm')

IMG_H, IMG_W = 120, 160
IMG_SIZE     = 224
ACTION_DIM   = 2
EMBED_DIM    = 192
HISTORY      = 3
N_STEPS_DEF  = 200
TELEPORT_DEF = 100
SEED         = 42
GIF_FPS      = 8
CHUNK_SIZE   = 16   # frames per encode batch

MAPS = ['small_loop', 'loop_empty', 'straight_road', 'zigzag_dists', 'udem1']


def log(msg):
    ts = datetime.now(timezone.utc).strftime('%H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)


# ── S3 helpers ─────────────────────────────────────────────────────────────────
def make_s3():
    return boto3.client('s3', region_name='us-east-1')

def s3_upload(s3, local_path, s3_key):
    s3.upload_file(str(local_path), S3_BUCKET, s3_key)
    log(f's3 upload: s3://{S3_BUCKET}/{s3_key}')

def s3_download_ckpt(s3, ckpt_arg, local_path):
    if ckpt_arg.startswith('s3://'):
        bucket, key = ckpt_arg[5:].split('/', 1)
        size = s3.head_object(Bucket=bucket, Key=key)['ContentLength']
        log(f'Downloading checkpoint from s3://{bucket}/{key}  ({size/1e6:.0f} MB)')
        s3.download_file(bucket, key, str(local_path))
    else:
        import shutil
        shutil.copy(ckpt_arg, local_path)


# ── Duckietown setup ───────────────────────────────────────────────────────────
def setup_duckietown():
    dt_world = subprocess.check_output(
        ['python3', '-c',
         'import duckietown_world,os;print(os.path.dirname(duckietown_world.__file__))'],
        stderr=subprocess.DEVNULL,
    ).decode().strip().splitlines()[-1]
    dt_gym = subprocess.check_output(
        ['python3', '-c',
         'import gym_duckietown,os;print(os.path.dirname(gym_duckietown.__file__))'],
        stderr=subprocess.DEVNULL,
    ).decode().strip().splitlines()[-1]
    maps_dst = Path(dt_gym) / 'maps'
    if not maps_dst.exists():
        maps_dst.symlink_to(Path(dt_world) / 'data/gd1/maps')
        log('Maps symlinked')


# ── PD controller ──────────────────────────────────────────────────────────────
class PDController:
    def __init__(self, kp=0.45, kd=0.25, noise_std=0.06, speed=0.35):
        self.kp, self.kd, self.noise_std, self.speed = kp, kd, noise_std, speed
        self.prev_error = 0.0

    def act(self, obs, rng):
        yellow = (obs[:, :, 0] > 170) & (obs[:, :, 1] > 150) & (obs[:, :, 2] < 100)
        cx = obs.shape[1] // 2
        if yellow.any():
            _, xs = np.where(yellow)
            error = float((xs.mean() - cx) / cx)
        else:
            error = self.prev_error
        steer = -(self.kp * error + self.kd * (error - self.prev_error))
        self.prev_error = error
        vel   = float(np.clip(self.speed + rng.normal(0, 0.03), 0.1, 0.6))
        steer = float(np.clip(steer + rng.normal(0, self.noise_std), -1.0, 1.0))
        return np.array([vel, steer], dtype=np.float32)

    def reset(self):
        self.prev_error = 0.0


# ── Model ──────────────────────────────────────────────────────────────────────
def build_model(device):
    if str(LEWM_DIR) not in sys.path:
        sys.path.insert(0, str(LEWM_DIR))
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    try:
        import stable_pretraining as spt
        encoder = spt.backbone.utils.vit_hf(
            'tiny', patch_size=14, image_size=IMG_SIZE,
            pretrained=False, use_mask_token=False,
        )
        log('Encoder: ViT-Tiny')
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
    model = JEPA(encoder, predictor, action_encoder, projector, pred_proj)
    return model.to(device)


def preprocess_frames(frames_uint8, device):
    """list of (H,W,3) uint8 → (N,3,224,224) float tensor on device, normalised to [-1,1]."""
    arr = np.stack(frames_uint8).astype(np.float32) / 255.0     # (N,H,W,3)
    t   = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)  # (N,3,H,W)
    t   = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
    return t * 2.0 - 1.0


@torch.no_grad()
def encode_all(model, frames, actions, device):
    """
    Encode N frames + actions in chunks.
    Returns embs (N, D) and act_embs (N, D) as CPU tensors.
    """
    N = len(frames)
    embs_list     = []
    act_embs_list = []
    for start in range(0, N, CHUNK_SIZE):
        chunk_frames  = frames[start:start + CHUNK_SIZE]
        chunk_actions = actions[start:start + CHUNK_SIZE]
        px  = preprocess_frames(chunk_frames, device)             # (B, 3, 224, 224)
        act = torch.from_numpy(np.stack(chunk_actions)).float().to(device)  # (B, 2)
        # model.encode expects (B, T, ...) — add T=1 dim
        out = model.encode({'pixels': px.unsqueeze(1), 'action': act.unsqueeze(1)})
        embs_list.append(out['emb'][:, 0].cpu())       # (B, D)
        act_embs_list.append(out['act_emb'][:, 0].cpu())
    return torch.cat(embs_list, dim=0), torch.cat(act_embs_list, dim=0)


@torch.no_grad()
def compute_voe(model, embs, act_embs, device):
    """
    Sliding-window VoE: for each step t >= HISTORY, predict embedding t from
    context [t-HISTORY, ..., t-1] and return MSE against actual emb t.
    Returns list of length (N - HISTORY).
    """
    N       = embs.shape[0]
    embs    = embs.to(device)
    act_embs = act_embs.to(device)
    surprise = []
    ctx_emb  = embs[:HISTORY].unsqueeze(0)          # (1, H, D)
    for t in range(HISTORY, N):
        ctx_act  = act_embs[t - HISTORY:t].unsqueeze(0)  # (1, H, D)
        pred     = model.predict(ctx_emb, ctx_act)[:, -1]  # (1, D)
        actual   = embs[t].unsqueeze(0)                    # (1, D)
        surprise.append((pred - actual).pow(2).mean().item())
        ctx_emb  = torch.cat([ctx_emb[:, 1:], actual.unsqueeze(1)], dim=1)
    return surprise


# ── Visualisation ──────────────────────────────────────────────────────────────
def make_gif_frame(obs_frame, step, surprise_so_far, teleport_at, n_steps):
    from PIL import Image, ImageDraw, ImageFont

    # left panel: camera
    cam_h, cam_w = 240, 320
    cam = Image.fromarray(obs_frame).resize((cam_w, cam_h), Image.BILINEAR)

    # right panel: VoE sparkline
    fig, ax = plt.subplots(figsize=(3.2, 2.4), dpi=100)
    if surprise_so_far:
        xs = range(HISTORY, HISTORY + len(surprise_so_far))
        ax.plot(list(xs), surprise_so_far, color='royalblue', lw=1.3)
        ax.axvline(teleport_at, color='red', ls='--', lw=1.2, alpha=0.8)
        if step >= HISTORY and surprise_so_far:
            ax.scatter([HISTORY + len(surprise_so_far) - 1], [surprise_so_far[-1]],
                       color='orange', zorder=5, s=25)
    ax.set_xlim(0, n_steps)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Step', fontsize=7)
    ax.set_ylabel('Surprise', fontsize=7)
    ax.set_title('VoE Signal', fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.25)
    plt.tight_layout(pad=0.4)
    fig.canvas.draw()
    plot_arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    plot_img = Image.fromarray(plot_arr).resize((cam_w, cam_h), Image.BILINEAR)

    # compose side by side
    out = Image.new('RGB', (cam_w * 2, cam_h + 24), (30, 30, 30))
    out.paste(cam, (0, 24))
    out.paste(plot_img, (cam_w, 24))

    # header bar
    draw = ImageDraw.Draw(out)
    header = f'Step {step:3d}/{n_steps}  |  LeWM Duckietown VoE Demo'
    if step == teleport_at:
        header += '  *** TELEPORT ***'
        draw.rectangle([0, 0, cam_w * 2, 24], fill=(180, 30, 30))
    draw.text((6, 4), header, fill=(230, 230, 230))

    return np.array(out)


def make_summary_figure(frames, surprise, teleport_at, n_steps, path):
    n_sample = 8
    sample_idx = np.linspace(0, len(frames) - 1, n_sample, dtype=int)

    fig = plt.figure(figsize=(16, 5))
    gs  = plt.GridSpec(2, n_sample, figure=fig, hspace=0.4, wspace=0.05)

    for i, si in enumerate(sample_idx):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(frames[si])
        ax.set_title(f't={si}', fontsize=7)
        ax.axis('off')

    ax_voe = fig.add_subplot(gs[1, :])
    if surprise:
        xs = list(range(HISTORY, HISTORY + len(surprise)))
        ax_voe.plot(xs, surprise, color='royalblue', lw=1.5, label='Surprise (MSE)')
        ax_voe.axvline(teleport_at, color='red', ls='--', lw=1.5,
                       label=f'Teleport injected (t={teleport_at})')
        peak_i = int(np.argmax(surprise))
        peak_t = HISTORY + peak_i
        ax_voe.axvline(peak_t, color='orange', ls=':', lw=1.5,
                       label=f'Peak surprise (t={peak_t})')
        ax_voe.set_xlim(0, n_steps)
        ax_voe.set_ylim(bottom=0)
        ax_voe.set_xlabel('Step')
        ax_voe.set_ylabel('Prediction error (MSE in latent space)')
        ax_voe.set_title('VoE: Surprise signal after random teleport')
        ax_voe.legend(fontsize=8)
        ax_voe.grid(True, alpha=0.3)

    fig.suptitle('LeWM Duckietown Eval — VoE Summary', fontsize=11)
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',      default=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--ckpt',        default=f's3://{S3_BUCKET}/{DEFAULT_CKPT}')
    parser.add_argument('--steps',       type=int, default=N_STEPS_DEF)
    parser.add_argument('--teleport-at', type=int, default=TELEPORT_DEF)
    parser.add_argument('--map',         default='small_loop')
    args = parser.parse_args()

    run_id     = args.run_id
    run_prefix = f'evals/runs/{run_id}'
    local_run  = Path(f'/tmp/eval_{run_id}')
    local_run.mkdir(parents=True, exist_ok=True)

    log(f'Run: {run_id}  steps={args.steps}  teleport_at={args.teleport_at}  map={args.map}')

    # ── Clone le-wm ──────────────────────────────────────────────────────────
    if not LEWM_DIR.exists():
        log('Cloning le-wm ...')
        r = subprocess.run(['git', 'clone', '--depth', '1',
                            'https://github.com/lucas-maes/le-wm.git', str(LEWM_DIR)])
        if r.returncode != 0:
            raise RuntimeError('git clone le-wm failed')

    # ── Download + load model ─────────────────────────────────────────────────
    s3 = make_s3()
    ckpt_local = local_run / 'checkpoint.pt'
    s3_download_ckpt(s3, args.ckpt, ckpt_local)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'Device: {device}')

    model = build_model(device)
    ckpt  = torch.load(ckpt_local, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    log(f'Model loaded ({sum(p.numel() for p in model.parameters()):,} params)')

    # ── Setup duckietown ──────────────────────────────────────────────────────
    setup_duckietown()
    from gym_duckietown.envs import DuckietownEnv
    from PIL import Image as PILImage

    rng  = np.random.default_rng(SEED)
    ctrl = PDController()
    teleport_map = [m for m in MAPS if m != args.map][0]

    env = DuckietownEnv(seed=int(rng.integers(0, 2**31)), map_name=args.map,
                        distortion=False, max_steps=args.steps + 20)
    obs = env.reset()
    ctrl.reset()
    log(f'Duckietown ready. Map: {args.map}  Teleport map: {teleport_map}')

    def resize_obs(frame):
        return np.array(PILImage.fromarray(frame.astype(np.uint8))
                        .resize((IMG_W, IMG_H), PILImage.BILINEAR))

    # ── Phase 1: run episode ──────────────────────────────────────────────────
    raw_frames = []
    raw_actions = []
    t0 = time.time()

    for step in range(args.steps):
        if step == args.teleport_at:
            log(f'Step {step}: TELEPORT — switching to map {teleport_map}')
            env.close()
            env = DuckietownEnv(seed=int(rng.integers(0, 2**31)), map_name=teleport_map,
                                distortion=False, max_steps=args.steps + 20)
            obs = env.reset()
            ctrl.reset()

        obs_r  = resize_obs(obs)
        action = ctrl.act(obs_r, rng)
        raw_frames.append(obs_r)
        raw_actions.append(action)

        obs, _, done, _ = env.step(action)
        if done:
            log(f'  Episode ended at step {step}, resetting')
            obs = env.reset()
            ctrl.reset()

        if step % 50 == 0:
            log(f'  Collected {step}/{args.steps} frames  ({time.time()-t0:.0f}s)')

    env.close()
    log(f'Episode done: {len(raw_frames)} frames in {time.time()-t0:.1f}s')

    # ── Phase 2: encode ───────────────────────────────────────────────────────
    log('Encoding frames ...')
    t0 = time.time()
    embs, act_embs = encode_all(model, raw_frames, raw_actions, device)
    log(f'Encoded {embs.shape[0]} frames in {time.time()-t0:.1f}s  shape: {tuple(embs.shape)}')

    # ── Phase 3: VoE ─────────────────────────────────────────────────────────
    log('Computing VoE surprise signal ...')
    surprise = compute_voe(model, embs, act_embs, device)
    log(f'VoE done: {len(surprise)} surprise values')

    if surprise:
        peak_i = int(np.argmax(surprise))
        peak_t = HISTORY + peak_i
        detected = abs(peak_t - args.teleport_at) <= 5
        log(f'Peak surprise at t={peak_t}  teleport_at={args.teleport_at}  '
            f'detected={detected}')
    else:
        peak_t, detected = -1, False

    # ── Phase 4: GIF ──────────────────────────────────────────────────────────
    log('Generating GIF ...')
    try:
        import imageio.v2 as imageio_v2
        gif_writer = imageio_v2
    except ImportError:
        import imageio as gif_writer  # type: ignore

    gif_path = local_run / 'voe_demo.gif'
    gif_frames = []
    for step in range(args.steps):
        sur_so_far = surprise[:max(0, step - HISTORY + 1)]
        frame_img  = make_gif_frame(raw_frames[step], step, sur_so_far,
                                    args.teleport_at, args.steps)
        gif_frames.append(frame_img)

    gif_writer.mimwrite(str(gif_path), gif_frames, fps=GIF_FPS, loop=0)
    log(f'GIF saved: {gif_path.stat().st_size/1e6:.1f} MB')

    # ── Phase 5: summary figure ───────────────────────────────────────────────
    summary_fig = local_run / 'voe_summary.png'
    make_summary_figure(raw_frames, surprise, args.teleport_at, args.steps, summary_fig)
    log(f'Summary figure saved: {summary_fig}')

    # ── Phase 6: metrics JSON ─────────────────────────────────────────────────
    metrics = {
        'run_id':              run_id,
        'steps':               args.steps,
        'teleport_at':         args.teleport_at,
        'map':                 args.map,
        'teleport_map':        teleport_map,
        'peak_surprise_step':  peak_t,
        'peak_surprise_value': float(max(surprise)) if surprise else 0.0,
        'voe_detected':        detected,
        'surprise_at_teleport': float(surprise[args.teleport_at - HISTORY])
                                 if len(surprise) > args.teleport_at - HISTORY else None,
        'completed_at':        datetime.now(timezone.utc).isoformat(),
    }
    metrics_path = local_run / 'metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2))
    log(f'Metrics: {json.dumps(metrics)}')

    # ── Phase 7: upload to S3 ─────────────────────────────────────────────────
    s3_upload(s3, gif_path,     f'{run_prefix}/voe_demo.gif')
    s3_upload(s3, summary_fig,  f'{run_prefix}/voe_summary.png')
    s3_upload(s3, metrics_path, f'{run_prefix}/metrics.json')

    log(f'All done. Artifacts: s3://{S3_BUCKET}/{run_prefix}/')
    log(f'Presign GIF:')
    log(f'  aws s3 presign s3://{S3_BUCKET}/{run_prefix}/voe_demo.gif --expires-in 3600')
    log(f'Presign summary:')
    log(f'  aws s3 presign s3://{S3_BUCKET}/{run_prefix}/voe_summary.png --expires-in 3600')


if __name__ == '__main__':
    main()

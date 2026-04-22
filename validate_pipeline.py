"""Validates the LeWM duckietown pipeline end-to-end (no jupyter needed)."""

import os, sys, time
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from einops import rearrange
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# add le-wm to path
LEWM_DIR = Path('/home/ubuntu/le-wm')
sys.path.insert(0, str(LEWM_DIR))

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg

DEVICE = torch.device('cpu')
DTYPE  = torch.float32
IS_COLAB = False
SEED = 42

# ── Config ────────────────────────────────────────────────────────────────────
N_TRANSITIONS = 500
DATA_PATH     = Path('data/duckietown_train.h5')
EMBED_DIM     = 64
HISTORY       = 3
N_PREDS       = 1
SEQ_LEN       = HISTORY + N_PREDS
FRAMESKIP     = 1
N_EPOCHS      = 2
BATCH_SIZE    = 32
LR            = 5e-4
SIGREG_W      = 0.09
N_VOE_STEPS   = 40
TELEPORT_AT   = 20

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Mock env & controller ────────────────────────────────────────────────────

class DuckietownMock:
    ROAD_COLOR  = np.array([55, 55, 55],   dtype=np.uint8)
    LANE_WHITE  = np.array([220, 220, 220], dtype=np.uint8)
    LANE_YELLOW = np.array([230, 200, 40],  dtype=np.uint8)
    SKY_COLOR   = np.array([100, 160, 200], dtype=np.uint8)
    GRASS_COLOR = np.array([60, 110, 50],   dtype=np.uint8)
    OBS_H, OBS_W = 120, 160
    MAX_STEPS    = 150

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.x_offset = self.heading = 0.0
        self.step_count = 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.x_offset   = self.rng.uniform(-0.4, 0.4)
        self.heading    = self.rng.uniform(-0.2, 0.2)
        self.step_count = 0
        return self._render(), {}

    def step(self, action):
        vel, steer = float(action[0]), float(action[1])
        self.x_offset  += steer * 0.08 + self.rng.normal(0, 0.01)
        self.heading   += steer * 0.05 + self.rng.normal(0, 0.005)
        self.x_offset   = np.clip(self.x_offset, -0.9, 0.9)
        self.heading    = np.clip(self.heading, -0.5, 0.5)
        self.step_count += 1
        done = abs(self.x_offset) > 0.85 or self.step_count >= self.MAX_STEPS
        return self._render(), 1.0 - abs(self.x_offset), done, done, {}

    def _render(self):
        H, W = self.OBS_H, self.OBS_W
        f = np.empty((H, W, 3), dtype=np.uint8)
        sky_rows = H // 3
        f[:sky_rows] = self.SKY_COLOR
        f[sky_rows:] = self.ROAD_COLOR
        f[sky_rows:, :20]   = self.GRASS_COLOR
        f[sky_rows:, W-20:] = self.GRASS_COLOR
        cx = int(W // 2 + self.x_offset * 35 + self.heading * 15)
        hw = 38
        for col, color in [(cx - hw, self.LANE_WHITE), (cx + hw, self.LANE_WHITE)]:
            if 0 <= col < W:
                f[sky_rows:, max(0, col-3):min(W, col+3)] = color
        for rs in range(sky_rows, H, 18):
            if 0 <= cx < W:
                f[rs:rs+8, max(0, cx-2):min(W, cx+2)] = self.LANE_YELLOW
        return f


class PDController:
    def __init__(self, kp=0.6, kd=0.3, noise_std=0.08):
        self.kp, self.kd, self.noise_std = kp, kd, noise_std
        self.prev_error = 0.0

    def act(self, obs, rng=None):
        rng = rng or np.random.default_rng()
        col_half = obs.shape[1] // 2
        yellow = (obs[:,:,0]>180) & (obs[:,:,1]>160) & (obs[:,:,2]<80)
        error = (np.where(yellow)[1].mean() - col_half) / col_half if yellow.any() else self.prev_error
        steer = -(self.kp * error + self.kd * (error - self.prev_error))
        self.prev_error = error
        return np.array([
            np.clip(0.4 + rng.normal(0, 0.05), 0, 1),
            np.clip(steer + rng.normal(0, self.noise_std), -1, 1),
        ], dtype=np.float32)


# ── Data collection ───────────────────────────────────────────────────────────

def collect_dataset(n_transitions, path, seed=SEED):
    rng = np.random.default_rng(seed)
    env = DuckietownMock(seed=seed)
    ctrl = PDController()
    pixels, actions, ep_idx, step_idx, ep_len = [], [], [], [], []
    ep_id = ep_start = ep_seed = collected = 0
    obs, _ = env.reset(seed=ep_seed); ep_seed += 1
    t0 = time.time()
    while collected < n_transitions:
        action = ctrl.act(obs, rng=rng)
        next_obs, _, terminated, truncated, _ = env.step(action)
        pixels.append(obs.copy())
        actions.append(action.copy())
        ep_idx.append(ep_id)
        step_idx.append(collected - ep_start)
        collected += 1
        if terminated or truncated or (collected - ep_start) >= 200:
            ln = collected - ep_start
            ep_len.extend([ln] * ln)
            ep_id += 1; ep_start = collected
            ctrl.prev_error = 0.0
            obs, _ = env.reset(seed=ep_seed); ep_seed += 1
        else:
            obs = next_obs
    while len(ep_len) < len(pixels):
        ep_len.append(len(pixels) - ep_start)
    arr_px = np.stack(pixels).astype(np.uint8)
    arr_act = np.stack(actions).astype(np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.create_dataset('pixels', data=arr_px, compression='gzip', compression_opts=4)
        f.create_dataset('action', data=arr_act)
        f.create_dataset('episode_idx', data=np.array(ep_idx, dtype=np.int32))
        f.create_dataset('step_idx',    data=np.array(step_idx, dtype=np.int32))
        f.create_dataset('episode_len', data=np.array(ep_len,   dtype=np.int32))
        f.attrs['n_episodes']    = int(ep_id)
        f.attrs['n_transitions'] = int(len(arr_px))
    print(f'  Collected {len(arr_px)} transitions, {ep_id} episodes ({time.time()-t0:.1f}s)')


print('=== Step 1: Data Collection ===')
if not DATA_PATH.exists():
    collect_dataset(N_TRANSITIONS, str(DATA_PATH))
else:
    print(f'  Using existing {DATA_PATH}')

with h5py.File(DATA_PATH, 'r') as f:
    n_tr = f.attrs['n_transitions']
    n_ep = f.attrs['n_episodes']
    sample = f['pixels'][:6]
print(f'  Dataset: {n_tr} transitions, {n_ep} episodes, obs={sample.shape[1:]}')

# save sample frames
fig, axes = plt.subplots(1, 6, figsize=(14, 2.5))
for ax, frame in zip(axes, sample):
    ax.imshow(frame); ax.axis('off')
fig.suptitle('Sample duckietown frames', fontsize=10)
plt.tight_layout()
plt.savefig('data/sample_frames.png', dpi=100)
plt.close()
print('  Saved data/sample_frames.png')


# ── Dataset ───────────────────────────────────────────────────────────────────

class DuckietownH5Dataset(Dataset):
    def __init__(self, path, num_steps=4, frameskip=1):
        self.path      = path
        self.num_steps = num_steps
        self.frameskip = frameskip
        with h5py.File(path, 'r') as f:
            self.ep_idx   = f['episode_idx'][:]
            self.actions  = f['action'][:]
            self.n        = len(self.ep_idx)
        episodes = np.unique(self.ep_idx)
        window   = num_steps * frameskip
        self.valid = []
        for ep in episodes:
            ep_inds = np.where(self.ep_idx == ep)[0]
            for s in ep_inds[:max(1, len(ep_inds) - window + 1)]:
                if s + window <= ep_inds[-1] + 1:
                    self.valid.append(s)
        self.valid = np.array(self.valid)

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        start = self.valid[idx]
        inds  = np.arange(start, start + self.num_steps * self.frameskip, self.frameskip)
        inds  = np.clip(inds, 0, self.n - 1)
        with h5py.File(self.path, 'r') as f:
            frames = f['pixels'][inds]
        pixels  = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        actions = torch.from_numpy(self.actions[inds])
        return {'pixels': pixels, 'action': actions}


full_ds  = DuckietownH5Dataset(DATA_PATH, num_steps=SEQ_LEN, frameskip=FRAMESKIP)
n_train  = int(0.9 * len(full_ds))
n_val    = len(full_ds) - n_train
train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                 generator=torch.Generator().manual_seed(SEED))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

batch = next(iter(train_loader))
print(f'\n=== Step 2: Dataset ===')
print(f'  Train: {len(train_ds)} | Val: {len(val_ds)}')
print(f'  pixels: {tuple(batch["pixels"].shape)}  action: {tuple(batch["action"].shape)}')


# ── CNN Encoder ───────────────────────────────────────────────────────────────

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, embed_dim, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
    def forward(self, pixel_values, **kw):
        b = pixel_values.size(0)
        feat = self.net(pixel_values).view(b, -1)
        return type('Out', (), {'last_hidden_state': feat.unsqueeze(1)})()


# ── Build JEPA model ──────────────────────────────────────────────────────────

encoder        = CNNEncoder(embed_dim=EMBED_DIM)
projector      = MLP(EMBED_DIM, EMBED_DIM, EMBED_DIM)
pred_proj      = MLP(EMBED_DIM, EMBED_DIM, EMBED_DIM)
action_encoder = Embedder(input_dim=2, smoothed_dim=2, emb_dim=EMBED_DIM, mlp_scale=4)
predictor      = ARPredictor(
    num_frames=HISTORY, input_dim=EMBED_DIM, hidden_dim=EMBED_DIM,
    output_dim=EMBED_DIM, depth=2, heads=4, dim_head=16,
    mlp_dim=EMBED_DIM * 2, dropout=0.1,
)
sigreg = SIGReg(knots=17, num_proj=256)
model  = JEPA(encoder, predictor, action_encoder, projector, pred_proj).to(DEVICE)
sigreg = sigreg.to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f'\n=== Step 3: Model ===')
print(f'  Parameters: {n_params:,}')

# sanity check
with torch.no_grad():
    b2 = {k: v[:2].to(DEVICE) for k, v in batch.items()}
    out = model.encode(b2)
    ctx = model.predict(out['emb'][:, :HISTORY], out['act_emb'][:, :HISTORY])
    print(f'  Forward pass OK: emb {tuple(out["emb"].shape)}, pred {tuple(ctx.shape)}')


# ── Training ──────────────────────────────────────────────────────────────────

def step_fn(batch):
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)
    out      = model.encode(batch)
    emb      = out['emb']
    act_emb  = out['act_emb']
    ctx_emb  = emb[:, :HISTORY]
    ctx_act  = act_emb[:, :HISTORY]
    tgt_emb  = emb[:, N_PREDS:]
    pred_emb = model.predict(ctx_emb, ctx_act)
    pred_loss = (pred_emb - tgt_emb).pow(2).mean()
    sig_loss  = sigreg(emb.transpose(0, 1))
    return pred_loss + SIGREG_W * sig_loss, pred_loss.item(), sig_loss.item()


optimizer    = torch.optim.AdamW(
    list(model.parameters()) + list(sigreg.parameters()), lr=LR, weight_decay=1e-3)
train_losses = []
val_losses   = []
best_val     = float('inf')

print(f'\n=== Step 4: Training ({N_EPOCHS} epochs) ===')
for epoch in range(1, N_EPOCHS + 1):
    model.train()
    ep_train = []
    for b in train_loader:
        optimizer.zero_grad()
        loss, pl, sl = step_fn(b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ep_train.append(loss.item())

    model.eval()
    ep_val = []
    with torch.no_grad():
        for b in val_loader:
            loss, _, _ = step_fn(b)
            ep_val.append(loss.item())

    tl, vl = float(np.mean(ep_train)), float(np.mean(ep_val))
    train_losses.append(tl); val_losses.append(vl)
    if vl < best_val:
        best_val = vl
        torch.save(model.state_dict(), 'data/lewm_best.pt')
    print(f'  Epoch {epoch}/{N_EPOCHS}  train={tl:.4f}  val={vl:.4f}')

# loss curve
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, N_EPOCHS+1), train_losses, marker='o', label='train')
ax.plot(range(1, N_EPOCHS+1), val_losses,   marker='s', label='val')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('LeWM Loss Curve'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('data/loss_curve.png', dpi=100); plt.close()
print(f'  Loss curve saved. Best val: {best_val:.4f}')


# ── Latent space (PCA) ────────────────────────────────────────────────────────

print('\n=== Step 5: Latent Space Visualisation ===')
model.eval()
with h5py.File(DATA_PATH, 'r') as f:
    n_viz  = min(200, len(f['pixels']))
    frames = f['pixels'][:n_viz]

px = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
embs = []
with torch.no_grad():
    for i in range(0, n_viz, 32):
        out = encoder(px[i:i+32].to(DEVICE))
        embs.append(out.last_hidden_state[:, 0].cpu())
embs = torch.cat(embs)
embs_norm = (embs - embs.mean(0)) / (embs.std(0) + 1e-8)

U, S, Vt = torch.pca_lowrank(embs_norm, q=2)
coords = (embs_norm @ Vt).numpy()

fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(coords[:,0], coords[:,1], c=np.arange(len(coords))/len(coords),
                cmap='viridis', alpha=0.6, s=12)
plt.colorbar(sc, ax=ax, label='Timestep')
ax.set_title('Latent Space (PCA)'); ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
plt.tight_layout(); plt.savefig('data/latent_space.png', dpi=100); plt.close()
print(f'  PCA plot saved (embs shape: {embs.shape})')


# ── VoE ───────────────────────────────────────────────────────────────────────

print('\n=== Step 6: VoE Evaluation ===')
rng_voe = np.random.default_rng(SEED + 1)

with h5py.File(DATA_PATH, 'r') as f:
    n_total = len(f['pixels'])
    voe_frames  = f['pixels'][:N_VOE_STEPS]
    voe_actions = f['action'][:N_VOE_STEPS]
    tp_idx      = rng_voe.integers(n_total // 2, n_total)
    tp_frame    = f['pixels'][tp_idx]

voe_inj = voe_frames.copy()
voe_inj[TELEPORT_AT] = tp_frame

def encode_seq(frames):
    px = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
    with torch.no_grad():
        out = encoder(px.to(DEVICE))
    return out.last_hidden_state[:, 0]

actual_embs = encode_seq(voe_inj)
action_seq  = torch.from_numpy(voe_actions).float().to(DEVICE)
surprise    = []
model.eval()
with torch.no_grad():
    act_embs = model.action_encoder(action_seq.unsqueeze(0))
    ctx_emb  = actual_embs[:HISTORY].unsqueeze(0)
    for t in range(HISTORY, N_VOE_STEPS):
        ctx_act = act_embs[:, t-HISTORY:t]
        pred    = model.predict(ctx_emb, ctx_act)[:, -1]
        actual  = actual_embs[t].unsqueeze(0)
        surprise.append((pred - actual).pow(2).mean().item())
        ctx_emb = torch.cat([ctx_emb[:, 1:], actual.unsqueeze(1)], dim=1)

steps = np.arange(HISTORY, N_VOE_STEPS)
peak  = int(steps[np.argmax(surprise)])

fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [2,1]})
axes[0].plot(steps, surprise, color='royalblue')
axes[0].axvline(TELEPORT_AT, color='red', ls='--', label=f'Teleport (t={TELEPORT_AT})')
axes[0].axvline(peak, color='orange', ls=':', label=f'Peak (t={peak})')
axes[0].set_xlabel('Step'); axes[0].set_ylabel('Prediction error')
axes[0].set_title('VoE Surprise Signal'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].imshow(tp_frame); axes[1].axis('off'); axes[1].set_title('Injected teleport frame')
plt.tight_layout(); plt.savefig('data/voe_surprise.png', dpi=100); plt.close()

detected = TELEPORT_AT - 2 <= peak <= TELEPORT_AT + 4
print(f'  Teleport at t={TELEPORT_AT}, peak surprise at t={peak}  ->  {"DETECTED" if detected else "not detected yet"}')

print('\n' + '='*55)
print('  VALIDATION COMPLETE')
print('='*55)
print(f'  Transitions : {n_tr}  |  Episodes: {n_ep}')
print(f'  Best val loss: {best_val:.4f}')
print(f'  Outputs: data/sample_frames.png, loss_curve.png, latent_space.png, voe_surprise.png')

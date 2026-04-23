"""
Collect duckietown rollouts and save as HDF5 dataset.
Uses real gym-duckietown if available, otherwise falls back to a synthetic mock.

Usage:
    python collect_duckietown.py --n_transitions 500 --output data/duckietown_train.h5
"""

import argparse
import os
import sys
import time
import numpy as np
import h5py
from pathlib import Path


# ── Synthetic mock environment ──────────────────────────────────────────────

class DuckietownMock:
    """Lightweight procedural duckietown-like env for pipeline testing.
    Produces (120, 160, 3) uint8 RGB frames with lane markings.
    Action: [forward_velocity, steering_angle] in [-1, 1].
    """

    ROAD_COLOR = np.array([55, 55, 55], dtype=np.uint8)
    LANE_WHITE  = np.array([220, 220, 220], dtype=np.uint8)
    LANE_YELLOW = np.array([230, 200, 40], dtype=np.uint8)
    SKY_COLOR   = np.array([100, 160, 200], dtype=np.uint8)
    GRASS_COLOR = np.array([60, 110, 50], dtype=np.uint8)

    OBS_H, OBS_W = 120, 160
    MAX_STEPS     = 200

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.x_offset   = 0.0   # lateral offset from lane centre [-1, 1]
        self.heading     = 0.0   # heading error in radians
        self.step_count  = 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.x_offset  = self.rng.uniform(-0.4, 0.4)
        self.heading   = self.rng.uniform(-0.2, 0.2)
        self.step_count = 0
        return self._render(), {}

    def step(self, action):
        vel, steer = float(action[0]), float(action[1])
        self.x_offset  += steer * 0.08 + self.rng.normal(0, 0.01)
        self.heading   += steer * 0.05 + self.rng.normal(0, 0.005)
        self.x_offset   = np.clip(self.x_offset, -0.9, 0.9)
        self.heading    = np.clip(self.heading, -0.5, 0.5)
        self.step_count += 1

        reward = 1.0 - abs(self.x_offset) - 0.5 * abs(self.heading)
        terminated = abs(self.x_offset) > 0.85
        truncated  = self.step_count >= self.MAX_STEPS
        return self._render(), reward, terminated, truncated, {}

    def _render(self):
        H, W = self.OBS_H, self.OBS_W
        frame = np.empty((H, W, 3), dtype=np.uint8)

        sky_rows = H // 3
        frame[:sky_rows]  = self.SKY_COLOR
        frame[sky_rows:]  = self.ROAD_COLOR

        # grass edges
        frame[sky_rows:, :20]   = self.GRASS_COLOR
        frame[sky_rows:, W-20:] = self.GRASS_COLOR

        # compute lane centre pixel col
        cx = int(W // 2 + self.x_offset * 35 + self.heading * 15)
        half_lane = 38

        left_line  = cx - half_lane
        right_line = cx + half_lane

        for col, color in [(left_line, self.LANE_WHITE), (right_line, self.LANE_WHITE)]:
            if 0 <= col < W:
                frame[sky_rows:, max(0, col-3):min(W, col+3)] = color

        # dashed centre line
        for row_start in range(sky_rows, H, 18):
            if 0 <= cx < W:
                frame[row_start:row_start+8, max(0, cx-2):min(W, cx+2)] = self.LANE_YELLOW

        return frame


# ── PD lane-follower ─────────────────────────────────────────────────────────

class PDController:
    """Simple PD controller for lane following."""
    def __init__(self, kp=0.6, kd=0.3, noise_std=0.05):
        self.kp = kp
        self.kd = kd
        self.noise_std = noise_std
        self.prev_error = 0.0

    def act(self, obs, info=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        if hasattr(obs, 'x_offset'):
            error = obs.x_offset
        else:
            # estimate lateral error from image: deviation of yellow line from centre
            col_half = obs.shape[1] // 2
            yellow_mask = (obs[:, :, 0] > 180) & (obs[:, :, 1] > 160) & (obs[:, :, 2] < 80)
            if yellow_mask.any():
                ys, xs = np.where(yellow_mask)
                error = (xs.mean() - col_half) / col_half
            else:
                error = self.prev_error

        d_error = error - self.prev_error
        self.prev_error = error

        steer = -(self.kp * error + self.kd * d_error)
        steer += rng.normal(0, self.noise_std)
        vel = 0.4 + rng.normal(0, 0.05)
        return np.array([np.clip(vel, 0, 1), np.clip(steer, -1, 1)], dtype=np.float32)


# ── Real gym-duckietown (optional) ───────────────────────────────────────────

def try_import_duckietown():
    try:
        import gym
        from gym_duckietown.envs import DuckietownEnv
        return DuckietownEnv
    except ImportError:
        return None


# ── Collection loop ───────────────────────────────────────────────────────────

def collect(n_transitions: int, output_path: str, seed: int = 42, use_mock: bool = False):
    rng = np.random.default_rng(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    DuckietownEnv = None if use_mock else try_import_duckietown()

    if DuckietownEnv is not None:
        print("Using real gym-duckietown")
        import os
        os.environ.setdefault("DISPLAY", ":1")
        env = DuckietownEnv(
            seed=seed,
            user_tile_start=[0, 0],
            distortion=False,
            randomize_maps_on_reset=True,
        )
    else:
        print("Using synthetic DuckietownMock (gym-duckietown not available)")
        env = DuckietownMock(seed=seed)

    controller = PDController(noise_std=0.08)

    all_pixels   = []
    all_actions  = []
    all_ep_idx   = []
    all_step_idx = []
    all_ep_len   = []

    episode_idx   = 0
    collected     = 0
    ep_start      = 0
    ep_seed       = seed

    obs, _ = env.reset(seed=ep_seed)
    ep_seed += 1

    print(f"Collecting {n_transitions} transitions...")
    t0 = time.time()

    while collected < n_transitions:
        action = controller.act(obs, rng=rng)
        next_obs, reward, terminated, truncated, info = env.step(action)

        all_pixels.append(obs.copy())
        all_actions.append(action.copy())
        all_ep_idx.append(episode_idx)
        all_step_idx.append(collected - ep_start)

        collected += 1

        if terminated or truncated or (collected - ep_start) >= 200:
            ep_len = collected - ep_start
            for i in range(ep_start, collected):
                all_ep_len.append(ep_len)
            episode_idx += 1
            ep_start = collected
            controller.prev_error = 0.0
            obs, _ = env.reset(seed=ep_seed)
            ep_seed += 1
        else:
            obs = next_obs

        if collected % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {collected}/{n_transitions} transitions  ({elapsed:.1f}s)")

    # pad episode_len for any incomplete episode at the end
    while len(all_ep_len) < len(all_pixels):
        all_ep_len.append(len(all_pixels) - ep_start)

    # save
    pixels  = np.stack(all_pixels,  axis=0).astype(np.uint8)
    actions = np.stack(all_actions, axis=0).astype(np.float32)
    ep_idx  = np.array(all_ep_idx,  dtype=np.int32)
    step_idx = np.array(all_step_idx, dtype=np.int32)
    ep_len  = np.array(all_ep_len,  dtype=np.int32)

    print(f"\nSaving {len(pixels)} transitions across {episode_idx} episodes...")
    print(f"  pixels shape : {pixels.shape}")
    print(f"  action shape : {actions.shape}")

    with h5py.File(output_path, "w") as f:
        f.create_dataset("pixels",      data=pixels,    compression="gzip", compression_opts=4)
        f.create_dataset("action",      data=actions)
        f.create_dataset("episode_idx", data=ep_idx)
        f.create_dataset("step_idx",    data=step_idx)
        f.create_dataset("episode_len", data=ep_len)
        f.attrs["n_episodes"]   = int(episode_idx)
        f.attrs["n_transitions"] = int(len(pixels))
        f.attrs["obs_shape"]    = list(pixels.shape[1:])
        f.attrs["action_dim"]   = int(actions.shape[1])

    print(f"Saved to {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_transitions", type=int, default=500)
    parser.add_argument("--output", type=str, default="data/duckietown_train.h5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock", action="store_true", help="Force synthetic mock env")
    args = parser.parse_args()

    collect(args.n_transitions, args.output, seed=args.seed, use_mock=args.mock)

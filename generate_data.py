"""
Generate duckietown dataset from the real gym-duckietown simulator.
Saves an HDF5 file and optionally uploads to S3.

Requirements (Python 3.10):
    pip install duckietown-gym-daffy pyglet==1.5.27 h5py boto3
    sudo apt-get install libgl1 libglu1-mesa libglib2.0-0 xvfb
    # symlink maps:
    ln -sf $(python3 -c "import duckietown_world,os;print(os.path.dirname(duckietown_world.__file__))")/data/gd1/maps \
        $(python3 -c "import gym_duckietown,os;print(os.path.dirname(gym_duckietown.__file__))")/maps

Usage:
    # Start virtual display first (headless server):
    Xvfb :99 -screen 0 1024x768x24 &
    DISPLAY=:99 python3 generate_data.py --n-transitions 100000 --upload
"""

import argparse, os, time, warnings
warnings.filterwarnings('ignore')
import logging; logging.disable(logging.INFO)

import numpy as np
import h5py
from pathlib import Path

S3_BUCKET   = 'test-854656252703'
S3_DATA_KEY = 'lewm-duckietown/duckietown_100k.h5'

MAPS = [
    'small_loop', 'small_loop_cw', 'loop_empty',
    'straight_road', 'zigzag_dists', 'udem1',
]

IMG_H, IMG_W = 120, 160   # resize from 480x640 to match training config


class PDController:
    """Simple PD lane-follower on the yellow centre line."""
    def __init__(self, kp=0.45, kd=0.25, noise_std=0.06, speed=0.35):
        self.kp = kp
        self.kd = kd
        self.noise_std = noise_std
        self.speed = speed
        self.prev_error = 0.0

    def act(self, obs, rng):
        # obs: (H, W, 3) uint8
        yellow = (obs[:, :, 0] > 170) & (obs[:, :, 1] > 150) & (obs[:, :, 2] < 100)
        cx = obs.shape[1] // 2
        if yellow.any():
            _, xs = np.where(yellow)
            error = (xs.mean() - cx) / cx
        else:
            error = self.prev_error
        steer = -(self.kp * error + self.kd * (error - self.prev_error))
        self.prev_error = error
        vel   = float(np.clip(self.speed + rng.normal(0, 0.03), 0.1, 0.6))
        steer = float(np.clip(steer + rng.normal(0, self.noise_std), -1.0, 1.0))
        return np.array([vel, steer], dtype=np.float64)

    def reset(self):
        self.prev_error = 0.0


def resize(frame):
    """Fast nearest-neighbour resize (H,W,3) → (IMG_H,IMG_W,3)."""
    import cv2
    return cv2.resize(frame, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)


def collect(n_transitions, seed=42, max_ep_steps=400):
    from gym_duckietown.envs import DuckietownEnv

    rng  = np.random.default_rng(seed)
    ctrl = PDController()

    pixels, actions, ep_idx_list, step_idx_list = [], [], [], []
    ep_id = 0
    ep_start = 0
    collected = 0

    map_cycle = list(MAPS) * (n_transitions // (max_ep_steps * len(MAPS)) + 2)

    print(f'Collecting {n_transitions:,} transitions from {len(MAPS)} maps ...')
    t0 = time.time()

    while collected < n_transitions:
        map_name = map_cycle[ep_id % len(MAPS)]
        ep_seed  = int(rng.integers(0, 2**31))
        env = DuckietownEnv(seed=ep_seed, map_name=map_name,
                            distortion=False, max_steps=max_ep_steps)
        obs = env.reset()
        ctrl.reset()
        ep_step = 0

        while collected < n_transitions and ep_step < max_ep_steps:
            frame = resize(obs)
            action = ctrl.act(obs, rng)

            pixels.append(frame)
            actions.append(action.astype(np.float32))
            ep_idx_list.append(ep_id)
            step_idx_list.append(ep_step)

            obs, _, done, _ = env.step(action)
            collected += 1
            ep_step   += 1

            if done:
                break

        env.close()
        ep_id    += 1
        ep_start  = collected

        elapsed = time.time() - t0
        rate    = collected / max(elapsed, 1)
        eta     = (n_transitions - collected) / max(rate, 1)
        print(f'  {collected:6d}/{n_transitions}  maps={map_name:<20s}  {rate:.0f} tr/s  ETA {eta:.0f}s',
              end='\r', flush=True)

    print(f'\nDone: {collected:,} transitions, {ep_id} episodes  ({time.time()-t0:.1f}s)')
    return (np.stack(pixels),
            np.stack(actions),
            np.array(ep_idx_list, dtype=np.int32),
            np.array(step_idx_list, dtype=np.int32),
            ep_id)


def save_hdf5(out_path, pixels, actions, ep_idx, step_idx, n_episodes):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('pixels',      data=pixels,   compression='gzip', compression_opts=4)
        f.create_dataset('action',      data=actions)
        f.create_dataset('episode_idx', data=ep_idx)
        f.create_dataset('step_idx',    data=step_idx)
        f.attrs['n_episodes']    = n_episodes
        f.attrs['n_transitions'] = len(pixels)
        f.attrs['img_h']         = pixels.shape[1]
        f.attrs['img_w']         = pixels.shape[2]
        f.attrs['maps']          = ','.join(MAPS)
    print(f'Saved {len(pixels):,} transitions → {out_path}  ({Path(out_path).stat().st_size/1e6:.1f} MB)')


def upload_s3(local_path):
    import boto3
    s3 = boto3.client('s3', region_name='us-east-1')
    print(f'Uploading to s3://{S3_BUCKET}/{S3_DATA_KEY} ...')
    s3.upload_file(str(local_path), S3_BUCKET, S3_DATA_KEY)
    print('Upload complete.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-transitions', type=int, default=100_000)
    parser.add_argument('--out',           default='data/duckietown_100k.h5')
    parser.add_argument('--seed',          type=int, default=42)
    parser.add_argument('--upload',        action='store_true')
    args = parser.parse_args()

    pixels, actions, ep_idx, step_idx, n_ep = collect(args.n_transitions, seed=args.seed)
    save_hdf5(args.out, pixels, actions, ep_idx, step_idx, n_ep)
    if args.upload:
        upload_s3(args.out)


if __name__ == '__main__':
    main()

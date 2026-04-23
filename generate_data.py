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

S3_BUCKET   = 'leworldduckie'
S3_DATA_KEY = 'data/duckietown_100k.h5'

MAPS = [
    'small_loop', 'small_loop_cw', 'loop_empty',
    'straight_road', 'zigzag_dists', 'udem1',
]

IMG_H, IMG_W = 120, 160
WRITE_CHUNK  = 1000  # flush to disk every N transitions


class PDController:
    """Simple PD lane-follower on the yellow centre line."""
    def __init__(self, kp=0.45, kd=0.25, noise_std=0.06, speed=0.35):
        self.kp = kp
        self.kd = kd
        self.noise_std = noise_std
        self.speed = speed
        self.prev_error = 0.0

    def act(self, obs, rng):
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
    import cv2
    return cv2.resize(frame, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)


def collect_to_hdf5(out_path, n_transitions, seed=42, max_ep_steps=400):
    from gym_duckietown.envs import DuckietownEnv

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Pre-allocate the full HDF5 file on disk — no data held in RAM beyond one chunk.
    with h5py.File(out_path, 'w') as f:
        px_ds  = f.create_dataset('pixels',      shape=(n_transitions, IMG_H, IMG_W, 3),
                                  dtype='uint8',   chunks=(WRITE_CHUNK, IMG_H, IMG_W, 3),
                                  compression='gzip', compression_opts=4)
        act_ds = f.create_dataset('action',      shape=(n_transitions, 2),
                                  dtype='float32', chunks=(WRITE_CHUNK, 2))
        ep_ds  = f.create_dataset('episode_idx', shape=(n_transitions,),
                                  dtype='int32',   chunks=(WRITE_CHUNK,))
        st_ds  = f.create_dataset('step_idx',    shape=(n_transitions,),
                                  dtype='int32',   chunks=(WRITE_CHUNK,))

        rng  = np.random.default_rng(seed)
        ctrl = PDController()

        buf_px  = np.empty((WRITE_CHUNK, IMG_H, IMG_W, 3), dtype='uint8')
        buf_act = np.empty((WRITE_CHUNK, 2),                dtype='float32')
        buf_ep  = np.empty((WRITE_CHUNK,),                  dtype='int32')
        buf_st  = np.empty((WRITE_CHUNK,),                  dtype='int32')

        ep_id     = 0
        collected = 0
        buf_pos   = 0

        # Batch episodes by map: run all episodes for one map before switching.
        # Cycling maps every episode recreates the GL context every ~400 steps
        # which leaks memory and OOMs well before 100k transitions.
        eps_per_map = max(1, n_transitions // max_ep_steps // len(MAPS) + 10)

        print(f'Collecting {n_transitions:,} transitions from {len(MAPS)} maps ...')
        t0 = time.time()

        def flush(start, count):
            px_ds[start:start+count]  = buf_px[:count]
            act_ds[start:start+count] = buf_act[:count]
            ep_ds[start:start+count]  = buf_ep[:count]
            st_ds[start:start+count]  = buf_st[:count]
            f.flush()

        env          = None
        current_map  = None

        while collected < n_transitions:
            map_name = MAPS[(ep_id // eps_per_map) % len(MAPS)]
            ep_seed  = int(rng.integers(0, 2**31))

            # Only recreate the env when the map changes — avoids per-episode
            # GL context churn which leaks memory and eventually OOMs.
            if map_name != current_map:
                if env is not None:
                    env.close()
                env = DuckietownEnv(seed=ep_seed, map_name=map_name,
                                    distortion=False, max_steps=max_ep_steps)
                current_map = map_name

            env.seed(ep_seed)
            obs = env.reset()
            ctrl.reset()
            ep_step = 0

            while collected < n_transitions and ep_step < max_ep_steps:
                buf_px[buf_pos]  = resize(obs)
                buf_act[buf_pos] = ctrl.act(obs, rng).astype(np.float32)
                buf_ep[buf_pos]  = ep_id
                buf_st[buf_pos]  = ep_step

                obs, _, done, _ = env.step(buf_act[buf_pos])
                collected += 1
                ep_step   += 1
                buf_pos   += 1

                if buf_pos == WRITE_CHUNK:
                    flush(collected - WRITE_CHUNK, WRITE_CHUNK)
                    buf_pos = 0

                if done:
                    break

            ep_id += 1

            elapsed = time.time() - t0
            rate    = collected / max(elapsed, 1)
            eta     = (n_transitions - collected) / max(rate, 1)
            print(f'  {collected:6d}/{n_transitions}  maps={map_name:<20s}  {rate:.0f} tr/s  ETA {eta:.0f}s',
                  end='\r', flush=True)

        if env is not None:
            env.close()

        # flush any remaining partial chunk
        if buf_pos > 0:
            flush(collected - buf_pos, buf_pos)

        f.attrs['n_episodes']    = ep_id
        f.attrs['n_transitions'] = n_transitions
        f.attrs['img_h']         = IMG_H
        f.attrs['img_w']         = IMG_W
        f.attrs['maps']          = ','.join(MAPS)

    size_mb = Path(out_path).stat().st_size / 1e6
    print(f'\nDone: {collected:,} transitions, {ep_id} episodes  ({time.time()-t0:.1f}s)')
    print(f'Saved → {out_path}  ({size_mb:.1f} MB)')


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

    collect_to_hdf5(args.out, args.n_transitions, seed=args.seed)
    if args.upload:
        upload_s3(args.out)


if __name__ == '__main__':
    main()

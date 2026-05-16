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
S3_EXPLORE_KEY = 'data/duckie_explore.h5'

MAPS = [
    'small_loop', 'small_loop_cw', 'loop_empty',
    'straight_road', 'zigzag_dists', 'udem1',
]

IMG_H, IMG_W = 120, 160
WRITE_CHUNK  = 1000  # flush to disk every N transitions


class LaneFollowController:
    """Ground-truth lane follower using env.get_lane_pos2().

    Uses the simulator's exact lane offset (dist) and heading error (angle_rad)
    rather than pixel-level heuristics, producing clean, consistent trajectories.
    Small Gaussian noise on both velocity and steering ensures data diversity.
    """
    def __init__(self, k_lat=10.0, k_heading=5.0, speed=0.35,
                 speed_noise=0.03, steer_noise=0.03):
        self.k_lat       = k_lat
        self.k_heading   = k_heading
        self.speed       = speed
        self.speed_noise = speed_noise
        self.steer_noise = steer_noise

    def act(self, env, rng):
        lp    = env.get_lane_pos2(env.cur_pos, env.cur_angle)
        steer = self.k_lat * lp.dist + self.k_heading * lp.angle_rad
        vel   = float(np.clip(self.speed + rng.normal(0, self.speed_noise), 0.1, 0.6))
        steer = float(np.clip(steer + rng.normal(0, self.steer_noise), -1.0, 1.0))
        return np.array([vel, steer], dtype=np.float64)

    def reset(self):
        pass


def sample_random_action(rng, action_low, action_high):
    return rng.uniform(action_low, action_high).astype(np.float64)


def resize(frame):
    import cv2
    return cv2.resize(frame, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)


def collect_to_hdf5(
    out_path,
    n_transitions,
    seed=42,
    max_ep_steps=400,
    explore=False,
    explore_vel_std=0.15,
    explore_steer_std=0.30,
    random_action_prob=0.10,
    offlane_dist_thresh=0.10,
):
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
        ctrl = LaneFollowController()

        buf_px  = np.empty((WRITE_CHUNK, IMG_H, IMG_W, 3), dtype='uint8')
        buf_act = np.empty((WRITE_CHUNK, 2),                dtype='float32')
        buf_ep  = np.empty((WRITE_CHUNK,),                  dtype='int32')
        buf_st  = np.empty((WRITE_CHUNK,),                  dtype='int32')

        ep_id     = 0
        collected = 0
        buf_pos   = 0
        total_ep_steps = 0
        offlane_count = 0
        action_low = np.array([0.1, -1.0], dtype=np.float64)
        action_high = np.array([0.6, 1.0], dtype=np.float64)

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
                lane_pos = env.get_lane_pos2(env.cur_pos, env.cur_angle)
                base_action = ctrl.act(env, rng)
                if explore:
                    action = base_action + np.array([
                        rng.normal(0.0, explore_vel_std),
                        rng.normal(0.0, explore_steer_std),
                    ], dtype=np.float64)
                    if rng.random() < random_action_prob:
                        action = sample_random_action(rng, action_low, action_high)
                    action = np.clip(action, action_low, action_high)
                else:
                    action = base_action
                buf_act[buf_pos] = action.astype(np.float32)
                buf_ep[buf_pos]  = ep_id
                buf_st[buf_pos]  = ep_step
                if abs(float(lane_pos.dist)) > offlane_dist_thresh:
                    offlane_count += 1

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
            total_ep_steps += ep_step

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
        f.attrs['explore']       = int(explore)
        f.attrs['explore_vel_std'] = float(explore_vel_std)
        f.attrs['explore_steer_std'] = float(explore_steer_std)
        f.attrs['random_action_prob'] = float(random_action_prob)
        f.attrs['offlane_dist_thresh'] = float(offlane_dist_thresh)
        f.attrs['offlane_fraction_estimate'] = float(offlane_count / max(collected, 1))
        f.attrs['mean_episode_length'] = float(total_ep_steps / max(ep_id, 1))

    size_mb = Path(out_path).stat().st_size / 1e6
    print(f'\nDone: {collected:,} transitions, {ep_id} episodes  ({time.time()-t0:.1f}s)')
    print(f'Saved → {out_path}  ({size_mb:.1f} MB)')
    print(f'Mean episode length: {total_ep_steps / max(ep_id, 1):.1f}')
    print(f'Off-lane fraction estimate (|dist|>{offlane_dist_thresh:.2f}): {offlane_count / max(collected, 1):.3f}')


def upload_s3(local_path, s3_key):
    import boto3
    s3 = boto3.client('s3', region_name='us-east-1')
    print(f'Uploading to s3://{S3_BUCKET}/{s3_key} ...')
    s3.upload_file(str(local_path), S3_BUCKET, s3_key)
    print('Upload complete.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-transitions', type=int, default=100_000)
    parser.add_argument('--out',           default='data/duckietown_100k.h5')
    parser.add_argument('--seed',          type=int, default=42)
    parser.add_argument('--upload',        action='store_true')
    parser.add_argument('--s3-key',        default=None,
                        help='S3 key under bucket leworldduckie. Defaults to duckie_explore.h5 when --explore else duckietown_100k.h5')
    parser.add_argument('--explore',       action='store_true',
                        help='Enable exploratory collection mode (large action noise + occasional random actions).')
    parser.add_argument('--explore-vel-std', type=float, default=0.15)
    parser.add_argument('--explore-steer-std', type=float, default=0.30)
    parser.add_argument('--random-action-prob', type=float, default=0.10)
    parser.add_argument('--offlane-dist-thresh', type=float, default=0.10)
    args = parser.parse_args()

    collect_to_hdf5(
        args.out,
        args.n_transitions,
        seed=args.seed,
        explore=args.explore,
        explore_vel_std=args.explore_vel_std,
        explore_steer_std=args.explore_steer_std,
        random_action_prob=args.random_action_prob,
        offlane_dist_thresh=args.offlane_dist_thresh,
    )
    if args.upload:
        s3_key = args.s3_key
        if s3_key is None:
            s3_key = S3_EXPLORE_KEY if args.explore else S3_DATA_KEY
        upload_s3(args.out, s3_key)


if __name__ == '__main__':
    main()

# MPC Lane-Following Controller

The MPC controller uses the trained LeWM as a latent dynamics model to do goal-reaching via CEM (Cross-Entropy Method) planning.

## Concept

Instead of planning in pixel space, we:
1. Encode the current observation → latent `z_t`
2. Encode a goal image → `z_goal` (a frame from a "good" lane-following trajectory)
3. Plan action sequences that minimise MSE(`predicted_z`, `z_goal`) in latent space

This avoids pixel-level reconstruction and leverages the learned dynamics directly.

## CEM Planner Design

Custom ~80-line CEM (not stable_worldmodel's CEMSolver — JEPA's `criterion` has a shape bug with `(B, S, ...)` expanded info_dicts).

```
N = 200 samples per iteration
horizon = 10 steps
iterations = 3 CEM iterations
elite fraction = 10% (top 20 of 200)
warm-start: shift plan by 1 each real step
```

At each CEM step `k`:
```python
full_act = cat([act_win (N, H-1, D), fut_act_embs[:, k:k+1] (N, 1, D)])  # → (N, H, D)
next_emb = model.predict(emb_win, full_act)[:, -1]
```

Both context windows (embeddings and actions) slide forward after each real step.

**Cost function:**
```
cost = sum_over_horizon( MSE(pred_emb, z_goal) ) + 0.01 * action_smoothness_penalty
```

## Action Indexing (verified from training code)

Training `step_fn`: `ctx_emb = emb[:, :3]`, `ctx_act = act_emb[:, :3]`, `tgt_emb = emb[:, 1:]`

So `predict([z0, z1, z2], [a0, a1, a2])[-1] ≈ z3`.

`ctx_act[-1] = a2` = action at the last context position, predicts 1 subsampled step ahead.

For MPC: `ctx_act[-1] = a_t^(sampled)` → predicts `z_{t+1}`. Correct with FRAMESKIP=1.

## Episode Setup

```
lag_frames = 4   # burn LaneFollower steps at episode start (PWM warm-up, matches training)
frameskip  = 1   # raw env steps per planning step (must match FRAMESKIP in training)
steps      = 300 # max steps per episode
episodes   = 10
```

The lag frames replicate the `skip_initial_steps=4` in the training data collection, ensuring the controller starts with the same observation distribution the model was trained on.

## Eval Infrastructure

Launch on EC2 spot (t3.medium):
```bash
bash infra/launch_mpc_eval.sh [--ckpt s3://...] [--steps N] [--map NAME]
```

Outputs uploaded to `s3://leworldduckie/evals/mpc/<run_id>/`:
- `results.txt` — stdout summary (all episodes)
- `instance.log` — full EC2 bootstrap + eval log
- `best_episode.gif` — best episode by total reward
- `ep0.gif ... ep9.gif` — all episode GIFs
- `progress.txt` — updated after each episode (live monitoring)

Monitor live:
```bash
aws s3 cp s3://leworldduckie/evals/mpc/<run_id>/progress.txt -
```

## Eval Results

### Run 20260424_180243 (FRAMESKIP=1, vel_weight=1.0, colab_v1 checkpoint — LaneFollowController data)

```
0/10 episodes successful
Best: ep=7, 234 steps, reward=-4.163
Mean: 63.7 steps, reward=-37.376
```

Significant improvement over previous runs with the new checkpoint trained on `LaneFollowController` data.
Best episode survived 234 steps (vs 103 with PDController checkpoint). z_dist behaviour is still high (~20-22) and generally increasing, but ep=5 showed z_dist starting at ~16.4 and briefly decreasing — the first sign of partial latent guidance. The planner still doesn't converge to the goal but the representations are meaningfully better.

### Run 20260424_153252 (FRAMESKIP=1, vel_weight=1.0, steer_weight=0.1, PDController checkpoint)

```
0/10 episodes successful
Best: ep=7, 103 steps, reward=-10.1
Mean: 48.7 steps, reward=-36.6
Typical: goes straight, does not follow lane
```

Velocity reward added (`vel_weight=1.0`) broke the spinning local minimum — agent now moves forward instead of spinning. However, it just goes straight regardless of lane geometry: confirmed by inspecting GIFs.

`z_dist` diagnostic showed ~20-21 and **increasing** throughout every episode. This is not a local-minimum issue — it means the agent is drifting away from the goal latent, i.e. the PDController checkpoint's latent representations are too poor to guide the planner at all. The vel_weight fix is masking the underlying model quality problem by pushing the agent forward with no latent guidance.

Root cause: checkpoint trained on old `PDController` data (not `LaneFollowController`).

### Run 20260424_140919 (FRAMESKIP=1, PDController checkpoint, no vel reward)

```
0/10 episodes successful
Best: ep=6, 88 steps, reward=-10.7
Typical: spinning in circles
```

### Previous runs (FRAMESKIP=3)

All 0/10. Core cause was policy-entanglement (see `lessons.md`).

## Goal Image

Stored at `s3://leworldduckie/evals/mpc_goal.png`. A representative frame from a lane-following trajectory. The MPC drives toward latent states similar to this frame.

**Limitation**: a single goal frame may not capture the full diversity of "good" lane-following. Future work: use a goal trajectory or a lane-centre reward instead.

## Known Issues

### Policy-entanglement (fixed with FRAMESKIP=1)

See `lessons.md` for full explanation. Short version: training with FRAMESKIP=3 on 1-step data creates a distribution mismatch at MPC time because MPC repeats the same action 3 raw steps in a row (never seen during training).

### Single-frame goal limitation

The MPC minimises MSE to a single goal latent. This is sensitive to which frame is chosen and doesn't encode "follow the lane continuously". A better objective would be a lane-centre reward function or online goal conditioning from a reference trajectory.

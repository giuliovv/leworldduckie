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
cost = sum_over_horizon( MSE(pred_emb, z_goal) )
     + 0.01 * action_smoothness_penalty
     - vel_weight * mean_vel                          # forward-vel reward (off by default in trajectory mode)
     + vel_lambda * clamp(vel_floor - mean_vel, 0)²   # velocity floor (default: floor=0.4, lambda=50)
```

`--vel-floor` and `--vel-lambda` are CLI args. Velocity floor confirmed to bind at lambda=50.

## Action Indexing (verified from training code)

Training `step_fn`: `ctx_emb = emb[:, :3]`, `ctx_act = act_emb[:, :3]`, `tgt_emb = emb[:, 1:]`

So `predict([z0, z1, z2], [a0, a1, a2])[-1] ≈ z3`.

`ctx_act[-1] = a2` = action at the last context position, predicts 1 subsampled step ahead.

For MPC: `ctx_act[-1] = a_t^(sampled)` → predicts `z_{t+1}`. Correct with FRAMESKIP=1.

## Checkpoints

| Name | Training FRAMESKIP | Data | Notes |
|------|--------------------|------|-------|
| `colab_v1` | 1 | LaneFollowController 100k (2026-04-24) | Current best; used in all MPC evals from 20260424_180243 onward |
| `PDController` | 3 | PDController (old) | Deprecated; latent quality too poor for guidance |

> colab_v1 confirmed trained at FRAMESKIP=1 (matches latent_index.npz and MPC operating condition — no distribution shift in T4 diagnostic).

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

## Model Diagnostics (colab_v1, run 20260425_094044)

Results from T4 (rollout error) and T5 (linear probe) on the colab_v1 checkpoint.
Both use FRAMESKIP=1 — no distribution shift from training.

### T4 — Rollout error vs horizon (n=300 sequences, max_horizon=15)

| k  | mean L2 | ±std  |
|----|---------|-------|
| 1  | 0.861   | 0.192 |
| 2  | 1.403   | 0.291 |
| 3  | 1.928   | 0.413 |
| 5  | 2.913   | 0.595 |
| 10 | 5.317   | 1.060 |
| 15 | 7.650   | 1.583 |

Error doubles by k≈3, then grows linearly ~0.5 L2/step. No catastrophic cliff — the predictor degrades gracefully. At the MPC horizon of k=10, cumulative error is ~5.3 L2.

### T5 — Linear probe z → steering (lane-offset proxy, 80 epochs)

**Val R² = 0.9816**

The encoder strongly captures lane position. R²≈0.98 rules out encoder quality as the MPC bottleneck.

### T6 — Steering sensitivity: predictor action-conditioning (2026-04-26, run 20260426_082524)

Tests whether the ARPredictor distinguishes left-vs-right steering at H=3. 100 context windows where |steering| < 0.2 from the training HDF5. Rollout k=3 steps with `a_right = [0.4, +0.5] × 3` and `a_left = [0.4, -0.5] × 3`.

| metric | value |
|--------|-------|
| Mean L2(z_right_3, z_left_3) | **0.370 ± 0.077** |
| Median | 0.358 |
| Min / Max | 0.242 / 0.597 |
| T4 noise floor at k=3 | 1.928 |
| Ratio (signal / noise) | **0.19×** |

**Verdict: FAIL — the predictor is action-blind.**

Hard right vs hard left (±0.5) for 3 steps produces latents only 0.37 apart, buried under 1.93 of rollout noise. The steering signal is 5× smaller than the noise floor. The ARPredictor is not meaningfully incorporating action conditioning.

Implications:
- Increasing CEM samples (N=1000, more iterations) will not help — the cost landscape is flat with respect to steering.
- Bezier-anchored goals will not help — even correct goals are unreachable if the predictor ignores the actions needed to reach them.
- The root cause is in training: the ARPredictor either (a) has a bug where the action encoder output is not properly integrated, or (b) minimises the JEPA prediction loss by ignoring actions (visual context alone is sufficient to predict the next frame on smooth training trajectories).
- **Retraining is required**, with either a debugging pass on action-conditioning wiring or an auxiliary action-contrastive loss.

### Diagnosis (updated 2026-04-26)

T5 confirmed the encoder is strong (R²=0.98). T6 now identifies the actual bottleneck: the predictor does not use action conditioning. All MPC failures — self-gaming, poor steering at real speed, inability to close the latent gap — are downstream of this single root cause. The CEM is planning in a cost landscape that is nearly flat with respect to steering, so it cannot distinguish good trajectories from bad ones.

## BC Baseline

A 2-layer MLP (192→256→64→2) trained on (z_t, a_t) pairs from the latent index + HDF5 (frozen encoder, BC imitation). Trained 50 epochs, Adam lr=1e-3, batch 2048, 80/20 train/val split. Run on the same 10-episode Duckietown eval.

### Run bc_20260425_114123

```
Success: 1/10
Mean steps: 117.2
```

| ep | map          | steps | success |
|----|--------------|-------|---------|
| 0  | loop_empty   | 63    | fail    |
| 1  | straight_road| 81    | fail    |
| 2  | straight_road| 9     | fail    |
| 3  | loop_empty   | 219   | fail    |
| 4  | loop_empty   | 132   | fail    |
| 5  | straight_road| 31    | fail    |
| 6  | udem1        | **300** | **SUCCESS** |
| 7  | udem1        | 51    | fail    |
| 8  | straight_road| 33    | fail    |
| 9  | udem1        | 253   | fail    |

BC rewards are very negative (-968 to -2565) vs MPC (-7 to -28) — the BC agent goes off-road more aggressively but survives longer in step count. Success criterion is steps ≥ 300.

**BC is qualitatively superior to all MPC variants.** It drives fast, takes turns, and behaves like an actual lane-follower — it just crashes occasionally. By contrast, H=3 MPC achieves better raw step counts by crawling nearly straight and barely moving, which inflates its numbers without reflecting real lane-following behaviour. A direct z→a MLP sidesteps the planning problem by never needing to predict across the latent gap.

### Run bc_20260425_232845 (3-frame context BC)

```
Success: 0/10
Mean steps: 98.6
```

Regressed vs single-frame (1/10). The 3-frame temporal context (input 576-dim flattened) did not help with the same data volume. The bc_controller.py code already had BC_HISTORY=3 prior to this run.

## Self-Gaming Diagnosis (2026-04-26)

Per-step diagnostic fields added to mpc_controller.py: `vel` (mean CEM chosen velocity), `g_spc` (mean L2 between consecutive z_goal frames), `z_dist`.

**Run 20260425_232854 (H=3, offset=0):** Confirmed self-gaming. Successful episodes had vel=0.153 and vel=0.162. Pattern: high-vel episodes (>0.4) crash in ≤76 steps; low-vel episodes "succeed" by parking. g_spc=0.66 (goal trajectory is dynamic), so the agent actively chooses low velocity rather than exploiting a static goal.

**Run 20260425_232902 (H=3, offset=10):** Marginally better (mean steps 165.9 vs 146.9). First-ever positive reward (ep9=+0.105). Same self-gaming pattern — success episodes still vel≈0.15.

**Fix attempt — vel floor (run 20260426_010556, H=3, offset=10, vel_floor=0.4, vel_lambda=50):**

```
Success: 1/10
Mean steps: 90.5
Mean vel: 0.447  ← floor binds ✓
Mean reward: -29.292
```

Per episode:
| ep | steps | reward    | vel   | result  |
|----|-------|-----------|-------|---------|
| 0  | 75    | -13.781   | 0.560 | fail    |
| 1  | 10    | -102.142  | 0.452 | fail    |
| 2  | 104   | -9.253    | 0.464 | fail    |
| 3  | 16    | -64.771   | 0.354 | fail    |
| 4  | 180   | -6.507    | 0.427 | fail    |
| 5  | 69    | -33.720   | 0.441 | fail    |
| 6  | 59    | -17.445   | 0.539 | fail    |
| 7  | **300** | **-0.365** | 0.407 | SUCCESS |
| 8  | 51    | -20.473   | 0.395 | fail    |
| 9  | 41    | -24.461   | 0.429 | fail    |

Floor binds but mean steps drops (90.5 vs 165.9). Interpretation: **case 2 — agent drives at target speed but steers badly.** CEM with N=200, 3 iters doesn't find reliable steering trajectories at proper speed.

**T6 update (2026-04-26):** Case 2 is not a CEM budget problem. T6 showed the predictor is action-blind (steering sensitivity 0.37 vs noise floor 1.93). More CEM samples cannot help when the cost landscape is flat w.r.t. steering. Root cause is in training — see T6 section above.

## Eval Results

### Run 20260425_112855 (H=3, FRAMESKIP=1, goal-mode=trajectory, colab_v1 checkpoint)

Shortest horizon tested. Hypothesis: a 3-step planning window keeps predictor rollouts within the reliable L2 range, reducing compounding error enough for the CEM to close the latent gap.

```
Success rate : 3/10 (30%)
Mean steps   : 150.0
Mean reward  : -22.054
Best         : ep=7, 300 steps, reward=-0.042
```

Per episode:
| ep | map                          | steps   | reward    | result  |
|----|------------------------------|---------|-----------|---------|
| 0  | robotarium2                  | 76      | -13.68    | fail    |
| 1  | straight_road                | 9       | -113.22   | fail    |
| 2  | ETHZ_autolab_technical_track | 106     | -9.25     | fail    |
| 3  | ETH_small_loop_1             | 30      | -35.75    | fail    |
| 4  | TTIC_loop                    | **300** | **-0.67** | SUCCESS |
| 5  | udem1                        | 207     | -17.15    | fail    |
| 6  | robotarium2                  | **300** | **-0.91** | SUCCESS |
| 7  | robotarium2                  | **300** | **-0.04** | SUCCESS |
| 8  | udem1_empty                  | 125     | -8.43     | fail    |
| 9  | robotarium1                  | 47      | -21.45    | fail    |

z_dist range: ~9–13 throughout (same order as H=8, well below the ~20–22 seen at H=10). The shorter horizon keeps each rollout step within the reliable predictor region.

**First MPC configuration to achieve success (3/10, 30%).** On paper outperforms all MPC horizons and the BC baseline in step count. However, **qualitatively the H=3 agent was moving very slowly and nearly straight** — it survived long enough to score "success" by not crashing, not by following the lane. The rewards on successful episodes (-0.67, -0.91, -0.04) are very low magnitude but still negative, consistent with minimal forward progress rather than clean lane-following.

In contrast, **the BC agent drove fast and took turns** — it behaved more like a real lane-follower, just not reliably enough to survive 300 steps in most episodes. BC's step-count failures reflect aggressive driving that goes off-road, not a slow creep to the boundary.

**Conclusion: the step-count metric is misleading here.** H=3 MPC "succeeds" by barely moving forward; BC is behaviourally superior despite lower measured success rate. The latent-goal MPC has not solved lane-following — it has learned to survive by going slow.

### Run 20260425_102408 (H=8, FRAMESKIP=1, goal-mode=trajectory, colab_v1 checkpoint)

Horizon calibrated from pairwise z-distance: mean_pairwise=18.74, threshold=0.25×18.74=4.684, max k where rollout_error[k]<threshold → **H=8**.

```
Success rate : 0/10
Mean steps   : 94.0
Mean reward  : -24.852
Best         : ep=5, 245 steps, reward=-14.265
```

Per episode:
| ep | map                          | steps   | reward  | result |
|----|------------------------------|---------|---------|--------|
| 0  | robotarium2                  | 76      | -13.69  | fail   |
| 1  | straight_road                | 10      | -102.14 | fail   |
| 2  | ETHZ_autolab_technical_track | 73      | -13.58  | fail   |
| 3  | ETH_small_loop_1             | 29      | -36.89  | fail   |
| 4  | TTIC_loop                    | 139     | -8.47   | fail   |
| 5  | udem1                        | **245** | -14.27  | fail   |
| 6  | robotarium2                  | 87      | -12.55  | fail   |
| 7  | robotarium2                  | 152     | -7.02   | fail   |
| 8  | udem1_empty                  | 94      | -11.19  | fail   |
| 9  | robotarium1                  | 35      | -28.73  | fail   |

Mean steps improved from 84.4 (H=10) to **94.0** (+11%). Best episode improved from 198 (H=10) to **245 steps**. z_dist dropped from ~20–22 (H=10) to **9–13** — the planner is operating in a more reliable region of the predictor. Still 0/10 overall, but survival time is meaningfully higher.

### Run 20260424_211453 (H=10, FRAMESKIP=1, goal-mode=trajectory, colab_v1 checkpoint)

```
Success rate : 0/10
Mean steps   : 84.4
Mean reward  : -26.333
Best         : ep=5, 198 steps, reward=-15.250
```

Per episode:
| ep | map                          | steps   | reward  | result |
|----|------------------------------|---------|---------|--------|
| 0  | robotarium2                  | 75      | -13.83  | fail   |
| 1  | straight_road                | 9       | -113.22 | fail   |
| 2  | ETHZ_autolab_technical_track | 69      | -14.39  | fail   |
| 3  | ETH_small_loop_1             | 31      | -34.68  | fail   |
| 4  | TTIC_loop                    | 123     | -9.43   | fail   |
| 5  | udem1                        | **198** | -15.25  | fail   |
| 6  | robotarium2                  | 72      | -14.74  | fail   |
| 7  | robotarium2                  | 148     | -7.22   | fail   |
| 8  | udem1_empty                  | 83      | -12.64  | fail   |
| 9  | robotarium1                  | 36      | -27.95  | fail   |

Trajectory-goal mode improved mean survival by 32% (84.4 vs 63.7 steps). Best episode was ep=5 (udem1) at 198 steps. The trajectory KNN retrieval gives the planner a richer goal signal than one fixed frame, though the overall z_dist behaviour (~20–22, generally increasing) persists — confirming the latent quality is still the binding constraint, not the goal representation.

**t-SNE diagnostic (run colab_v1)**: Separation ratio T1 = 2.51 (ambiguous zone). The t-SNE plot showed well-mixed episode colours rather than clearly isolated clusters, suggesting Case 2 (encoder learned lane position, episodes differ by track region) rather than Case 1 (memorised episode-specific features). **Confirmed by T5 R²=0.98**: the encoder does capture lane geometry.

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

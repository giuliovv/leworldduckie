# Lessons Learned & Current Diagnosis

This document is the current summary of the Duckietown LeWM investigation as of 2026-04-30.

## Goal

Apply LeWorldModel (Maes et al. 2026, arXiv:2603.19312) as the dynamics model for an MPC lane-following controller in gym-duckietown.

## High-Level Outcome

- Encoder quality is strong.
- Predictor rollout quality is usable at short horizon.
- MPC failure is not primarily CEM tuning.
- Main issue is weak action influence in predictor outputs relative to state/history influence.

## Chronology of Findings

1. Encoder quality validated.
- Linear probe `z -> steering` reached `R^2 ~= 0.98`.
- Latent structure diagnostics did not indicate catastrophic collapse.

2. Predictor rollout error characterized.
- Error grows roughly linearly with horizon (~0.5 L2 per step).
- Practical planning horizon is short (~3-4 steps).

3. MPC self-gaming observed.
- With latent-goal cost, agent discovered "move minimally" behavior.
- Episodes labeled as success often had low mean velocity (~0.15).

4. Velocity-floor intervention exposed steering failure.
- Forcing forward speed (~0.45) removed parking behavior.
- Agent then moved faster but crashed, indicating poor steering control.

5. T6 steering sensitivity showed weak action conditioning.
- `frameskip=1` checkpoint: action-separation/noise ratio ~`0.19x`.
- `frameskip=3` checkpoint: ratio ~`0.33x`.
- Action effect was much smaller than rollout noise floor.

6. Identity shortcut hypothesis investigated.
- Consecutive-frame latent distance was too small to force strong action use.
- Predictor could satisfy JEPA loss via near-identity temporal extrapolation.

7. Architecture/wiring audit.
- AdaLN wiring and gradient flow appeared technically correct.
- No single obvious implementation bug was found that fully explains behavior.

8. Retraining with stronger temporal stride.
- `frameskip=6, n_preds=4` reduced pure identity shortcut viability.
- Predictor now modeled change better than before.

9. But action conditioning remained weak.
- New checkpoint T6 ratio improved only to ~`0.86x` (still under noise floor).

10. History-shortcut hypothesis tested and rejected.
- Randomized-history T6 stayed similar (~`0.89x`).
- Weak action signal persisted even with degraded history coherence.

11. Encoder-redundancy hypothesis confirmed.
- Sensitivity test:
  - output std under varied actions, fixed state: ~`0.020`
  - output std under varied states, fixed action: ~`0.999`
  - ratio ~`50.6x` (state dominates action).

## Current Interpretation

The predictor is not literally action-agnostic, but in this domain the action signal is dominated by state/history signal.

A plausible explanation is that single-frame latent `z_t` already encodes enough information about current motion and correction trend, so minimizing 1-step JEPA prediction error does not force heavy reliance on explicit action inputs.

## Practical Implications

- Increasing CEM budget alone is unlikely to fix steering quality.
- Better metrics are needed: raw step-count can overrate "slow survival" behavior.
- Future model work should explicitly increase action identifiability in dynamics learning.

## Recommended Next Work

1. Keep Duckietown as a LeWM-methodology testbed, not a paper-replication benchmark.
2. Run one official LeWM benchmark (e.g., Push-T) to verify external baseline reproducibility.
3. For Duckietown dynamics learning, prioritize objectives/augmentations that force action use.
4. Evaluate with behavior-centric metrics (lane tracking quality, steering-response tests), not only episode length.

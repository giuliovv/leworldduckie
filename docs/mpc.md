# MPC Lane-Following Controller

This document describes how MPC is used in Duckietown and what limits performance.

## Concept

MPC plans in latent space:
1. Encode current observation to `z_t`.
2. Define goal latent(s) `z_goal` (single frame or trajectory goals).
3. Use CEM to find action sequences minimizing latent rollout cost.

## CEM Design (Current)

- Samples per iter: `N=200`
- Horizon: commonly `H=3..10` (short horizons perform less badly)
- Iterations: `3`
- Warm start: shift previous plan by one step

Cost uses latent goal error plus action smoothness and optional velocity shaping (`vel_floor`, `vel_lambda`).

## What Was Ruled Out

- Pure encoder failure: ruled out (`z -> steering` probe is strong).
- Catastrophic predictor instability: ruled out (rollout degrades gradually, not explosively).
- CEM under-budget as primary cause: unlikely after T6-style diagnostics.

## Main Failure Mode

MPC can optimize for low-motion "survival" under latent-goal cost unless velocity is constrained.
When velocity constraints are added, steering quality remains poor because predictor outputs are weakly sensitive to action compared to state/history.

## Current Diagnosis (2026-04-30)

- In Duckietown, predictor output variance from varying input state/history is much larger than from varying actions.
- This makes planning landscape relatively flat with respect to steering choices.
- Result: MPC cannot reliably distinguish steering-good from steering-bad trajectories even when moving forward.

In short: this is a dynamics-identifiability issue more than a planner-search issue.

## Implications for Validation

- Episode length alone is misleading (slow creeping can look "successful").
- Validation should include steering-response tests and lane-tracking behavior metrics.
- Comparing to BC baselines remains useful as a behavior sanity check.

## Next Priorities

1. Improve action identifiability in training objective/data regimen.
2. Keep temporal semantics consistent between training and MPC execution.
3. Maintain T6/sensitivity diagnostics as a precondition before expensive MPC sweeps.

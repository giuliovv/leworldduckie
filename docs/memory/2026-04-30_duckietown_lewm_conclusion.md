# Memory Note (2026-04-30): Duckietown LeWM Conclusion

- Main finding: failure mode is task-architecture interaction, not a simple code bug.
- In smooth lane-following dynamics, current latent state strongly predicts next latent state; action signal is dominated by state/history signal.
- MPC infrastructure is usable, but performance is limited by weak action identifiability in learned dynamics for this domain.
- Encoder quality validated; predictor stable but weakly action-sensitive under this task.
- Next external validation: run official LeWM benchmark task (e.g., Push-T) to verify baseline reproducibility.

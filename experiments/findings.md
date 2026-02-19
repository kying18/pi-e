# Findings Log

Use one short entry per completed ablation slice.

## Template

### [DATE] [RUN GROUP]

- Hypothesis:
- Result:
- Evidence (run_ids):
- Failure modes observed:
- Decision / Next action:

---

### 2026-02-19 Initial baseline snapshot

- Hypothesis: RH execution improves learned policy behavior vs open-loop.
- Result: Supported for ACT/ViT in current environment.
- Evidence (run_ids): `act_rh4_s42`, `vit_rh4_s42`
- Failure modes observed: occasional long-tail episodes in BC-style policies.
- Decision / Next action: run capacity and data-scale ablations with same eval protocol.


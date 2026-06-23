# Previous Work Summary — SONIC Token VLA (knowledge transfer)

**Who this is for:** an agent about to modify **DiT4DiT**. The work below happened in the parent
repo `humanoid-vla` (one level up from this `dit4dit/` submodule). It built and validated the SONIC
**motion-token** pipeline + datasets that DiT4DiT will most likely model. Everything about the token
space, the datasets, the decode/eval tooling, and (especially) the **pitfalls** transfers directly.

> Trust nothing here blindly — every non-obvious claim has a verification command. We were burned
> repeatedly this project by trusting comments/assumptions; verify with the scripts in §6.

---

## 1. The idea

A VLA for humanoid (Unitree G1) whole-body control that **predicts SONIC motion tokens instead of
raw joint actions**. SONIC (NVIDIA GEAR) maps human motion *and* G1 robot motion into one
**FSQ-discretized latent space**; a **frozen SONIC decoder** turns tokens back into executable
29-DOF motion at 50 Hz. Two payoffs we exploit:
1. The action space collapses from 43-DOF raw joints to a compact, physically-grounded 64-d latent.
2. Human egocentric data becomes direct `(obs, language, token)` supervision — the data engine.

**Pipeline:** `ego RGB + language + (proprio or previous tokens) → policy → SONIC token chunk
[50, 64] (1 s @ 50 Hz) → frozen SONIC decoder (closed-loop, 50 Hz) → 29-DOF joint targets`.
Hands (14 DOF) are **not** in the token space — a separate hand head handles them.

We implemented the policy on a **pi0.5** backbone (in `openpi/`). DiT4DiT is presumably an
alternative/complementary modeling direction over the **same token target** — so the modeling code
differs, but the **token spec, datasets, and eval tooling below are shared.**

---

## 2. SONIC essentials (you MUST respect these)

- **Encoder**: one ONNX, `1762-d obs → 64-d token`. Mode set by the first 4 input dims
  (0 = G1 robot, 1 = VR-teleop, 2 = SMPL human). Wrapper + exact input layout:
  `utils/sonic_encoder_3modes.py` (`run_g1`, `run_teleop`, `run_smpl`). Encoder ONNX is **batch=1**
  (loop one frame at a time; for small models set `intra_op_num_threads=1` or it thrashes).
- **Token**: 64-d FSQ code (2 tokens × 32 dims, values on the FSQ grid, range ≈ [-0.875, 0.81]).
  Each per-frame token encodes ~1 s of future motion (10 future frames at stride 5). Consecutive
  per-frame tokens overlap ~98% — beware copy-forward shortcuts (predicting target ≈ previous token).
- **Decoder**: ONNX `994 → 29`. It is a **closed-loop balance POLICY, not a motion player.**
  Input = `token(64) + 10-frame history of [base_ang_vel(3), joint_pos(29), joint_vel(29),
  last_action(29), gravity_dir(3)]` (= 64 + 10×93 = 994). Output = normalized 29-DOF joint targets,
  PD-tracked in sim. **No hands.** Frame order in the history is **oldest→newest**; **no per-channel
  scaling**; `last_action` = the decoder's own previous raw output. PyTorch port (matches ONNX to
  ~8e-6): `utils/sonic_decoder_pytorch.py`. Consequence: it **needs a warm-start from a real
  balanced state** (joint pos/vel + base quat) or it free-falls — a synthetic cold start fails.
- **50 Hz** native everywhere; a 1 s action chunk = `[50, 64]`.
- **JOINT ORDER — the #1 footgun.** The encoder & decoder *networks* operate in **IsaacLab joint
  order**. The datasets store joints in **"SONIC-grouped" / MuJoCo order**:
  `[left_leg(6), right_leg(6), waist(3), left_arm(7), right_arm(7)]` (= the order in
  `gear_sonic_deploy/g1/scene_29dof.xml`). Convert SONIC-grouped → IsaacLab with the permutation
  `_MJ_TO_IL` (== deploy `mujoco_to_isaaclab`):
  `[0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28]`.
  **Anything fed to the encoder/decoder network must be IsaacLab order; stored dataset joints stay
  SONIC-grouped.** (This is exactly what we got wrong for HE — see §4.)
- **Quaternions are wxyz everywhere** in converted data.
- **Normalization**: FSQ tokens use **identity** (NOT mean/std — the decoder snaps to the grid). No
  norm stats shipped/needed for the token target.
- **SONIC artifacts** live at
  `/lustre/home/malbaba/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/`:
  `model_encoder.onnx`, `model_decoder.onnx`, `model_decoder_state_dict.pt`; scene
  `gear_sonic_deploy/g1/scene_29dof.xml`. The default paths inside the `utils/` wrappers are stale —
  pass explicit paths.

---

## 3. Datasets (all converted, on disk, VERIFIED CORRECT as of 2026-06-18)

| Dataset | Embodiment | Episodes | Location | Format |
|---|---|---|---|---|
| **Humanoid Everyday (HE)** | real G1, 29 body + 14 hand DOF | 4064 (879 `Locomanip`) | `/ps/project/datasets/humanoid_everyday/sonic_vla_50hz` | LeRobot v2.1 |
| **LeVERB_Bench** | sim G1 (IsaacLab), no hands | 3555 | `/lustre/home/malbaba/leverb_sonic_vla_50hz` | LeRobot v2.1 |
| **Xperience-10M** | human egocentric (SMPL mocap) | ~12.7k encoded | `/ps/project/datasets/robo-xperience-10m` | per-episode dirs |

**Shared LeRobot SONIC-VLA schema (robot datasets):**
- `observation.images.ego_view` — 480×640 video @ 50 fps (LeVERB also has `tpv_view`).
- `observation.state` (43) — SONIC-grouped order (legs, waist, arms, hands). **THE TRAINING TARGET
  is `action.motion_token` (64) — the SONIC token.** `action.wbc` (43) = next-frame joint positions
  (raw-action baseline). `observation.root_orientation` (4, wxyz), `observation.eef_state` (14).
  `teleop.*` are mostly zero-padded placeholders — **drop them** from data configs.
- **Xperience** = per-episode dirs (`tokens.npy` (T,64), `valid.npy` occlusion mask (~24% frames
  masked/episode), `meta.json` with `token_alignment`). SMPL human source; cross-embodiment decode
  to G1 is validated.

**Splits** (`pi05_sonic_vla/data/sonic_token_dataset.py`, deterministic crc32-of-path hash):
- `train` (3744 HE eps) — excludes both holdouts below.
- `eval` (206 eps) — random ~5% of HE, all categories. General in-loop metric.
- `test_locomotion` (114 eps) — held-out ~13% of the HE **`Locomanip`** category (HE has no pure-
  locomotion category; Locomanip = walk+manipulate). The dedicated locomotion test set.
- Disjoint by construction (verified train∩eval, train∩test, eval∩test all = 0).

---

## 4. What we did (chronological, with the big one)

1. **Wired SONIC token prediction into openpi/pi0.5** — config `pi05_sonic` (action_dim=64,
   action_horizon=50, prev_token_history=50 projected into the prefix, masked flow-matching loss for
   occluded targets, FSQ identity norm). Files: `openpi/src/openpi/{models/pi0.py, pi0_config.py,
   models/model.py, policies/sonic_policy.py, training/config.py}`. *Relevant to DiT4DiT only as a
   reference for how the token target is consumed.*
2. **Eval stack** (`pi05_sonic_vla/eval/`): in-loop token metrics (MSE/MAE/cosine over valid
   tokens); a **decoupled sim-viz** that decodes predicted+GT tokens in MuJoCo and renders
   side-by-side `GT‖pred` mp4s (warm-started); a checkpoint **watcher**; **autoregressive
   multi-second** eval (`--n-chunks`); the locomotion test split. Decode harness reused verbatim:
   `explore/token_decode_validation.py`.
3. **⚠️ FOUND & FIXED A CRITICAL DATA BUG: HE tokens were encoded in the WRONG joint order.**
   `data/humanoid_everyday_converter/convert_fast.py` (the documented production converter) **omitted
   the `_MJ_TO_IL` permutation**, feeding the encoder SONIC-grouped order instead of IsaacLab. The
   tokens were mis-ordered. They still decoded to *some* walking gait (the decoder is robust), which
   masked the bug — but they did **not** reproduce the recorded motion.
   - **How we proved it** (three independent, agreeing lines):
     - A/B re-encode (`he_reencode_check.py`): on-disk tokens == SONIC-grouped re-encode **cos 1.0**,
       vs IsaacLab re-encode **cos 0.18** → confirms which order the data used.
     - Decode **fidelity** vs recorded joints (`he_decode_fidelity.py`, the *correct* metric):
       IsaacLab order legs RMSE **3.2°** vs wrong order **12.6°** (arms 5° vs 27°).
     - Side-by-side **video** (`he_compare_render.py`): RECORDED | STORED | IsaacLab — the IsaacLab
       panel tracks the recorded gait, the stored one is visibly off.
   - **Fix**: added `_MJ_TO_IL` to `convert_fast.py` and permuted **only the encoder input**
     (`observation.state`/`action.wbc`/`eef` stay SONIC-grouped). Re-converted all 4064 HE episodes;
     verified the production dataset (`stored vs IsaacLab cos = 1.0000`). See HE README **§8**.
   - **LeVERB and Xperience were audited and are NOT affected** (LeVERB feeds IsaacLab order;
     Xperience is SMPL-mode with a verified root field). The bug was HE-`convert_fast`-specific.

---

## 5. Hard-won lessons (read these — they are the real value of this doc)

1. **Correctness of a token = does `encode→decode` reproduce the *recorded* motion** (joint-RMSE vs
   recorded, esp. legs, early window). **"Does it walk / how far does the base travel" is NOT a
   correctness signal** — a wrong-joint-order token still decodes to a balanced walking-ish gait and
   can even translate *more*. We wasted time on translation distance; fidelity is the metric.
2. **Verify against ground truth + empirics, never comments or prior-agent claims.** Ground truth =
   the deploy C++ (`gear_sonic_deploy`) and the released ONNX, plus the encode→decode fidelity test.
   A converter labeled "identical numerics / fast drop-in" was *not* identical.
3. **The decoder is closed-loop** → warm-start from the episode's real state (`observation.state[t,:29]`
   as `q0`, finite-diff `qd0×50`, `observation.root_orientation[t]` as base quat). "Survival/upright"
   ≠ motion fidelity; the offline MuJoCo harness was only ever validated for survival.
4. **"VLA doesn't walk" in eval can be a short-window artifact** — decoding 1 s chunks moves the base
   ~3 cm, which reads as "marching in place" in a tracking camera. Judge locomotion on multi-second
   (autoregressive) rollouts, not 1 s chunks.
5. **Operational footguns**: `/ps/project` NFS can throttle to ~1 MB/s (test one copy first); the
   encoder ONNX is batch-1; importing the heavy converter modules can hang (inline pure functions for
   quick checks); quats are wxyz; FSQ tokens use identity norm.

---

## 6. Verify the tokens yourself (commands)

```bash
PY=/lustre/home/malbaba/GR00T-WholeBodyControl/.venv_sim/bin/python   # env: mujoco, onnxruntime, torch, pyarrow

# (a) token JOINT ORDER of any episode — correct ⇒ "stored vs SLOW (IsaacLab) cos≈1.0"
$PY ../explore/he_token_order_debug/he_reencode_check.py \
   --parquet /ps/project/datasets/humanoid_everyday/sonic_vla_50hz/data/chunk-002/episode_002632.parquet

# (b) decode FIDELITY vs recorded motion (lower leg-RMSE = correct order)
$PY ../explore/he_token_order_debug/he_decode_fidelity.py

# (c) decode any stored tokens in MuJoCo (survival / motion sanity) — the reusable harness
$PY ../explore/token_decode_validation.py --help
```
(Paths above are relative to this `dit4dit/` dir; `../` = the `humanoid-vla` repo root.)

---

## 7. Key file map (in the parent `humanoid-vla` repo)

- **Converters** (+ a README each): `data/humanoid_everyday_converter/` (`convert_fast.py` =
  production; **README §8 = the joint-order bug**), `data/leverb_bench_converter/`,
  `data/xperience10m_converter/`.
- **Dataset / dataloader + splits**: `pi05_sonic_vla/data/sonic_token_dataset.py`.
- **Eval**: `pi05_sonic_vla/eval/` (`predict_chunks.py`, `decode_and_render.py`, `run_sim_eval.sh`,
  `watch_and_eval.sh`, `token_metrics.py`, `README.md`).
- **SONIC wrappers**: `utils/sonic_encoder_3modes.py`, `utils/sonic_decoder_omnx.py`,
  `utils/sonic_decoder_pytorch.py`.
- **Decode/verify**: `explore/token_decode_validation.py`; this session's debug scripts +
  per-script README: `explore/he_token_order_debug/`.
- **SONIC release**: `/lustre/home/malbaba/GR00T-WholeBodyControl/gear_sonic_deploy/`.
- **Project facts (read first)**: `humanoid-vla/CLAUDE.md`.

---

## 8. Current state & what's pending

- **Datasets**: HE re-converted & verified correct; LeVERB & Xperience verified correct. Safe to
  train on now.
- **pi0.5 VLA**: must be **retrained from scratch** on the corrected HE tokens (pre-fix checkpoints
  learned wrong-order labels and are invalid). Two stale sim-eval watchers were running on those
  invalid checkpoints (`condor_rm` them).
- **For DiT4DiT**: the token target (`action.motion_token`, 64-d FSQ, identity-normalized, IsaacLab
  joint convention *inside* the encoder, SONIC-grouped in the stored state), the three datasets, and
  the decode/eval harness are ready to reuse. If you ever re-encode motion to tokens yourself,
  **apply `_MJ_TO_IL`** and A/B-verify with §6(a) before trusting the output. Evaluate generated
  tokens with the decode harness §6(c) and, where you have a reference trajectory, the fidelity test
  §6(b) — not translation distance.

*Last updated: 2026-06-18.*

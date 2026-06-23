# DiT4DiT-SONIC — the world-model-backbone arm of the SONIC token VLA

This document specifies how we adapt **DiT4DiT** to our project, and the initial TODOs. It is a
spec, not a status log — update it as decisions are made.

## Why this exists

We already have the **VLM-backbone** arm of our token VLA: pi0.5 fine-tuned to predict SONIC
motion tokens (`pi05_sonic`, see `../pi05_sonic_vla/README.md`). This directory builds the
**world-model-backbone** arm: the *same* token-predicting policy, but with the VLM swapped for a
**video world model** (Cosmos-Predict2.5-2B, via DiT4DiT's joint video+action framework).

The point is a **controlled comparison for the paper**: the literature on world-model-backbone
VLAs (VPP, GR-2, WorldVLA, the "video-model-as-backbone" class) claims they are *more
data-efficient* than VLM-backbone VLAs. We want to test that claim in our humanoid whole-body
token setting, where the action target, the data, and the eval are identical across the two arms
and **only the backbone differs**.

Consequence for design decisions: **this is a research baseline, not a deployment candidate.**
Inference cost / real-time feasibility do NOT drive choices here — a *fair, controlled head-to-head*
does. (Deployment of the winning arm is a separate, later question, owned with Jens.)

## Read first (do not re-derive)

- `../README.md` — the project thesis (SONIC token as action space + data engine).
- `previous_summary.md` (this dir) — SONIC token spec, datasets, **pitfalls**, verify commands.
- `../pi05_sonic_vla/README.md` — the VLM arm we are mirroring; the "What we changed" table is the
  template for the change-list below.
- `../CLAUDE.md` — SONIC facts (encoder/decoder, FSQ token, `_MJ_TO_IL` joint-order footgun).

Non-negotiables inherited from the token spec: token = **64-d FSQ code**, **identity
normalization** (grid codes; snap to grid at inference, never mean/std), 50 Hz native, chunk
`[50, 64]` per 1 s, quaternions wxyz, occluded human (Xperience) targets handled by a **per-frame
validity mask**, not by dropping data.

## DiT4DiT as cloned (what we start from)

A Vision-Action-Model: **Cosmos-Predict2.5-2B video diffusion transformer** backbone +
**flow-matching action DiT** head, trained in one of three modes (`video` / `action` / `joint`).

- Framework + forward: `DiT4DiT/model/framework/DiT4DiT.py`. `joint` mode returns both
  `future_video_loss` (backbone, flow-matching on future ego frames) and `action_loss` (head),
  with the head cross-attending to backbone hidden states at `extract_layer: 17`
  (`DiT4DiT.py:103-173`).
- Action head: `DiT4DiT/model/modules/action_model/ActionDiT.py` (`FlowmatchingActionHead`).
  Generic continuous flow matching; `action_dim` / `state_dim` / `future_action_window_size` /
  `action_horizon` are config values. **The loss already multiplies by `action_mask`**
  (`ActionDiT.py:316`) — masked-loss support is native, we just have to feed the mask.
- Raw-action assumptions to remove: `base_framework.py:192-202` (`unnormalize_actions`) does
  q01/q99 min-max **and** hardcodes a gripper bit at channel 6 (`normalized_actions[:, 6]`).
  Both are wrong for FSQ tokens.
- Shipped G1 config (`DiT4DiT/config/real_robot/dit4dit_g1_decoupled_wbc.yaml`): `action_dim: 36`,
  `state_dim: 64`, `future_action_window_size: 15`, `action_horizon: 16`, `training: joint`,
  `video_delta_indices: [0..16]`, `action_video_freq_ratio: 4`. **It is decoupled-WBC**: the VAM
  predicts upper-body DOF while a separate lower-body controller (`decoupled_wbc/`) handles balance.
- Dataloader: `DiT4DiT/dataloader/lerobot_datasets.py` + `dataloader/gr00t_lerobot/`. `forward`
  consumes a list of `{image, lang, action, state, action_mask}` examples.

## Target design (what we want)

```
ego video clip (N cond frames) + language + previous SONIC tokens
        │
        ▼
   Cosmos-Predict2.5-2B world-model backbone          ── joint mode ──►  future-video FM loss
        │ (hidden states @ extract_layer 17)                              (ego frames; world model)
        ▼
   flow-matching action DiT  (cross-attends backbone features)
        │
        ▼
   SONIC token chunk  [50, 64]   (1 s @ 50 Hz, one 64-d FSQ token / 20 ms tick)
        │
        ▼   (downstream, NOT in this training loop)
   frozen SONIC decoder → 29-DOF joint targets @ 50 Hz   (whole-body; balance owned by the decoder)
```

We run DiT4DiT in **whole-body token mode**, NOT decoupled-WBC: the SONIC decoder owns the whole
29-DOF body and balance, so there is no separate lower-body controller in this arm. Hands (14 DOF)
are outside the token space and out of scope here (same as the pi0.5 arm).

### Held invariant across the two arms (for a fair comparison)

| Axis | Value (identical to `pi05_sonic`) |
|---|---|
| Action target | `action.motion_token` (64-d FSQ), identity norm, snap-to-grid at inference |
| Chunk shape | `[50, 64]` per 1 s (subject to the horizon decision below) |
| Data + splits | the 3 corpora via the unified loader; `train`/`eval`/`test_locomotion` splits from `../pi05_sonic_vla/data/sonic_token_dataset.py` |
| Occlusion | per-frame validity mask → masked flow-matching loss |
| Eval | the `../pi05_sonic_vla/eval/` harness (token metrics + decode-in-MuJoCo + LeVERB sim) |

### Differs by design (the independent variable)

- **Backbone**: Cosmos video world model (here) vs PaliGemma/SigLIP VLM (pi0.5).
- **Auxiliary objective**: a future-video flow-matching loss is **on** (`training: joint`) — this is
  where the data-efficiency edge is hypothesized to come from. Running action-only would discard the
  world model and make the comparison meaningless.

## Required changes (file-mapped)

| File | Change |
|---|---|
| `DiT4DiT/config/real_robot/dit4dit_g1_sonic.yaml` (new) | Copy of the G1 config: `action_dim: 64`, `state_dim` per the conditioning decision, `future_action_window_size` per the horizon decision, `training: joint`, point `data_root_dir`/`data_mix` at our SONIC-VLA corpora, identity norm. |
| action head / `ActionDiT.py` | Set `action_dim=64`. No loss change needed (mask already applied at `:316`). Confirm `predict_action` (`:320-369`) returns raw tokens; add grid-snap at the call site, not in the head. |
| `base_framework.py:192-202` | Bypass `unnormalize_actions` for the token target (identity); remove the channel-6 gripper hardcode from this code path. Provide identity `norm_stats` so `from_pretrained` loads. |
| dataloader (`lerobot_datasets.py` / `gr00t_lerobot/`) | Emit `action = action.motion_token` `[T,64]`, `action_mask = target_valid` `[T]→[T,64]`, the previous-token history, and the ego video clip + future ego frames for the video FM target. Drop `teleop.*` placeholders. Reuse the corpus list / split logic from `../pi05_sonic_vla/data/sonic_token_dataset.py` so splits match the pi0.5 arm exactly. |
| prev-token conditioning (new, in framework or head) | A projection mirroring pi0.5's `prev_token_proj` so the world-model arm conditions on the same previous-token history (see decision #2). |
| eval adapter (new, under `examples/` or a `sonic_eval/`) | Dump DiT4DiT `predict_action` outputs (predicted + GT tokens) into the **same `.npz` format** `../pi05_sonic_vla/eval/predict_chunks.py` produces, so `decode_and_render.py` / `token_metrics.py` / the sim eval run unchanged. |

## Open design decisions (resolve before/while building; defaults proposed)

1. **Token horizon: full 50 vs subsampled.** pi0.5 predicts the full `[50, 64]` chunk
   (`action_horizon=50`). DiT4DiT ships `action_horizon=16`. Consecutive SONIC tokens overlap ~98 %
   (the "token chunk compression" open question in `../README.md`). *Default:* match pi0.5 with
   `future_action_window_size=49` (horizon 50) for a clean head-to-head; treat subsample+interpolate
   on the FSQ grid as a later ablation shared by both arms.
2. **Conditioning history (the subtle one).** pi0.5 is fully latent: ego frame + language + prev
   tokens, no proprio. DiT4DiT instead gets temporal context from the **video cond frames**, and has
   a `state_encoder` slot. For the cleanest comparison we want the *same* history signal. *Default:*
   add a prev-token projection (mirror `prev_token_proj`) and set `state=None` (no proprio), so both
   arms see ego pixels + language + prev tokens; the only extra signal the world-model arm gets is
   the multi-frame video it must also predict. Flag explicitly in the paper that the world-model arm
   inherently consumes a short video clip as input.
3. **Video window / fps / cond-frame count.** Reconcile 50 Hz tokens with the Cosmos video clip
   (`video_delta_indices`, `action_video_freq_ratio`). *Default:* a small number of ego cond frames
   + a coarse future-frame target for the video FM loss; tune for memory. Keep the token branch at
   50 Hz regardless.
4. **Native heads vs shared head.** *Default:* run native heads first for a quick directional signal
   (fast). If the data-efficiency gap is real, port the *same* flow-matching head onto both backbones
   so a gap is attributable purely to the backbone — reviewers will ask for this; budget for it.
5. **FSQ + flow matching.** Identity norm + grid-snap at inference (same as pi0.5). Verify the head
   regresses grid-valued targets cleanly; discrete classification over FSQ levels is the fallback if
   regression underfits (shared with the pi0.5 arm's open question).

## Progress (started 2026-06-20)

- **Decisions #1–#5: settled at the proposed defaults** (full 50-token horizon; prev-token
  conditioning + `state=None`; 5-frame clip `[0,12,24,36,48]` ≈ 1 s; native heads first; FSQ
  identity + grid-snap).
- **TODO 0 — DONE (verified end-to-end on an H100).** Env: `dit4dit/.venv` (uv; torch 2.7.0+cu128,
  diffusers 0.37.0.dev0 pinned, transformers 4.57.6). Weights: `/fast/malbaba/hfcache/Cosmos-Predict2.5-2B`
  (set as `framework.cosmos25.base_model` in the config). `validate_backbone.py` loads the full
  framework (**10.64B params**; action DiT ≈152M), pulls 2 real HE `eval` samples through the token
  dataloader, and confirms joint forward returns finite `action_loss` (1.05) + `future_video_loss`
  (0.49) and `predict_action` → `(2, 50, 64)`. Run:
  `cd dit4dit && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/fast/malbaba/hfcache ./.venv/bin/python validate_backbone.py`.
  (Benign warning: DiT4DiT swaps a dummy safety_checker. Predicted token range is wide/random —
  untrained, grid-snap is TODO 7.)
- **TODO 1 — DONE (verified).** `DiT4DiT/dataloader/sonic_token_dataset.py`
  (`SonicVideoTokenDataset`, subclasses the canonical loader) + `build_dataloader` branch
  (`dataset_py: sonic_token`). CPU self-test passes: splits are byte-identical to the pi0.5 arm
  (`eval`=206 HE, `train`=19,745, `test_locomotion`=114; 20,065 episodes total), samples emit
  `image=list[5]` (1 cond + 4 future), `action(50,64)`, `action_mask(50,64)`, `prev_tokens(50,64)`.
  Wrote the `_v2` index caches for all 3 corpora (Xperience scan, one-time 208 s).
- **TODO 2 — config DONE; head build pending.** `config/real_robot/dit4dit_g1_sonic.yaml`
  (`action_dim=64`, `state_dim=0`, horizon 50, `training: joint`). Identity norm is automatic
  (we bypass GR00T transforms). The `unnormalize_actions` q01/q99 + channel-6 hardcode is NOT in
  the train/predict path → its bypass + grid-snap moved to **TODO 7** (inference). Runtime head
  instantiation at 64-d folded into **TODO 5** (CPU shape check; needs only torch, not Cosmos).

## Initial TODOs (ordered)

- [x] **0. Confirm backbone access.** DONE + verified (H100). Env `dit4dit/.venv`, weights at
      `/fast/malbaba/hfcache/Cosmos-Predict2.5-2B`. Probe: `dit4dit/validate_backbone.py`.
- [x] **1. Token dataloader.** DONE + verified — splits match the pi0.5 arm exactly. Files:
      `DiT4DiT/dataloader/sonic_token_dataset.py`, `DiT4DiT/dataloader/__init__.py`.
- [x] **2. Action head → 64-d FSQ (config).** `action_dim=64` config DONE. Identity norm automatic
      (transforms bypassed). `unnormalize_actions` bypass + grid-snap → TODO 7 (not in train path);
      runtime head build → TODO 5.
- [x] **3. Prev-token conditioning.** DONE + verified (probe). `DiT4DiT.py`: `prev_token_proj`
      (`Linear(action_dim, vl_hidden_dim)`) projects the prev-token history into extra
      cross-attention context tokens (`_append_prev_token_context`, used in forward + predict_action);
      flag-gated by `action_model.use_prev_tokens` (off → original DiT4DiT path untouched). The
      projection is length-agnostic, so no `prev_token_history == data.history` assert is needed.
- [x] **4. New config** `dit4dit_g1_sonic.yaml` — DONE (created with TODO 2): `training: joint`,
      horizon 50, identity norm, our corpora, `use_prev_tokens: true`, seed 42, wandb `humanoid-vla`.
- [x] **5. Shape/contract validation.** Covered by `validate_backbone.py` (full chain on GPU) +
      the dataloader self-test (`sonic_token_dataset.py --no-images`, CPU). A pure-CPU full-model
      check isn't practical here — the Cosmos backbone needs a GPU (unlike the pi0.5 jax.eval_shape).
- [x] **6. Smoke train — PASSED (save; load pending).** 6 steps on 1×A100-80GB
      (`accelerate launch --config_file .../deepspeed_zero2.yaml --num_processes 1`, text_encoder+vae
      frozen, ~77 GB peak): both losses finite + video loss moving (0.487→0.406), eval path runs
      (`mse_score`), 20 GB `steps_4_pytorch_model.pt` written to `/fast`. NOTE: DiT4DiT detaches
      backbone features for the action loss (`Cosmos25.forward(detach=True)`), so the backbone trains
      ONLY via the video-FM loss and the head/`prev_token_proj` only via the action loss. Still TODO:
      reload via `from_pretrained` (+ the `base_framework.unnormalize_actions` token bypass, TODO 7).
      Launch: `dit4dit/`, env `HF_HUB_OFFLINE=1 HF_HOME=/fast/malbaba/hfcache WANDB_MODE=offline`.
- [x] **7. Eval adapter — DONE + verified.** `dit4dit_sonic_vla/eval/predict_chunks.py` (DiT4DiT
      `predict_action` → identical pi05 `ep*.npz{pred,gt,instruction,init_*}` + `metrics.json`,
      shared `token_metrics`, same held-out episodes) + `eval/run_sim_eval.sh` (predict in dit4dit
      venv → decode/render via `.venv_sim` reusing pi05's `decode_and_render.py` verbatim). Verified
      on the smoke checkpoint: `load_state_dict` missing=0/unexpected=0, correct `.npz`. The eval
      path uses raw `predict_action` output (no `unnormalize_actions`), so the q01/q99 + channel-6
      hardcode is moot here — it only matters for deployment.
- [x] **Condor training scripts** (`dit4dit_sonic_vla/condor/`): `run_dit4dit_sonic.sh`
      (accelerate+deepspeed, 8×A100-80GB, resume-if-checkpoint, HF-offline Cosmos, /fast ckpts,
      text_encoder+vae frozen by default, local TRITON_CACHE to dodge the flock teardown hang) +
      `dit4dit_sonic.sub`. Data/split = pi0.5 setup: all 3 corpora in training; HE held out 5% eval
      (random) + 12% of Locomanip as test_locomotion (`test_frac=0.12`, `test_category=Locomanip`).
- [ ] **8. Full train on HE (100 % data)**; report token MSE/cos + decoded fidelity + LeVERB sim
      vs the `pi05_sonic` arm on the identical `eval` / `test_locomotion` splits.
- [ ] **9. Data-efficiency curve** (the money plot): retrain both arms at {5, 10, 25, 50, 100} % of
      HE; plot fidelity/success vs data fraction. Crossover behavior is the paper claim.
- [ ] **10. Stage-2 compounding (stretch).** Add Xperience human ego video — the world-model arm can
      use it for *both* the token loss and the video FM loss; test whether the data-efficiency gap
      *widens* with human-video co-training (the strongest result).

## Eval protocol (must match the pi0.5 arm)

Reuse `../pi05_sonic_vla/eval/` verbatim via the adapter (TODO 7): in-loop token metrics
(MSE/MAE/cosine over **valid** tokens), decode-in-MuJoCo `GT‖pred` videos (warm-started from the
episode's real state), and LeVERB sim rollouts (token → SONIC decoder → IsaacLab). Correctness of a
token = decode reproduces the *recorded* motion (joint-RMSE, legs/early-window), **not** "does it
walk / how far it travels" — see `previous_summary.md §5`. The headline figure is the
data-efficiency curve (TODO 9), both arms on identical splits.

## File map (where new code lands)

```
dit4dit/
  readme_sonic.md                                  # this spec
  DiT4DiT/config/real_robot/dit4dit_g1_sonic.yaml  # new token config (TODO 4)
  DiT4DiT/dataloader/...                            # token target + prev-token + video wiring (TODO 1)
  DiT4DiT/model/modules/action_model/ActionDiT.py  # action_dim 64, grid-snap (TODO 2)
  DiT4DiT/model/framework/...                       # prev-token conditioning (TODO 3)
  sonic_eval/  (new)                                # adapter to ../pi05_sonic_vla/eval (TODO 7)
```

*Conventions:* argparse + seeds for every new script; `MMDD_`-prefixed output dirs; `metrics.json`
in output folders; skip-if-exists / `--resume` for long jobs; wandb to `humanoid-vla`. Verify token
correctness with the commands in `previous_summary.md §6` before trusting any re-encode.

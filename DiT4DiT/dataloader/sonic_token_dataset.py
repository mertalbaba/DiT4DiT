# SONIC token target for DiT4DiT (the world-model-backbone arm of the token VLA).
# See ../../readme_sonic.md.  TODO 1 in that spec.
"""Bridge the canonical SONIC token dataset into DiT4DiT's example format.

`pi05_sonic_vla/data/sonic_token_dataset.py` (at the humanoid-vla repo root) is the single
source of truth for corpus discovery and the deterministic train/eval/test_locomotion splits.
We *subclass* it so the world-model arm and the pi0.5 arm train/eval on byte-identical data;
the only additions here are (a) reading a short clip of frames for the video flow-matching
branch and (b) reformatting a sample into the dict DiT4DiT's trainer consumes:

    {
      "image":       [cond_frame, fut_1, ..., fut_K]   list of (H, W, 3) uint8
                       frame 0 = conditioning; frames 1.. = future-video FM targets
      "lang":        str
      "action":      (chunk_len, 64) f32   SONIC token chunk (raw FSQ codes, identity norm)
      "action_mask": (chunk_len, 64) f32   per-token validity broadcast over the 64 dims
      "prev_tokens": (history, 64)  f32    latent history (emitted now; consumed in TODO 3)
      "episode_ref", "window_t":           provenance for the shared eval harness (TODO 7)
    }

The clip frame offsets are derived from the same `video_delta_indices` /
`action_video_freq_ratio` the Cosmos backbone uses to compute `num_frames_out`
(`Cosmos25FeatureExtractor.forward`), so loader and backbone agree on the frame count:
    offsets = [video_delta_indices[i] for i in range(0, len(video_delta_indices), ratio)]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

# --- locate the canonical loader at the humanoid-vla repo root (one level above dit4dit/) ---
_HERE = Path(__file__).resolve()
_REPO_ROOT = os.environ.get("HUMANOID_VLA_ROOT")
if not _REPO_ROOT:
    for _p in _HERE.parents:
        if (_p / "pi05_sonic_vla" / "data" / "sonic_token_dataset.py").exists():
            _REPO_ROOT = str(_p)
            break
if not _REPO_ROOT:
    _REPO_ROOT = str(_HERE.parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pi05_sonic_vla.data.sonic_token_dataset import (  # noqa: E402
    CorpusSpec,
    SonicTokenDataset,
    TARGET_FPS,
    TOKEN_DIM,
    HAND_TOKEN_DIM,
    STATE_DIM,
    default_corpora,
    _load_state,
    _load_hand_window,
)

# Default 32-D state normalization: reuse the stats computed for the pi0.5 arm (byte-identical
# proprio data). Quantile (q01/q99) -> [-1,1], matching the pi0.5 arm + GR00T min-max spirit.
_STATE_STATS_DEFAULT = os.environ.get(
    "SONIC_STATE_STATS",
    f"{_REPO_ROOT}/openpi/assets/pi05_sonic_proprio/sonic_proprio/norm_stats.json")


def _load_state_norm(path: str):
    """Load (q01, q99) for the 32-D state from an openpi norm_stats.json; None if absent."""
    try:
        d = json.load(open(path))
        s = d.get("norm_stats", d)["state"]
        return np.asarray(s["q01"], np.float32), np.asarray(s["q99"], np.float32)
    except Exception:
        return None


def _video_offsets(video_delta_indices, action_video_freq_ratio: int) -> list[int]:
    """Frame offsets the backbone expects: every `ratio`-th entry of video_delta_indices.
    Matches `train_num_frames_out = len(range(0, n_raw, ratio))` in Cosmos25.forward."""
    vdi = list(video_delta_indices)
    ratio = max(int(action_video_freq_ratio), 1)
    return [vdi[i] for i in range(0, len(vdi), ratio)]


class SonicVideoTokenDataset(SonicTokenDataset):
    """SonicTokenDataset + a multi-frame clip + DiT4DiT's example dict."""

    def __init__(
        self,
        corpora: list[CorpusSpec],
        video_delta_indices=(0, 12, 24, 36, 48),
        action_video_freq_ratio: int = 1,
        use_state: bool = False,
        state_stats_path: str | None = None,
        **kwargs,
    ):
        super().__init__(corpora, **kwargs)
        self.video_offsets = _video_offsets(video_delta_indices, action_video_freq_ratio)
        if not self.video_offsets:
            raise ValueError("video_delta_indices/action_video_freq_ratio produced 0 frames")
        self.use_state = use_state
        self._state_norm = (_load_state_norm(state_stats_path or _STATE_STATS_DEFAULT)
                            if use_state else None)
        if use_state and self._state_norm is None:
            print(f"[SonicVideoTokenDataset] WARNING: use_state but no state stats at "
                  f"{state_stats_path or _STATE_STATS_DEFAULT} -> emitting RAW state", flush=True)
        print(f"[SonicVideoTokenDataset] clip offsets (frames @ 50 Hz): {self.video_offsets} "
              f"(cond=1, future={len(self.video_offsets) - 1}); use_state={use_state}", flush=True)

    def _read_clip(self, rec, t: int) -> list[np.ndarray]:
        """Read frames at `t + offset` for every clip offset. Source-fps aware (Xperience)."""
        blank = lambda: np.zeros((self.image_size, self.image_size, 3), np.uint8)
        if not self.load_images:
            return [blank() for _ in self.video_offsets]
        import cv2
        cap = cv2.VideoCapture(rec.image_ref)
        frames: list[np.ndarray] = []
        for off in self.video_offsets:
            fi = t + off
            if rec.image_kind != "lerobot_video":
                fi = round((t + off) * rec.fps_source / TARGET_FPS)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(fi), 0))
            ok, fr = cap.read()
            if not ok or fr is None:
                frames.append(blank())
                continue
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            if fr.shape[0] != self.image_size or fr.shape[1] != self.image_size:
                fr = cv2.resize(fr, (self.image_size, self.image_size))
            frames.append(fr)
        cap.release()
        return frames

    def __getitem__(self, _idx):
        self._maybe_anneal()   # 0905: bhs5-parity linear-in-samples mix anneal (no-op without weights_end)
        # episode + window rejection sampling (identical policy to the parent loader)
        out = None
        for _ in range(self.max_resample):
            ei = int(np.searchsorted(np.cumsum(self._ep_p), self._rng.random()))
            ei = min(ei, len(self.recs) - 1)
            rec = self.recs[ei]
            out = self._sample_window(rec)
            if out is not None:
                break
        if out is None:
            rec = self.recs[0]
            out = self._sample_window(rec) or (
                np.zeros((self.horizon, TOKEN_DIM), np.float32),
                np.zeros((self.history, TOKEN_DIM), np.float32),
                np.ones(self.horizon, bool), 0,
            )
        target, prev, valid, t = out

        # body+hand mode (use_hand, inherited from the parent): concat the frame-aligned hand
        # tokens -> (H, 128) = body(64) ++ hand(64).
        hand_present = True
        if self.use_hand:
            hand, hand_present = _load_hand_window(rec, t, self.horizon)
            target = np.concatenate([target, hand], axis=-1)

        # history dropout -> fight the copy-forward shortcut (same as the pi0.5 arm)
        if self._rng.random() < self.history_dropout:
            prev = np.zeros_like(prev)

        frames = self._read_clip(rec, t)
        # per-token validity -> (horizon, D) mask the action DiT multiplies into its loss.
        # Hand-less corpora (LeVERB) get zeroed hand columns: ActionDiT's loss is
        # sum(err*mask)/sum(mask), so those dims carry no loss and no gradient natively.
        action_mask = np.repeat(valid.astype(np.float32)[:, None], target.shape[-1], axis=1)
        if self.use_hand and not hand_present:
            action_mask[:, TOKEN_DIM:] = 0.0

        sample = {
            "image": frames,                          # list of (H, W, 3) uint8
            "lang": rec.instruction,
            "action": target.astype(np.float32),      # (horizon, 64|128) raw FSQ tokens (bh: ++hand)
            "action_mask": action_mask,               # (horizon, same D; hand cols 0 when no hands)
            "prev_tokens": prev.astype(np.float32),    # (history, 64) -- TODO 3 consumes this
            "episode_ref": rec.tokens_ref,
            "window_t": int(t),
        }
        if self.use_state:
            st = _load_state(rec, t)                   # (32,) raw q_dev+gravity (shared pi0.5 loader)
            if self._state_norm is not None:          # quantile -> [-1,1] (reuse pi0.5 state stats)
                q01, q99 = self._state_norm
                st = np.clip(2.0 * (st - q01) / np.maximum(q99 - q01, 1e-6) - 1.0, -1.0, 1.0)
            # (1, 32): DiT4DiT.forward expects per-example state as [1, state_dim] (batches to
            # [B, 1, state_dim]; its .repeat(r, 1, 1) needs the 3-D layout). ActionDiT's native
            # state_encoder consumes it as one state token prepended to the action sequence.
            sample["state"] = st.astype(np.float32)[None, :]
        return sample


def collate_fn(batch):
    """DiT4DiT's trainer expects a list of example dicts (no tensor stacking)."""
    return batch


def _corpora_from_env(weights=None, he_all_categories=False) -> list[CorpusSpec]:
    """The SONIC corpora (VLA + proprio sidecars), shared with the pi0.5 arm via default_corpora.
    `weights` = sampling mix (None -> default); an `egosuite`/`xperience` key opts that corpus in.

    0905: mark every robot corpus q_order="isaac" (the pi0.5 arm's fix_state_order): all proprio
    sidecars store q_dev in IsaacLab order, and without the remap the state was served scrambled
    vs the SONIC-grouped order deploy sends -- one of the root causes of the 0714 stand-still."""
    import dataclasses
    corpora = default_corpora(weights,
                              he_category_filter=None if he_all_categories else "Locomanip")
    return [dataclasses.replace(c, q_order="isaac") if c.q_order != "isaac" else c
            for c in corpora]


def get_sonic_vla_dataset(cfg, split_override=None, samples_per_epoch_override=None,
                          history_dropout_override=None) -> SonicVideoTokenDataset:
    """Build the dataset from a DiT4DiT OmegaConf `cfg`.

    horizon is taken from the action head (future_action_window_size + 1) so the token chunk
    matches what `DiT4DiT.forward` slices; data window params default to the pi0.5 arm's values.
    The *_override args let the trainer build a held-out eval loader (split="eval", no history
    dropout, a small fixed number of samples) from the same config.
    """
    d = cfg.datasets.vla_data
    am = cfg.framework.action_model
    horizon = int(am.future_action_window_size) + 1

    img = d.get("image_size", 224)          # OmegaConf ListConfig is not a python list -> index/try
    try:
        image_size = int(img[0])
    except (TypeError, KeyError, IndexError):
        image_size = int(img)

    kwargs = dict(
        horizon=horizon,
        history=int(d.get("history", 50)),
        history_stride=int(d.get("history_stride", 4)),
        history_dropout=float(history_dropout_override if history_dropout_override is not None
                              else d.get("history_dropout", 0.5)),
        min_valid_frac=float(d.get("min_valid_frac", 0.9)),
        image_size=image_size,
        load_images=bool(d.get("load_images", True)),
        samples_per_epoch=int(samples_per_epoch_override if samples_per_epoch_override is not None
                              else d.get("samples_per_epoch", 200_000)),
        seed=int(cfg.get("seed", 42)),
        split=str(split_override if split_override is not None else d.get("split", "train")),
        test_frac=float(d.get("test_frac", 0.12)),
        test_category=str(d.get("test_category", "Locomanip")),
        train_exclude_corpora=tuple(d.get("train_exclude_corpora", ()) or ()),
        use_hand=bool(d.get("use_hand", False)),
    )
    # 0905 bhs5-parity: linear-in-samples mix anneal (needs _maybe_anneal() in __getitem__).
    we = d.get("weights_end", None)
    if we:
        kwargs["weights_end"] = {str(k): float(v) for k, v in dict(we).items()}
        kwargs["mix_anneal_samples"] = int(d.get("mix_anneal_samples", 12_800_000))
    index_dir = os.environ.get("SONIC_INDEX_DIR")
    if index_dir:
        kwargs["cache_dir"] = index_dir

    w = d.get("weights", None)
    weights = {str(k): float(v) for k, v in dict(w).items()} if w else None
    return SonicVideoTokenDataset(
        _corpora_from_env(weights, he_all_categories=bool(d.get("he_all_categories", False))),
        video_delta_indices=list(d.get("video_delta_indices", (0, 12, 24, 36, 48))),
        action_video_freq_ratio=int(d.get("action_video_freq_ratio", 1)),
        use_state=bool(d.get("include_state", False)),
        state_stats_path=d.get("state_stats_path", None),
        **kwargs,
    )


def save_identity_statistics(out_path):
    """FSQ tokens use identity normalization -> write trivial stats so checkpoint loading
    (base_framework.read_mode_config) has a file to read. NOT used to (un)normalize tokens."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "sonic": {
            "action": {
                "mean": [0.0] * TOKEN_DIM, "std": [1.0] * TOKEN_DIM,
                "q01": [-1.0] * TOKEN_DIM, "q99": [1.0] * TOKEN_DIM,
                "mask": [False] * TOKEN_DIM,  # identity: unnormalize passes values through
            },
            "num_transitions": 0, "num_trajectories": 0,
        }
    }
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Smoke-test the DiT4DiT SONIC token dataset (CPU).")
    ap.add_argument("--no-images", action="store_true", help="skip video decode (token-path only)")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--split", type=str, default="all")
    args = ap.parse_args()

    ds = SonicVideoTokenDataset(
        _corpora_from_env(),
        horizon=args.horizon, split=args.split,
        load_images=not args.no_images, samples_per_epoch=args.n,
    )
    seen: dict[str, int] = {}
    for i in range(args.n):
        s = ds[i]
        c = s["episode_ref"].split("/")[5] if "/" in s["episode_ref"] else "?"
        a = np.asarray(s["action"]); m = np.asarray(s["action_mask"]); p = np.asarray(s["prev_tokens"])
        seen[c] = seen.get(c, 0) + 1
        print(f"  image=list[{len(s['image'])}] frame0{tuple(np.asarray(s['image'][0]).shape)} "
              f"action{tuple(a.shape)} mask{tuple(m.shape)} valid={m.mean():.0%} "
              f"prev{tuple(p.shape)} | '{s['lang'][:40]}'")
    print("done.")

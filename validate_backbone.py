#!/usr/bin/env python
"""TODO 0 + (partial) TODO 5: end-to-end backbone/head probe for the DiT4DiT-SONIC arm.

Loads the full DiT4DiT framework (Cosmos-Predict2.5 world-model backbone + 64-d flow-matching
action head) from dit4dit_g1_sonic.yaml, pulls a tiny real batch through the SONIC token
dataloader, and checks:
  - the backbone loads and produces extract_layer hidden states + accepts language,
  - joint forward returns finite `action_loss` AND `future_video_loss`,
  - predict_action returns a (B, 50, 64) SONIC token chunk.

Run on a GPU node with the dit4dit venv:
  /lustre/home/malbaba/humanoid-vla/dit4dit/.venv/bin/python validate_backbone.py
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from omegaconf import OmegaConf

from DiT4DiT.model.framework import build_framework
from DiT4DiT.dataloader.sonic_token_dataset import get_sonic_vla_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(_HERE / "DiT4DiT/config/real_robot/dit4dit_g1_sonic.yaml"))
    ap.add_argument("--base_model", default=None, help="override framework.cosmos25.base_model")
    ap.add_argument("--n", type=int, default=2, help="probe batch size")
    ap.add_argument("--split", default="eval", help="dataset split to sample the probe batch from")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = OmegaConf.load(args.config)
    if args.base_model:
        cfg.framework.cosmos25.base_model = args.base_model
    cfg.datasets.vla_data.split = args.split
    cfg.datasets.vla_data.samples_per_epoch = args.n
    cfg.output_dir = str(_HERE / "results" / "validate_backbone")  # for any stats write

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[probe] device={device}  base_model={cfg.framework.cosmos25.base_model}", flush=True)

    print("[probe] building DiT4DiT framework (loads Cosmos-Predict2.5 weights, ~minutes)...", flush=True)
    model = build_framework(cfg).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[probe] model built: {n_params/1e9:.2f}B params", flush=True)

    print(f"[probe] sampling {args.n} real examples (split={args.split})...", flush=True)
    ds = get_sonic_vla_dataset(cfg)
    examples = [ds[i] for i in range(args.n)]
    e0 = examples[0]
    print(f"[probe]   image=list[{len(e0['image'])}] frame0{tuple(np.asarray(e0['image'][0]).shape)} "
          f"action{tuple(np.asarray(e0['action']).shape)} mask{tuple(np.asarray(e0['action_mask']).shape)} "
          f"| '{e0['lang'][:50]}'", flush=True)

    # --- joint forward: action_loss + future_video_loss ---
    print("[probe] forward (joint: action + future-video FM)...", flush=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(examples)
    al = out.get("action_loss")
    vl = out.get("future_video_loss")
    al_v = float(al.item()) if torch.is_tensor(al) else None
    vl_v = float(vl.item()) if torch.is_tensor(vl) else None
    print(f"[probe]   action_loss={al_v}  future_video_loss={vl_v}", flush=True)
    assert al is not None and np.isfinite(al_v), f"action_loss bad: {al_v}"
    assert vl is not None and np.isfinite(vl_v), f"future_video_loss bad: {vl_v}"

    # --- inference path: predicted token chunk shape ---
    print("[probe] predict_action...", flush=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pa = model.predict_action(examples)["normalized_actions"]
    pa = np.asarray(pa)
    print(f"[probe]   predicted tokens shape={pa.shape} dtype={pa.dtype} "
          f"range=[{pa.min():.3f}, {pa.max():.3f}]", flush=True)
    assert pa.shape == (args.n, 50, 64), f"expected ({args.n}, 50, 64), got {pa.shape}"

    print("[probe] PASS: backbone loads, joint losses finite, head predicts (B, 50, 64).", flush=True)


if __name__ == "__main__":
    main()

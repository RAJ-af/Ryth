"""`ryth-train` command-line interface.

Har TrainConfig field ke liye ek flag auto-generate hota hai. Example:

    ryth-train --data_dir rds_out --model_preset ryth_30m --max_steps 2000 \
        --micro_batch_size 8 --grad_accum_steps 4 --dtype bf16
    ryth-train --data_dir rds_out --resume latest        # auto-resume
"""

from __future__ import annotations

import argparse
import dataclasses

from .config import TrainConfig
from .trainer import Trainer


def build_parser():
    ap = argparse.ArgumentParser(prog="ryth-train", description="Train a Ryth model.")
    for f in dataclasses.fields(TrainConfig):
        ftype = f.type
        if ftype in ("bool", bool):
            ap.add_argument(f"--{f.name}",
                            type=lambda x: str(x).lower() in ("1", "true", "yes"),
                            default=None)
        else:
            pytype = {"int": int, "float": float}.get(str(ftype), str)
            ap.add_argument(f"--{f.name}", type=pytype, default=None)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    cfg = TrainConfig(**overrides)
    print("[ryth-train] config:", cfg)
    Trainer(cfg).train()


if __name__ == "__main__":
    main()

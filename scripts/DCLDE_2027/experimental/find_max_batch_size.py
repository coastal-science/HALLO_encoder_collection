"""Binary-searches the largest classifier batch size that fits on GPU, using a
synthetic input of the real spectrogram shape -- memory use only depends on
tensor shapes, not actual data, so this avoids paying dataloader/epoch
overhead per trial. Builds the exact same ClassifierModel + optimizer the real
ClassifierTrainer (model_trainer/train.py) uses.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from encoder_pipeline.model_trainer.config import ClassifierConfig
from encoder_pipeline.model_trainer.models import ClassifierModel
from encoder_pipeline.model_trainer.train import build_optimizer

# 00_base.yaml's Perch frontend: 160 mel bins, hop 320 @ 32kHz, 5s window
# (2 * annotation.time_offset) -> 1 + 160000 // 320 = 501 frames. Recompute if
# those spectrogram params change.
SPEC_SHAPE = (160, 501)


def fits(batch_size: int, config: ClassifierConfig, num_classes: int) -> bool:
    """True if one forward/backward/optimizer step at this batch size doesn't OOM."""
    model = ClassifierModel(config, num_classes).to(config.device)
    optimizer = build_optimizer(config.optimizer, model.parameters(), config.lr, config.weight_decay, config.momentum)
    try:
        specs = torch.randn(batch_size, 1, *SPEC_SHAPE, device=config.device)
        labels = torch.randint(0, num_classes, (batch_size,), device=config.device)
        with torch.autocast(config.device, dtype=torch.bfloat16, enabled=config.amp):
            loss = nn.CrossEntropyLoss()(model(specs), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        del model, optimizer
        torch.cuda.empty_cache()


def find_max_batch_size(config: ClassifierConfig, num_classes: int, start: int) -> int:
    """Doubles from start until OOM, then bisects to the exact largest batch size that fits."""
    batch_size, last_good = start, 0
    while fits(batch_size, config, num_classes):
        print(f"  {batch_size}: OK")
        last_good = batch_size
        batch_size *= 2
    print(f"  {batch_size}: OOM")

    lo, hi = last_good, batch_size
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(mid, config, num_classes):
            print(f"  {mid}: OK")
            lo = mid
        else:
            print(f"  {mid}: OOM")
            hi = mid
    return lo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--num-classes", type=int, default=4, help="4 for the kpalmer_full scheme: Background/SRKW/TKW/HW.")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--start", type=int, default=64, help="First batch size to try; doubles from here.")
    args = parser.parse_args()

    config = ClassifierConfig(backbone_name=args.backbone, amp=args.amp)
    print(f"probing backbone={args.backbone} amp={args.amp} device={config.device} spec_shape={SPEC_SHAPE}")

    max_batch_size = find_max_batch_size(config, args.num_classes, args.start)
    print(f"max batch size that fits: {max_batch_size}")
    print(f"suggested (10% headroom for augment/dataloader overhead not modeled here): {int(max_batch_size * 0.9)}")


if __name__ == "__main__":
    main()

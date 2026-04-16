
#!/usr/bin/env python3
"""Fine-tune WiFi-Radar models on real-world CSI datasets stored as NPZ files."""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wifi_radar.models.encoder import DualBranchEncoder
from wifi_radar.models.pose_estimator import PoseEstimator
from wifi_radar.utils.model_io import load_checkpoint, save_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("transfer_learning")


class RealWorldCSIDataset(Dataset):
    def __init__(self, paths: List[str]) -> None:
        self._samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                amp = data["amplitude"].astype(np.float32)
                phase = data["phase"].astype(np.float32)
                keypoints = data["keypoints"].astype(np.float32)
                confidence = data["confidence"].astype(np.float32) if "confidence" in data else np.ones((len(keypoints), 17), dtype=np.float32)
                for a, p, k, c in zip(amp, phase, keypoints, confidence):
                    self._samples.append((a, p, k, c))
        if not self._samples:
            raise ValueError("No training samples found in the provided dataset files")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        a, p, k, c = self._samples[idx]
        return torch.from_numpy(a), torch.from_numpy(p), torch.from_numpy(k), torch.from_numpy(c)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer learning on real-world CSI datasets")
    parser.add_argument("dataset", nargs="+", help="One or more .npz files or globs containing amplitude/phase/keypoints arrays")
    parser.add_argument("--weights", default=os.path.join("weights", "simulation_baseline.pth"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--output", default=os.path.join("weights", "realworld_transfer.pth"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_paths: List[str] = []
    for pattern in args.dataset:
        matches = sorted(glob.glob(pattern))
        dataset_paths.extend(matches if matches else [pattern])

    dataset = RealWorldCSIDataset(dataset_paths)
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = DualBranchEncoder().to(device)
    pose_estimator = PoseEstimator().to(device)

    if args.weights and os.path.exists(args.weights):
        load_checkpoint(encoder, pose_estimator, args.weights, device=device)
        log.info("Loaded checkpoint from %s", args.weights)
    else:
        encoder.initialize_weights()

    best_val_loss = float("inf")
    os.makedirs(Path(args.output).parent, exist_ok=True)

    for epoch in range(args.epochs):
        freeze_encoder = epoch < args.freeze_encoder_epochs
        for p in encoder.parameters():
            p.requires_grad = not freeze_encoder

        param_groups = [
            {"params": pose_estimator.parameters(), "lr": args.lr_head},
        ]
        if not freeze_encoder:
            param_groups.append({"params": encoder.parameters(), "lr": args.lr_backbone})
        optimizer = torch.optim.Adam(param_groups, weight_decay=1e-5)

        encoder.train()
        pose_estimator.train()
        train_loss = 0.0
        for amp, phase, kp, conf in train_loader:
            amp, phase, kp, conf = amp.to(device).float(), phase.to(device).float(), kp.to(device).float(), conf.to(device).float()
            optimizer.zero_grad()
            features = encoder(amp, phase)
            pred_kp, pred_conf, _ = pose_estimator(features, hidden=None)
            loss = F.mse_loss(pred_kp, kp) + 0.1 * F.binary_cross_entropy(pred_conf, conf)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(pose_estimator.parameters()), 1.0)
            optimizer.step()
            train_loss += float(loss.item())

        encoder.eval()
        pose_estimator.eval()
        val_loss = 0.0
        with torch.no_grad():
            for amp, phase, kp, conf in val_loader:
                amp, phase, kp, conf = amp.to(device).float(), phase.to(device).float(), kp.to(device).float(), conf.to(device).float()
                features = encoder(amp, phase)
                pred_kp, pred_conf, _ = pose_estimator(features, hidden=None)
                loss = F.mse_loss(pred_kp, kp) + 0.1 * F.binary_cross_entropy(pred_conf, conf)
                val_loss += float(loss.item())

        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))
        log.info("epoch=%03d train_loss=%.4f val_loss=%.4f freeze_encoder=%s", epoch + 1, train_loss, val_loss, freeze_encoder)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                encoder,
                pose_estimator,
                args.output,
                metadata={"mode": "transfer_learning", "dataset_files": dataset_paths},
                epoch=epoch + 1,
                val_loss=val_loss,
            )
            log.info("Saved improved checkpoint to %s", args.output)


if __name__ == "__main__":
    main()

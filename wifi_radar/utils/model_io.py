"""
ID: WR-UTIL-IO-001
Purpose: Serialise and restore complete model checkpoints with version metadata,
         enabling reproducible training and safe deployment of simulation-baseline weights.
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_CHECKPOINT_VERSION = "1.0"


def save_checkpoint(
    encoder: nn.Module,
    pose_estimator: nn.Module,
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    val_loss: float = float("inf"),
) -> None:
    """Save encoder + pose_estimator weights with metadata.

    Args:
        encoder:        Trained DualBranchEncoder instance.
        pose_estimator: Trained PoseEstimator (or MultiPersonPoseEstimator) instance.
        path:           Destination file path (*.pth).
        metadata:       Arbitrary dict stored alongside the weights (e.g. training config).
        optimizer:      Optional optimizer state for checkpoint resumption.
        epoch:          Current training epoch.
        val_loss:       Best validation loss reached so far.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "checkpoint_version": _CHECKPOINT_VERSION,
        "epoch": epoch,
        "val_loss": val_loss,
        "encoder_state_dict": encoder.state_dict(),
        "pose_estimator_state_dict": pose_estimator.state_dict(),
        "encoder_config": _extract_config(encoder),
        "pose_estimator_config": _extract_config(pose_estimator),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)
    logger.info("Checkpoint saved → %s  (epoch=%d  val_loss=%.4f)", path, epoch, val_loss)


def load_checkpoint(
    encoder: nn.Module,
    pose_estimator: nn.Module,
    path: str,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load encoder + pose_estimator weights from a checkpoint.

    Args:
        encoder:        Model instance to populate (must match saved architecture).
        pose_estimator: Model instance to populate.
        path:           Path to the .pth checkpoint file.
        device:         Target device; defaults to CPU when None.
        strict:         Passed to ``load_state_dict`` — set False for partial loads.

    Returns:
        Dict with keys: epoch, val_loss, metadata, checkpoint_version.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        RuntimeError:      If the checkpoint version is incompatible.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=True)

    version = checkpoint.get("checkpoint_version", "unknown")
    if version != _CHECKPOINT_VERSION:
        logger.warning(
            "Checkpoint version mismatch: file=%s  current=%s — proceeding anyway",
            version,
            _CHECKPOINT_VERSION,
        )

    encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=strict)
    pose_estimator.load_state_dict(
        checkpoint["pose_estimator_state_dict"], strict=strict
    )

    encoder.to(device)
    pose_estimator.to(device)
    encoder.eval()
    pose_estimator.eval()

    logger.info(
        "Loaded checkpoint ← %s  (epoch=%d  val_loss=%.4f)",
        path,
        checkpoint.get("epoch", 0),
        checkpoint.get("val_loss", float("inf")),
    )
    return {
        "epoch": checkpoint.get("epoch", 0),
        "val_loss": checkpoint.get("val_loss", float("inf")),
        "metadata": checkpoint.get("metadata", {}),
        "checkpoint_version": version,
    }


def _extract_config(model: nn.Module) -> Dict[str, Any]:
    """Capture constructor parameters stored as instance attributes for provenance."""
    attrs = {}
    for key in ("num_tx", "num_rx", "num_subcarriers", "hidden_dim", "output_dim",
                "input_dim", "num_keypoints", "max_people"):
        val = getattr(model, key, None)
        if val is not None:
            attrs[key] = val
    return attrs

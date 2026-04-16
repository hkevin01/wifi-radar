"""
ID: WR-UTIL-IO-001
Requirement: Provide deterministic checkpoint serialisation and deserialisation
             with version tagging for all WiFi-Radar neural network models.
Purpose: Serialise and restore complete model checkpoints with version metadata,
         enabling reproducible training and safe deployment of simulation-baseline weights.
Rationale: Centralised checkpoint I/O decouples model architecture from storage format,
           allows version detection at load time, and ensures provenance metadata
           (training config, epoch, val_loss) travels with the weights.
Assumptions: PyTorch >= 1.10 serialisation API; checkpoint format version "1.0".
Constraints: Files written via torch.save (pickle); typically 50-200 MB per model pair.
References: torch.save / torch.load documentation; _CHECKPOINT_VERSION constant.
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

    ID: WR-UTIL-IO-SAVE-001
    Requirement: Write a single versioned .pth file containing both model weight
                 dicts, training metadata, and optional optimiser state.
    Purpose: Persist training progress for experiment resumption and reproducible
             deployment without requiring separate encoder/pose-estimator files.
    Rationale: Bundling both models prevents synchronisation errors from separate
               saves; version tagging enables compatibility checks at load time.
    Inputs:
        encoder        — nn.Module (DualBranchEncoder), must have state_dict().
        pose_estimator — nn.Module (PoseEstimator or MultiPersonPoseEstimator).
        path           — str: destination .pth file path.
        metadata       — Optional[Dict[str,Any]]: arbitrary training-config dict.
        optimizer      — Optional[Optimizer]: optimizer state; None to omit.
        epoch          — int >= 0: current training epoch index.
        val_loss       — float: best validation loss reached so far.
    Outputs:
        None — writes to disk as a side effect.
    Preconditions:
        Both models must have callable state_dict() methods.
        Path parent directory must be writable (will be created if absent).
    Postconditions:
        A .pth file exists at ``path`` with ``checkpoint_version`` == "1.0".
    Assumptions:
        Sufficient disk space is available for the serialised state dicts.
        Model layers are picklable (no lambdas or open file handles).
    Side Effects:
        Creates parent directories (parents=True, exist_ok=True).
        Writes or overwrites the .pth file at ``path``.
        Emits an INFO log message with path, epoch, and val_loss.
    Failure Modes:
        OSError if path is not writable — propagates to caller.
        PicklingError if model contains non-serialisable components.
    Error Handling:
        No internal try/except; callers must handle OSError / IOError.
    Constraints:
        File size scales with model parameter count (~50-200 MB typical).
    Verification:
        Round-trip test: save then load; assert epoch, val_loss, and a
        sample weight tensor are numerically identical.
    References:
        torch.save docs; _CHECKPOINT_VERSION = "1.0" in this module.
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

    ID: WR-UTIL-IO-LOAD-001
    Requirement: Restore model weights from a versioned .pth file and return
                 training metadata (epoch, val_loss, config).
    Purpose: Enable experiment resumption and deployment of pre-trained models
             from saved checkpoints without re-training from scratch.
    Rationale: Centralised load logic handles version checks, device mapping,
               and eval-mode activation so callers avoid repetitive boilerplate.
    Inputs:
        encoder        — nn.Module: instance to populate (architecture must match).
        pose_estimator — nn.Module: instance to populate.
        path           — str: path to the .pth checkpoint file.
        device         — Optional[torch.device]: target device; defaults to CPU.
        strict         — bool: passed to load_state_dict; False allows partial loads.
    Outputs:
        Dict with keys: epoch (int), val_loss (float), metadata (dict),
        checkpoint_version (str).
    Preconditions:
        ``path`` must exist and be a valid torch checkpoint file.
        Both model architectures must match the saved state dicts when strict=True.
    Postconditions:
        Both models' parameters are populated from the checkpoint.
        Both models are moved to ``device`` and set to eval() mode.
    Assumptions:
        torch.load with weights_only=True is available (PyTorch >= 2.0).
        Checkpoint was created by save_checkpoint() in this module.
    Side Effects:
        Modifies model parameter tensors in-place.
        Moves models to ``device`` and calls .eval() on both.
        Emits an INFO log message with path, epoch, and val_loss.
    Failure Modes:
        FileNotFoundError if ``path`` does not exist — raised immediately.
        RuntimeError from load_state_dict on architecture mismatch (strict=True).
        Version mismatch logs a WARNING but does not raise.
    Error Handling:
        FileNotFoundError raised explicitly before torch.load is called.
        Version mismatch: logged as WARNING; loading proceeds regardless.
    Constraints:
        Loading large checkpoints (~200 MB) may be slow on network filesystems.
    Verification:
        Test: save checkpoint, load into new instances, compare forward-pass
        outputs on identical inputs to floating-point precision.
    References:
        torch.load(weights_only=True); _CHECKPOINT_VERSION = "1.0".
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
    """Capture constructor parameters stored as instance attributes for provenance.

    ID: WR-UTIL-IO-CFG-001
    Requirement: Extract known hyperparameter attributes from a model instance
                 into a plain dict suitable for JSON/YAML serialisation.
    Purpose: Record model architecture parameters alongside weights so that a
             checkpoint is self-describing and the model can be reconstructed
             without the original training script.
    Rationale: Inspecting instance attributes is more reliable than parsing the
               class signature, as subclasses may store extra parameters.
    Inputs:
        model — nn.Module: any model instance (DualBranchEncoder, PoseEstimator, etc.).
    Outputs:
        Dict[str, Any]: known attribute names -> values for non-None attributes.
        Empty dict if none of the known attributes are set on the model.
    Preconditions:
        ``model`` must be a valid Python object with a __dict__.
    Postconditions:
        Returned dict contains only attributes present on the model with non-None values.
    Assumptions:
        Relevant hyperparameters are stored directly on self (not in sub-modules).
    Side Effects:
        None — read-only attribute inspection.
    Failure Modes:
        If getattr raises (unusual), the attribute is silently omitted via default=None.
    Error Handling:
        Uses getattr(model, key, None); missing attributes are silently skipped.
    Constraints:
        Only inspects the fixed known attribute list; add new names explicitly here.
    Verification:
        Unit test: construct DualBranchEncoder(num_tx=2); assert _extract_config
        returns a dict containing {'num_tx': 2, 'num_rx': 3, ...}.
    References:
        Python getattr() built-in; hyperparameter naming convention in this codebase.
    """
    attrs = {}
    for key in ("num_tx", "num_rx", "num_subcarriers", "hidden_dim", "output_dim",
                "input_dim", "num_keypoints", "max_people"):
        val = getattr(model, key, None)
        if val is not None:
            attrs[key] = val
    return attrs

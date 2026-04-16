#!/usr/bin/env python3
"""
ID: WR-SCRIPT-ONNX-001
Purpose: Export DualBranchEncoder and PoseEstimator to ONNX format for
         edge deployment (Jetson Nano, Raspberry Pi 4, etc.) and validate
         outputs match the PyTorch reference implementation.

Exported files:
    weights/encoder.onnx         — CSI amplitude + phase → 256-d feature vector
    weights/pose_estimator.onnx  — 256-d features → 17 keypoints + confidence

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --weights weights/simulation_baseline.pth
    python scripts/export_onnx.py --opset 18 --output-dir deploy/onnx
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wifi_radar.models.encoder import DualBranchEncoder
from wifi_radar.models.pose_estimator import PoseEstimator
from wifi_radar.utils.model_io import load_checkpoint

log = logging.getLogger("export_onnx")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ─────────────────────────────────────────────────────────────────────────── #
# Wrapper: strip LSTM hidden state so ONNX sees simple I/O                    #
# ─────────────────────────────────────────────────────────────────────────── #

class _EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder: DualBranchEncoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return self.encoder(amplitude, phase)


class _PoseEstimatorWrapper(torch.nn.Module):
    """Single-frame (no sequence) pose estimator — hides the LSTM hidden state."""
    def __init__(self, pose_estimator: PoseEstimator) -> None:
        super().__init__()
        self.pe = pose_estimator

    def forward(self, features: torch.Tensor):
        keypoints, confidence, _ = self.pe(features, hidden=None)
        return keypoints, confidence


# ─────────────────────────────────────────────────────────────────────────── #
# Export helpers                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

def export_encoder(
    encoder: DualBranchEncoder,
    output_path: str,
    opset: int,
    batch: int = 1,
) -> None:
    """Export DualBranchEncoder to ONNX format.

    ONNX export rationale:
        - ``export_params=True``: Embed trained weights in the .onnx file so the
          runtime does not need a separate parameter file.
        - ``do_constant_folding=True``: Pre-compute static sub-expressions at
          export time (e.g. BatchNorm fused into Conv weights) to reduce runtime
          inference latency.
        - ``dynamic_axes``: Only the batch dimension is dynamic; spatial and
          channel dimensions are fixed at (3, 3, 64) as documented in the model.
          Making the batch axis dynamic allows the runtime to serve arbitrary
          batch sizes without re-exporting.
        - Opset 17 (default): First opset with ``AdaptiveAveragePool`` support
          required by DualBranchEncoder.  Pass ``--opset 18`` to target newer runtimes.

    Args:
        encoder:     Trained (or random-initialised) DualBranchEncoder in eval mode.
        output_path: Destination file path (e.g. ``weights/encoder.onnx``).
        opset:       ONNX opset version (≥ 17 required for AdaptiveAvgPool).
        batch:       Batch size for the dummy input used during tracing.

    Side Effects:
        Writes an ONNX model file to ``output_path``.
        Logs an info message on success.
    """
    model.eval()

    dummy_amp   = torch.randn(batch, 3, 3, 64)
    dummy_phase = torch.randn(batch, 3, 3, 64)

    torch.onnx.export(
        model,
        (dummy_amp, dummy_phase),
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["amplitude", "phase"],
        output_names=["features"],
        dynamic_axes={
            "amplitude": {0: "batch_size"},
            "phase":     {0: "batch_size"},
            "features":  {0: "batch_size"},
        },
    )
    log.info("Encoder exported → %s", output_path)


def export_pose_estimator(
    pose_estimator: PoseEstimator,
    output_path: str,
    opset: int,
    batch: int = 1,
) -> None:
    """Export PoseEstimator (single-frame, no LSTM hidden state) to ONNX.

    ONNX export rationale:
        - ``_PoseEstimatorWrapper`` strips the LSTM hidden-state tuple from
          the return value because ONNX does not support optional tuple outputs
          natively.  The wrapper returns only (keypoints, confidence), which
          map cleanly to two named ONNX output tensors.
        - ``do_constant_folding=True``: Fuses the sigmoid activations in the
          confidence head where possible.
        - The feature input axis is the only dynamic axis; keypoint count and
          coordinate dimension are always (17, 3) at inference time.

    Args:
        pose_estimator: Trained PoseEstimator in eval mode.
        output_path:    Destination file path (e.g. ``weights/pose_estimator.onnx``).
        opset:          ONNX opset version.
        batch:          Batch size for the dummy input tensor.

    Side Effects:
        Writes an ONNX model file to ``output_path``.
        Logs an info message on success.
    """
    model.eval()

    dummy_features = torch.randn(batch, 256)

    torch.onnx.export(
        model,
        dummy_features,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["features"],
        output_names=["keypoints", "confidence"],
        dynamic_axes={
            "features":   {0: "batch_size"},
            "keypoints":  {0: "batch_size"},
            "confidence": {0: "batch_size"},
        },
    )
    log.info("PoseEstimator exported → %s", output_path)


def validate_with_onnxruntime(
    encoder:          DualBranchEncoder,
    pose_estimator:   PoseEstimator,
    encoder_path:     str,
    pose_est_path:    str,
) -> None:
    """Run a forward pass in PyTorch and ONNX Runtime and compare outputs."""
    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime not installed — skipping validation.  pip install onnxruntime")
        return

    try:
        import onnx
        for p in (encoder_path, pose_est_path):
            m = onnx.load(p)
            onnx.checker.check_model(m)
            log.info("ONNX model check passed: %s", p)
    except ImportError:
        log.warning("onnx not installed — skipping model check.  pip install onnx")

    sess_enc  = ort.InferenceSession(encoder_path,  providers=["CPUExecutionProvider"])
    sess_pose = ort.InferenceSession(pose_est_path, providers=["CPUExecutionProvider"])

    batch = 4
    amp_np   = np.random.randn(batch, 3, 3, 64).astype(np.float32)
    phase_np = np.random.randn(batch, 3, 3, 64).astype(np.float32)
    amp_t    = torch.from_numpy(amp_np)
    phase_t  = torch.from_numpy(phase_np)

    # PyTorch reference
    with torch.no_grad():
        feat_pt    = encoder(amp_t, phase_t)
        kp_pt, cf_pt, _ = pose_estimator(feat_pt, hidden=None)
        feat_np_ref = feat_pt.numpy()
        kp_np_ref   = kp_pt.numpy()

    # ONNX Runtime encoder
    feat_ort = sess_enc.run(["features"], {"amplitude": amp_np, "phase": phase_np})[0]
    max_enc_diff = float(np.abs(feat_ort - feat_np_ref).max())
    log.info("Encoder max output diff (PyTorch vs ORT): %.2e", max_enc_diff)

    # ONNX Runtime pose estimator
    kp_ort, cf_ort = sess_pose.run(
        ["keypoints", "confidence"], {"features": feat_ort}
    )
    max_pose_diff = float(np.abs(kp_ort - kp_np_ref).max())
    log.info("Pose estimator max output diff (PyTorch vs ORT): %.2e", max_pose_diff)

    assert max_enc_diff  < 1e-4, f"Encoder mismatch too large: {max_enc_diff}"
    assert max_pose_diff < 1e-4, f"PoseEstimator mismatch too large: {max_pose_diff}"
    log.info("Validation passed ✓")


# ─────────────────────────────────────────────────────────────────────────── #
# CLI                                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export WiFi-Radar models to ONNX")
    p.add_argument("--weights",    default=None,         help="Path to .pth checkpoint (optional)")
    p.add_argument("--opset",      type=int, default=17, help="ONNX opset version (default 17)")
    p.add_argument("--output-dir", default="weights",    help="Directory for .onnx files")
    p.add_argument("--validate",   action="store_true",  default=True,
                   help="Validate outputs with onnxruntime after export")
    p.add_argument("--no-validate", dest="validate", action="store_false")
    return p.parse_args()


def main() -> None:
    """Export WiFi-Radar models to ONNX and optionally validate against OnnxRuntime.

    Steps:
        1. Parse CLI arguments.
        2. Instantiate DualBranchEncoder and PoseEstimator on CPU.
        3. Load a .pth checkpoint if ``--weights`` is provided; otherwise export
           with random weights (useful for graph-structure validation only).
        4. Set both models to eval mode (disables BatchNorm training stats and Dropout).
        5. Call ``export_encoder()`` and ``export_pose_estimator()``.
        6. Optionally run ``validate_with_onnxruntime()`` to confirm that ONNX
           Runtime output matches the PyTorch reference within 1e-4 tolerance.

    Side Effects:
        Creates ``args.output_dir`` if needed.
        Writes two .onnx files to ``args.output_dir``.
        Logs paths of the generated files on completion.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    device  = torch.device("cpu")
    encoder = DualBranchEncoder().to(device)
    pose_est = PoseEstimator().to(device)

    if args.weights and os.path.exists(args.weights):
        load_checkpoint(encoder, pose_est, args.weights, device=device)
        log.info("Loaded weights from %s", args.weights)
    else:
        log.warning("No weights file provided — exporting random-initialised models.")
        encoder.initialize_weights()

    encoder.eval()
    pose_est.eval()

    enc_path  = os.path.join(args.output_dir, "encoder.onnx")
    pose_path = os.path.join(args.output_dir, "pose_estimator.onnx")

    export_encoder(encoder, enc_path, opset=args.opset)
    export_pose_estimator(pose_est, pose_path, opset=args.opset)

    if args.validate:
        validate_with_onnxruntime(encoder, pose_est, enc_path, pose_path)

    log.info("Done.  ONNX models in: %s/", args.output_dir)
    log.info("  encoder:        %s", enc_path)
    log.info("  pose_estimator: %s", pose_path)


if __name__ == "__main__":
    main()

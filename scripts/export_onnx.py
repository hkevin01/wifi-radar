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
    """Thin wrapper that exposes DualBranchEncoder as a simple 2-input ONNX graph.

    ID: WR-SCRIPT-ONNX-ENCWRAP-001
    Requirement: Wrap DualBranchEncoder so torch.onnx.export sees exactly two
                 named inputs (amplitude, phase) and one output (features).
    Purpose: Prevent ONNX trace errors caused by tuple inputs or internal state;
             provides a clean interface for edge runtime inference.
    Rationale: A wrapper module isolates torch.onnx.export from optional hidden
               states or extra return values the encoder might add in future.
    Inputs:
        encoder — DualBranchEncoder: trained encoder instance.
    Outputs:
        None — wraps encoder in self.encoder for forward dispatch.
    Preconditions:
        DualBranchEncoder must be importable.
    Postconditions:
        self.encoder holds a reference to the passed encoder instance.
    Assumptions:
        DualBranchEncoder.forward(amplitude, phase) returns a single Tensor.
    Side Effects:
        None.
    Failure Modes:
        None.
    Error Handling:
        None.
    Constraints:
        None.
    Verification:
        Unit test: instantiate wrapper; assert wrapper.encoder is encoder.
    References:
        DualBranchEncoder; WR-SCRIPT-ONNX-001.
    """

    def __init__(self, encoder: DualBranchEncoder) -> None:
        """Initialise the encoder wrapper.

        ID: WR-SCRIPT-ONNX-ENCWRAP-INIT-001
        Requirement: Call super().__init__() and store encoder as self.encoder.
        Purpose: Register the encoder as a sub-module so its parameters are
                 included in the ONNX export.
        Rationale: torch.nn.Module sub-module registration ensures parameter
                   export by torch.onnx.export with export_params=True.
        Inputs:
            encoder — DualBranchEncoder: trained encoder.
        Outputs:
            None.
        Preconditions:
            None.
        Postconditions:
            self.encoder is set.
        Assumptions: None.
        Side Effects: None.
        Failure Modes: None.
        Error Handling: None.
        Constraints: None.
        Verification: assert _EncoderWrapper(enc).encoder is enc.
        References: torch.nn.Module.__init__; WR-SCRIPT-ONNX-ENCWRAP-001.
        """
        super().__init__()
        self.encoder = encoder

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Run the encoder forward pass.

        ID: WR-SCRIPT-ONNX-ENCWRAP-FWD-001
        Requirement: Delegate to self.encoder(amplitude, phase) and return features.
        Purpose: Provide a clean 2-input ONNX graph node for the encoder.
        Rationale: Direct delegation keeps the wrapper transparent.
        Inputs:
            amplitude — Tensor (B,3,3,64) float32.
            phase     — Tensor (B,3,3,64) float32.
        Outputs:
            features — Tensor (B,256) float32.
        Preconditions:
            self.encoder must be in eval mode before ONNX export.
        Postconditions:
            Output shape (B,256).
        Assumptions: None.
        Side Effects: None.
        Failure Modes: None.
        Error Handling: None.
        Constraints: None.
        Verification: assert wrapper(amp, phase).shape == (B, 256).
        References: DualBranchEncoder.forward; WR-SCRIPT-ONNX-ENCWRAP-001.
        """
        return self.encoder(amplitude, phase)


class _PoseEstimatorWrapper(torch.nn.Module):
    """Single-frame (no sequence) pose estimator — hides the LSTM hidden state.

    ID: WR-SCRIPT-ONNX-POSEWRAP-001
    Requirement: Wrap PoseEstimator so torch.onnx.export sees one input (features)
                 and two outputs (keypoints, confidence), with no hidden state.
    Purpose: Strip the LSTM hidden state tuple from the return value because
             ONNX does not support optional tuple outputs natively.
    Rationale: Returning only (keypoints, confidence) maps cleanly to two named
               ONNX output tensors without needing sequence-level state.
    Inputs:
        pose_estimator — PoseEstimator: trained estimator instance.
    Outputs:
        None — wraps estimator in self.pe.
    Preconditions:
        PoseEstimator must be importable.
    Postconditions:
        self.pe holds a reference to the passed estimator instance.
    Assumptions:
        PoseEstimator.forward(features, hidden=None) returns (kp, conf, hidden).
    Side Effects:
        None.
    Failure Modes:
        None.
    Error Handling:
        None.
    Constraints:
        None.
    Verification:
        Unit test: instantiate wrapper; assert wrapper.pe is pose_estimator.
    References:
        PoseEstimator; WR-SCRIPT-ONNX-001.
    """

    def __init__(self, pose_estimator: PoseEstimator) -> None:
        """Initialise the pose estimator wrapper.

        ID: WR-SCRIPT-ONNX-POSEWRAP-INIT-001
        Requirement: Call super().__init__() and store pose_estimator as self.pe.
        Purpose: Register the pose estimator as a sub-module for ONNX parameter export.
        Rationale: torch.nn.Module sub-module registration ensures parameter inclusion.
        Inputs:
            pose_estimator — PoseEstimator: trained estimator.
        Outputs: None.
        Preconditions: None.
        Postconditions: self.pe is set.
        Assumptions: None.
        Side Effects: None.
        Failure Modes: None.
        Error Handling: None.
        Constraints: None.
        Verification: assert _PoseEstimatorWrapper(pe).pe is pe.
        References: torch.nn.Module.__init__; WR-SCRIPT-ONNX-POSEWRAP-001.
        """
        super().__init__()
        self.pe = pose_estimator

    def forward(self, features: torch.Tensor):
        """Run the pose estimator forward pass, discarding hidden state.

        ID: WR-SCRIPT-ONNX-POSEWRAP-FWD-001
        Requirement: Delegate to self.pe(features, hidden=None) and return only
                     (keypoints, confidence).
        Purpose: Provide a clean single-input ONNX graph node.
        Rationale: Discarding the hidden state tuple makes the ONNX graph
                   exportable without sequence-level state management.
        Inputs:
            features — Tensor (B,256) float32.
        Outputs:
            keypoints   — Tensor (B,17,3) float32.
            confidence  — Tensor (B,17) float32.
        Preconditions:
            self.pe must be in eval mode before ONNX export.
        Postconditions:
            keypoints.shape == (B,17,3); confidence.shape == (B,17).
        Assumptions: None.
        Side Effects: None.
        Failure Modes: None.
        Error Handling: None.
        Constraints: None.
        Verification: assert wrapper(feat)[0].shape == (B,17,3).
        References: PoseEstimator.forward; WR-SCRIPT-ONNX-POSEWRAP-001.
        """
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

    ID: WR-SCRIPT-ONNX-EXPENC-001
    Requirement: Trace DualBranchEncoder with a dummy (B,3,3,64) input pair and
                 write an ONNX model file to output_path.
    Purpose: Produce a portable ONNX graph for edge deployment without a PyTorch
             runtime dependency.
    Rationale: export_params=True embeds weights; do_constant_folding=True fuses
               BatchNorm into Conv to reduce runtime latency.
    Inputs:
        encoder     — DualBranchEncoder: trained model in eval mode.
        output_path — str: destination .onnx file path.
        opset       — int >=17: ONNX opset version.
        batch       — int: batch size for dummy input.
    Outputs:
        None — writes ONNX file to output_path.
    Preconditions:
        encoder.eval() must have been called.
        torch.onnx must be available.
    Postconditions:
        output_path file exists and is a valid ONNX model.
    Assumptions:
        Input spatial dims are always (3,3,64); only batch dim is dynamic.
    Side Effects:
        Writes to filesystem.
        Logs an INFO message on success.
    Failure Modes:
        Opset < 17: AdaptiveAvgPool unsupported; export raises RuntimeError.
        output_path unwritable: raises IOError.
    Error Handling:
        None; exceptions propagate to caller.
    Constraints:
        opset >= 17 required for AdaptiveAveragePool.
    Verification:
        Integration test: export and load with onnx.load; check_model passes.
    References:
        torch.onnx.export; WR-SCRIPT-ONNX-001.
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

    ID: WR-SCRIPT-ONNX-EXPPOSE-001
    Requirement: Trace _PoseEstimatorWrapper with a dummy (B,256) feature input
                 and write an ONNX model file to output_path.
    Purpose: Produce an ONNX pose estimator graph suitable for edge deployment.
    Rationale: _PoseEstimatorWrapper strips the LSTM hidden state so the ONNX
               graph has a clean (features) -> (keypoints, confidence) signature.
    Inputs:
        pose_estimator — PoseEstimator: trained model in eval mode.
        output_path    — str: destination .onnx file path.
        opset          — int: ONNX opset version.
        batch          — int: batch size for dummy input.
    Outputs:
        None — writes ONNX file to output_path.
    Preconditions:
        pose_estimator.eval() must have been called.
    Postconditions:
        output_path file exists and is a valid ONNX model.
    Assumptions:
        Feature vector is always 256-d; only batch dim is dynamic.
    Side Effects:
        Writes to filesystem.
        Logs an INFO message on success.
    Failure Modes:
        output_path unwritable: raises IOError.
    Error Handling:
        None; exceptions propagate to caller.
    Constraints:
        None.
    Verification:
        Integration test: export and load with onnx.load; check_model passes.
    References:
        _PoseEstimatorWrapper; torch.onnx.export; WR-SCRIPT-ONNX-001.
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
    """Run a forward pass in PyTorch and ONNX Runtime and compare outputs.

    ID: WR-SCRIPT-ONNX-VALIDATE-001
    Requirement: Load the exported ONNX files with onnxruntime, run inference
                 on the same random input as the PyTorch reference, and assert
                 that max absolute difference < 1e-4 for both models.
    Purpose: Confirm that the ONNX export is numerically equivalent to the
             PyTorch models before deploying to edge hardware.
    Rationale: Float32 numerics may differ slightly between PyTorch and ORT
               due to different kernel implementations; 1e-4 tolerance is
               conservative enough to catch export bugs without false positives.
    Inputs:
        encoder        — DualBranchEncoder: PyTorch reference model in eval mode.
        pose_estimator — PoseEstimator: PyTorch reference model in eval mode.
        encoder_path   — str: path to exported encoder.onnx.
        pose_est_path  — str: path to exported pose_estimator.onnx.
    Outputs:
        None — logs results; raises AssertionError on mismatch.
    Preconditions:
        Both ONNX files must exist.
        onnxruntime must be installed.
    Postconditions:
        Both max diffs are < 1e-4 or AssertionError is raised.
    Assumptions:
        CPUExecutionProvider is available in onnxruntime.
    Side Effects:
        Logs INFO messages with max diffs.
        Logs WARNING if onnxruntime or onnx not installed.
    Failure Modes:
        onnxruntime not installed: logs warning; returns early.
        onnx not installed: skips model check; continues.
        Max diff >= 1e-4: AssertionError.
    Error Handling:
        ImportError caught for onnxruntime and onnx; warning logged.
    Constraints:
        Uses batch size 4 for the validation forward pass.
    Verification:
        Integration test: run after export; assert no AssertionError.
    References:
        onnxruntime.InferenceSession; onnx.checker.check_model; WR-SCRIPT-ONNX-001.
    """
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
    """Parse CLI arguments for the ONNX export script.

    ID: WR-SCRIPT-ONNX-PARSEARGS-001
    Requirement: Define and parse --weights, --opset, --output-dir, --validate,
                 and --no-validate arguments; return a Namespace with defaults.
    Purpose: Allow operators to control output location, opset version, and
             validation without modifying source code.
    Rationale: argparse provides automatic --help, type coercion, and defaults.
    Inputs:
        sys.argv — CLI arguments.
    Outputs:
        argparse.Namespace with: weights, opset, output_dir, validate.
    Preconditions:
        Called before main().
    Postconditions:
        All argument fields populated.
    Assumptions:
        Default opset=17 is compatible with current DualBranchEncoder.
    Side Effects:
        May call sys.exit(2) on invalid arguments.
    Failure Modes:
        Invalid argument: argparse raises SystemExit.
    Error Handling:
        Handled by argparse.
    Constraints:
        --validate and --no-validate are mutually exclusive via dest='validate'.
    Verification:
        Unit test: call with []; assert args.opset == 17.
    References:
        argparse.ArgumentParser; WR-SCRIPT-ONNX-001.
    """
    p.add_argument("--weights",    default=None,         help="Path to .pth checkpoint (optional)")
    p.add_argument("--opset",      type=int, default=17, help="ONNX opset version (default 17)")
    p.add_argument("--output-dir", default="weights",    help="Directory for .onnx files")
    p.add_argument("--validate",   action="store_true",  default=True,
                   help="Validate outputs with onnxruntime after export")
    p.add_argument("--no-validate", dest="validate", action="store_false")
    return p.parse_args()


def main() -> None:
    """Export WiFi-Radar models to ONNX and optionally validate against OnnxRuntime.

    ID: WR-SCRIPT-ONNX-MAIN-001
    Requirement: Parse CLI args, instantiate models, optionally load weights,
                 export both models to ONNX, and optionally validate with ORT.
    Purpose: Serve as the single runnable entry point for the ONNX export
             workflow so all steps execute in the correct order.
    Rationale: Combining export and validation in one command reduces operator
               error compared to separate export/validate scripts.
    Inputs:
        sys.argv — via parse_args().
        Filesystem: optional .pth checkpoint.
    Outputs:
        Writes encoder.onnx and pose_estimator.onnx to args.output_dir.
    Preconditions:
        None — entry point.
    Postconditions:
        Two .onnx files exist in args.output_dir.
    Assumptions:
        If --weights is not provided, random-initialised models are exported
        (useful for graph-structure validation only).
    Side Effects:
        Creates args.output_dir if needed.
        Writes two .onnx files.
        Logs file paths on completion.
    Failure Modes:
        Missing onnxruntime: validation skipped with a warning.
        ONNX opset < 17: export_encoder raises RuntimeError.
    Error Handling:
        Propagates exceptions from export helpers to caller.
    Constraints:
        Both models are exported on CPU regardless of CUDA availability.
    Verification:
        Integration test: run script; assert both .onnx files exist.
    References:
        export_encoder; export_pose_estimator; validate_with_onnxruntime;
        parse_args; WR-SCRIPT-ONNX-001.
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

"""
ID: WR-UTIL-LIVECAP-001
Requirement: Validate recorded CSI capture files from live hardware or replay
             sessions and summarise their suitability for downstream inference.
Purpose: Provide an offline validation path for real-world CSI recordings so
         dataset quality can be assessed before deployment or transfer learning.
Rationale: Early validation of capture shape, finiteness, dynamic range, and DSP
           compatibility catches ingestion issues before runtime deployment.
Assumptions: Capture files are stored as .npz or .npy and contain amplitude/phase
             arrays or a complex CSI tensor.
Constraints: Validation is non-destructive and CPU-only by default.
References: numpy.load; SignalProcessor.process.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import numpy as np

from wifi_radar.processing.signal_processor import SignalProcessor


def load_capture_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load amplitude and phase tensors from a capture file.

    ID: WR-UTIL-LIVECAP-LOAD-001
    Requirement: Read a supported capture file and return amplitude/phase tensors.
    Purpose: Provide one canonical loader for replay, validation, and offline analysis.
    Rationale: Centralising format handling prevents drift between script and runtime paths.
    Inputs:
        path — str: file path to a .npz or .npy capture.
    Outputs:
        Tuple[np.ndarray, np.ndarray] containing amplitude and phase arrays.
    Preconditions:
        The file must exist and use one of the supported payload layouts.
    Postconditions:
        Returned arrays are float32 and aligned by frame index.
    Assumptions:
        Capture files contain amplitude/phase or complex CSI tensors.
    Side Effects:
        Reads the filesystem.
    Failure Modes:
        Missing file or unsupported payload layout raises an exception.
    Error Handling:
        FileNotFoundError and ValueError are raised to the caller.
    Constraints:
        Loading is eager in memory for simple offline validation use.
    Verification:
        Unit test: save a temp .npz with amplitude/phase and confirm the round-trip.
    References:
        numpy.load; numpy.angle; numpy.abs.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.endswith(".npz"):
        with np.load(path, allow_pickle=True) as data:
            if "amplitude" in data and "phase" in data:
                amplitude = np.asarray(data["amplitude"], dtype=np.float32)
                phase = np.asarray(data["phase"], dtype=np.float32)
            elif "csi" in data:
                csi = np.asarray(data["csi"], dtype=np.complex64)
                amplitude = np.abs(csi).astype(np.float32)
                phase = np.angle(csi).astype(np.float32)
            else:
                raise ValueError("Unsupported .npz format: expected amplitude/phase or csi")
        return amplitude, phase

    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype != object:
        if np.iscomplexobj(obj):
            return np.abs(obj).astype(np.float32), np.angle(obj).astype(np.float32)
        raise ValueError("Unsupported .npy array format: expected complex CSI or object container")

    payload = obj.item() if hasattr(obj, "item") else obj
    if isinstance(payload, dict) and "amplitude" in payload and "phase" in payload:
        return np.asarray(payload["amplitude"], dtype=np.float32), np.asarray(payload["phase"], dtype=np.float32)
    if isinstance(payload, (tuple, list)) and len(payload) == 2:
        return np.asarray(payload[0], dtype=np.float32), np.asarray(payload[1], dtype=np.float32)
    raise ValueError("Unsupported capture payload")


def validate_capture_arrays(amplitude: np.ndarray, phase: np.ndarray, max_frames: int | None = 16) -> Dict[str, Any]:
    """Validate the numerical health of a CSI capture tensor pair.

    ID: WR-UTIL-LIVECAP-VALIDATE-001
    Requirement: Inspect a capture tensor pair for shape integrity, finite values,
                 dynamic range, and compatibility with the signal-processing pipeline.
    Purpose: Provide an inexpensive offline gate before using a capture for demos,
             replay testing, or transfer learning.
    Rationale: Numerical validation catches malformed recordings early and reduces
               downstream debugging time in the inference stack.
    Inputs:
        amplitude — np.ndarray: capture amplitude tensor with frame axis.
        phase     — np.ndarray: capture phase tensor with frame axis.
        max_frames — Optional[int]: cap on how many frames are processed for validation.
    Outputs:
        Dict[str, Any] summary of the capture quality and key statistics.
    Preconditions:
        amplitude and phase share the same 4-D shape.
    Postconditions:
        Returned summary contains frame count, shape, finiteness, and DSP status.
    Assumptions:
        Input tensors use layout (frames, tx, rx, subcarriers).
    Side Effects:
        Runs SignalProcessor over a bounded number of frames.
    Failure Modes:
        Invalid shape raises ValueError.
    Error Handling:
        Processor failures are captured in processed_ok / processed_finite flags.
    Constraints:
        Validation remains CPU-only and bounded by max_frames for quick feedback.
    Verification:
        Unit test: validate a synthetic 4-frame tensor and assert summary fields.
    References:
        wifi_radar.processing.signal_processor.SignalProcessor.
    """
    if amplitude.shape != phase.shape:
        raise ValueError(f"Amplitude/phase shape mismatch: {amplitude.shape} != {phase.shape}")
    if amplitude.ndim != 4:
        raise ValueError(f"Expected capture shape (frames, tx, rx, subcarriers), got {amplitude.shape}")

    frames = int(amplitude.shape[0])
    finite = bool(np.isfinite(amplitude).all() and np.isfinite(phase).all())
    dynamic_range = float(np.max(amplitude) - np.min(amplitude))

    processor = SignalProcessor()
    processed_ok = True
    processed_finite = True
    n = min(frames, max_frames or frames)
    try:
        for i in range(n):
            amp_p, pha_p = processor.process(amplitude[i], phase[i])
            processed_finite = processed_finite and bool(np.isfinite(amp_p).all() and np.isfinite(pha_p).all())
    except Exception:
        processed_ok = False
        processed_finite = False

    return {
        "frames": frames,
        "shape": list(amplitude.shape[1:]),
        "finite": finite,
        "processed_ok": processed_ok,
        "processed_finite": processed_finite,
        "amplitude_mean": float(np.mean(amplitude)),
        "amplitude_std": float(np.std(amplitude)),
        "phase_mean": float(np.mean(phase)),
        "phase_std": float(np.std(phase)),
        "dynamic_range": dynamic_range,
        "quality": "good" if finite and processed_ok and dynamic_range > 1e-6 else "poor",
    }


def validate_capture_file(path: str, max_frames: int | None = 16) -> Dict[str, Any]:
    """Load and validate a single capture file.

    ID: WR-UTIL-LIVECAP-FILE-001
    Requirement: Combine loading and validation into one call for user-facing tools.
    Purpose: Simplify CLI and test usage for offline live-capture validation.
    Rationale: One wrapper reduces boilerplate and keeps results consistent across entry points.
    Inputs:
        path — str: capture file location.
        max_frames — Optional[int]: upper bound on validation frames.
    Outputs:
        Dict[str, Any] summary including the original path.
    Preconditions:
        path points to a supported capture file.
    Postconditions:
        Summary includes a path field for reporting.
    Assumptions:
        The file is readable from the current environment.
    Side Effects:
        Reads the filesystem and runs bounded DSP validation.
    Failure Modes:
        File or validation errors propagate to the caller.
    Error Handling:
        Relies on load_capture_file() / validate_capture_arrays() for validation.
    Constraints:
        Validation depth is bounded by max_frames.
    Verification:
        Unit test: validate a temp file and assert frames / shape fields.
    References:
        load_capture_file; validate_capture_arrays.
    """
    amplitude, phase = load_capture_file(path)
    summary = validate_capture_arrays(amplitude, phase, max_frames=max_frames)
    summary["path"] = path
    return summary


def main() -> None:
    """Run the live-capture validation command-line interface.

    ID: WR-UTIL-LIVECAP-MAIN-001
    Requirement: Parse CLI arguments, validate the requested capture files, and
                 print either JSON or concise human-readable summaries.
    Purpose: Give operators a quick offline tool for checking real-hardware capture health.
    Rationale: A small CLI reduces friction when validating data on laptops or embedded targets.
    Inputs:
        sys.argv — CLI arguments provided by the user.
    Outputs:
        None — emits validation summaries to stdout.
    Preconditions:
        At least one capture path is provided.
    Postconditions:
        One summary is printed for each capture file.
    Assumptions:
        The requested files are accessible from the current working environment.
    Side Effects:
        Reads capture files from disk and writes to stdout.
    Failure Modes:
        Invalid paths or malformed files raise exceptions during validation.
    Error Handling:
        argparse handles usage errors; validation errors propagate with context.
    Constraints:
        Output format is intentionally simple for shell automation.
    Verification:
        CLI smoke test: run with --help and confirm the usage text renders.
    References:
        argparse.ArgumentParser; validate_capture_file.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Validate live CSI capture files")
    parser.add_argument("capture", nargs="+", help="One or more .npz/.npy capture files")
    parser.add_argument("--max-frames", type=int, default=16, help="Maximum frames to process during validation")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text")
    args = parser.parse_args()

    results = [validate_capture_file(path, max_frames=args.max_frames) for path in args.capture]
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            print(f"{result['path']}: frames={result['frames']} shape={result['shape']} quality={result['quality']}")


if __name__ == "__main__":
    main()

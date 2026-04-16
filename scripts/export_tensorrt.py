
#!/usr/bin/env python3
"""Build TensorRT engines from WiFi-Radar ONNX exports using NVIDIA trtexec."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export WiFi-Radar ONNX models to TensorRT engines")
    parser.add_argument("--encoder-onnx", default=os.path.join("weights", "encoder.onnx"))
    parser.add_argument("--pose-onnx", default=os.path.join("weights", "pose_estimator.onnx"))
    parser.add_argument("--weights", default=os.path.join("weights", "simulation_baseline.pth"))
    parser.add_argument("--output-dir", default=os.path.join("weights", "tensorrt"))
    parser.add_argument("--precision", choices=["fp16", "int8", "best"], default="fp16")
    parser.add_argument("--workspace-mb", type=int, default=1024)
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=4)
    parser.add_argument("--timing-cache", default=os.path.join("weights", "tensorrt.cache"))
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def ensure_onnx(args: argparse.Namespace) -> None:
    if os.path.exists(args.encoder_onnx) and os.path.exists(args.pose_onnx):
        return
    subprocess.run(
        [sys.executable, os.path.join("scripts", "export_onnx.py"), "--weights", args.weights],
        check=True,
    )


def build_engine(onnx_path: str, engine_path: str, shapes: str, args: argparse.Namespace) -> None:
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        raise FileNotFoundError(
            "TensorRT trtexec was not found in PATH. Install TensorRT on the Jetson device first."
        )

    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes={shapes}",
        f"--optShapes={shapes}",
        f"--maxShapes={shapes}",
        f"--timingCacheFile={args.timing_cache}",
        f"--memPoolSize=workspace:{args.workspace_mb}M",
        "--useCudaGraph",
        "--builderOptimizationLevel=5",
        "--profilingVerbosity=detailed",
    ]
    if args.precision == "fp16":
        command.append("--fp16")
    elif args.precision == "int8":
        command.append("--int8")
    elif args.precision == "best":
        command.append("--best")
    if args.skip_inference:
        command.append("--skipInference")
    if args.verbose:
        command.append("--verbose")
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    ensure_onnx(args)
    os.makedirs(args.output_dir, exist_ok=True)

    encoder_engine = os.path.join(args.output_dir, "encoder.plan")
    pose_engine = os.path.join(args.output_dir, "pose_estimator.plan")

    build_engine(
        args.encoder_onnx,
        encoder_engine,
        "amplitude:1x3x3x64,phase:1x3x3x64",
        args,
    )
    build_engine(
        args.pose_onnx,
        pose_engine,
        "features:1x256",
        args,
    )
    print(f"Saved TensorRT engines to {args.output_dir}")


if __name__ == "__main__":
    main()

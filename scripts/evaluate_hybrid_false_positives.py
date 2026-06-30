#!/usr/bin/env python3
"""Evaluate hybrid fusion false-positive rate under controlled RF jitter bursts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

# Ensure project root is on path when executed as a script.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from wifi_radar.analysis.hybrid_activity_fusion import HybridActivityFusion


@dataclass
class EvalConfig:
    runs: int
    frames: int
    persons: int
    noise_std: float
    burst_std: float
    burst_every: int
    burst_len: int
    confidence: float
    risk_alarm_threshold: float
    seed: int


@dataclass
class EvalResult:
    false_alarm_frames: int
    total_frames: int
    false_positive_rate: float
    max_risk: float
    p95_risk: float
    mean_risk: float


def _simulate_stationary_session(cfg: EvalConfig, run_seed: int) -> EvalResult:
    rng = np.random.default_rng(run_seed)
    fusers: List[HybridActivityFusion] = [
        HybridActivityFusion(window_sizes=(4, 8, 16), hysteresis_frames=3)
        for _ in range(cfg.persons)
    ]

    base_amp = np.ones((3, 3, 64), dtype=np.float32)
    base_phase = np.zeros((3, 3, 64), dtype=np.float32)

    total_frames = cfg.frames * cfg.persons
    false_alarm_frames = 0
    risks: List[float] = []

    for frame in range(cfg.frames):
        in_burst = cfg.burst_every > 0 and (frame % cfg.burst_every) < cfg.burst_len
        sigma = cfg.burst_std if in_burst else cfg.noise_std

        for person_idx, fusion in enumerate(fusers):
            amp = base_amp + rng.normal(0.0, sigma, (3, 3, 64)).astype(np.float32)
            phase = base_phase + rng.normal(0.0, sigma, (3, 3, 64)).astype(np.float32)
            pose_conf = np.ones(17, dtype=np.float32) * cfg.confidence

            out = fusion.update(
                amplitude=amp,
                phase=phase,
                pose_confidence=pose_conf,
                gait_metrics=None,
                fall_severity=0,
            )

            risk = float(out["fall_risk"])
            risks.append(risk)
            if out["activity_label"] == "possible_fall" or risk >= cfg.risk_alarm_threshold:
                false_alarm_frames += 1

    risk_arr = np.asarray(risks, dtype=np.float32)
    return EvalResult(
        false_alarm_frames=false_alarm_frames,
        total_frames=total_frames,
        false_positive_rate=float(false_alarm_frames / max(1, total_frames)),
        max_risk=float(np.max(risk_arr) if risk_arr.size else 0.0),
        p95_risk=float(np.percentile(risk_arr, 95.0) if risk_arr.size else 0.0),
        mean_risk=float(np.mean(risk_arr) if risk_arr.size else 0.0),
    )


def _aggregate(results: List[EvalResult]) -> Dict[str, float]:
    fprs = np.asarray([r.false_positive_rate for r in results], dtype=np.float32)
    max_risks = np.asarray([r.max_risk for r in results], dtype=np.float32)
    p95_risks = np.asarray([r.p95_risk for r in results], dtype=np.float32)
    mean_risks = np.asarray([r.mean_risk for r in results], dtype=np.float32)

    return {
        "runs": int(len(results)),
        "fpr_mean": float(np.mean(fprs) if fprs.size else 0.0),
        "fpr_std": float(np.std(fprs) if fprs.size else 0.0),
        "fpr_max": float(np.max(fprs) if fprs.size else 0.0),
        "risk_max_mean": float(np.mean(max_risks) if max_risks.size else 0.0),
        "risk_p95_mean": float(np.mean(p95_risks) if p95_risks.size else 0.0),
        "risk_mean_mean": float(np.mean(mean_risks) if mean_risks.size else 0.0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate false positive rate for hybrid activity fusion")
    parser.add_argument("--runs", type=int, default=20, help="Number of independent Monte Carlo runs")
    parser.add_argument("--frames", type=int, default=500, help="Frames per run")
    parser.add_argument("--persons", type=int, default=2, help="Simulated stationary persons per run")
    parser.add_argument("--noise-std", type=float, default=0.01, help="Gaussian noise std for nominal frames")
    parser.add_argument("--burst-std", type=float, default=0.10, help="Gaussian noise std for burst frames")
    parser.add_argument("--burst-every", type=int, default=80, help="Burst period in frames (0 disables bursts)")
    parser.add_argument("--burst-len", type=int, default=3, help="Burst duration in frames")
    parser.add_argument("--confidence", type=float, default=0.95, help="Pose confidence used in simulation")
    parser.add_argument("--risk-alarm-threshold", type=float, default=0.80, help="Risk threshold counted as alarm")
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EvalConfig(
        runs=max(1, args.runs),
        frames=max(1, args.frames),
        persons=max(1, args.persons),
        noise_std=max(0.0, args.noise_std),
        burst_std=max(0.0, args.burst_std),
        burst_every=max(0, args.burst_every),
        burst_len=max(0, args.burst_len),
        confidence=float(np.clip(args.confidence, 0.0, 1.0)),
        risk_alarm_threshold=float(np.clip(args.risk_alarm_threshold, 0.0, 1.0)),
        seed=int(args.seed),
    )

    results = [_simulate_stationary_session(cfg, cfg.seed + run_id * 17) for run_id in range(cfg.runs)]
    summary = _aggregate(results)

    print("Hybrid False-Positive Evaluation")
    print(f"runs={cfg.runs} frames={cfg.frames} persons={cfg.persons}")
    print(
        "profile: "
        f"noise_std={cfg.noise_std:.4f} "
        f"burst_std={cfg.burst_std:.4f} "
        f"burst_every={cfg.burst_every} "
        f"burst_len={cfg.burst_len} "
        f"risk_alarm_threshold={cfg.risk_alarm_threshold:.2f}"
    )
    print(
        "metrics: "
        f"fpr_mean={summary['fpr_mean']:.6f} "
        f"fpr_std={summary['fpr_std']:.6f} "
        f"fpr_max={summary['fpr_max']:.6f} "
        f"risk_p95_mean={summary['risk_p95_mean']:.6f} "
        f"risk_max_mean={summary['risk_max_mean']:.6f}"
    )

    if args.output_json is not None:
        payload = {
            "config": asdict(cfg),
            "summary": summary,
            "runs": [asdict(item) for item in results],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote: {args.output_json}")


if __name__ == "__main__":
    main()

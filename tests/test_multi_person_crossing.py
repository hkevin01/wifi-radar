import numpy as np
import pytest

from wifi_radar.analysis.gait_analyzer import GaitMetrics
from wifi_radar.analysis.hybrid_activity_fusion import HybridActivityFusion

pytest.importorskip("torch")

from wifi_radar.models.multi_person_tracker import MultiPersonTracker


def _pose_at(cx: float, cy: float, cz: float = 1.0) -> np.ndarray:
    keypoints = np.zeros((17, 3), dtype=np.float32)
    for idx in range(17):
        keypoints[idx, 0] = cx + 0.03 * np.cos(idx / 3.0)
        keypoints[idx, 1] = cy + 0.02 * np.sin(idx / 4.0)
        keypoints[idx, 2] = cz - 0.015 * idx
    return keypoints


def _make_detection(cx: float, cy: float) -> dict:
    return {
        "existence": 0.98,
        "keypoints": _pose_at(cx, cy),
        "confidence": np.ones(17, dtype=np.float32),
    }


def test_multi_person_crossing_id_stability_and_per_person_hybrid_summary_correctness():
    rng = np.random.default_rng(2026)
    tracker = MultiPersonTracker(max_people=4, max_match_distance=0.40, id_timeout_frames=3)
    fusers = {}

    assignment_counts = {}
    actor_labels = {"A": [], "B": []}

    base_amp = np.ones((3, 3, 64), dtype=np.float32)
    base_phase = np.zeros((3, 3, 64), dtype=np.float32)

    for frame in range(48):
        t = frame / 47.0
        a_pos = np.array([-0.85 + 1.70 * t, 0.18, 1.0], dtype=np.float32)
        b_pos = np.array([0.85 - 1.70 * t, -0.18, 1.0], dtype=np.float32)

        detections = [_make_detection(float(a_pos[0]), float(a_pos[1])), _make_detection(float(b_pos[0]), float(b_pos[1]))]
        if rng.random() > 0.5:
            detections.reverse()

        tracked = tracker.update(detections, frame_id=frame)
        assert len(tracked) == 2

        amp = base_amp + rng.normal(0.0, 0.03, (3, 3, 64)).astype(np.float32)
        phase = base_phase + rng.normal(0.0, 0.03, (3, 3, 64)).astype(np.float32)

        per_person_hybrid = []
        for person in tracked:
            pid = int(person.person_id)
            if pid not in fusers:
                fusers[pid] = HybridActivityFusion(window_sizes=(4, 8), hysteresis_frames=2)

            dist_a = float(np.linalg.norm(person.centroid - a_pos))
            dist_b = float(np.linalg.norm(person.centroid - b_pos))
            actor = "A" if dist_a <= dist_b else "B"

            assignment_counts.setdefault(pid, {"A": 0, "B": 0})
            assignment_counts[pid][actor] += 1

            gait = (
                GaitMetrics(
                    cadence_spm=102.0,
                    stride_length=0.62,
                    step_symmetry=0.94,
                    speed_est=1.15,
                    num_steps=9,
                    window_s=8.0,
                )
                if actor == "A"
                else None
            )

            summary = fusers[pid].update(
                amplitude=amp,
                phase=phase,
                pose_confidence=person.confidence,
                gait_metrics=gait,
                fall_severity=0,
            )
            actor_labels[actor].append(summary["activity_label"])
            per_person_hybrid.append({"person_id": pid, **summary})

        per_person_hybrid.sort(key=lambda row: row["person_id"])
        assert len(per_person_hybrid) == 2
        assert per_person_hybrid[0]["person_id"] != per_person_hybrid[1]["person_id"]

    dominant_for_actor = {}
    for actor in ("A", "B"):
        best_pid = max(assignment_counts, key=lambda pid: assignment_counts[pid][actor])
        dominant_for_actor[actor] = best_pid

    assert dominant_for_actor["A"] != dominant_for_actor["B"]

    total_a = sum(assignment_counts[pid]["A"] for pid in assignment_counts)
    total_b = sum(assignment_counts[pid]["B"] for pid in assignment_counts)
    a_consistency = assignment_counts[dominant_for_actor["A"]]["A"] / max(1, total_a)
    b_consistency = assignment_counts[dominant_for_actor["B"]]["B"] / max(1, total_b)

    assert a_consistency >= 0.80
    assert b_consistency >= 0.80

    a_walking_ratio = sum(label in {"walking", "high_motion"} for label in actor_labels["A"]) / max(1, len(actor_labels["A"]))
    b_fall_ratio = sum(label == "possible_fall" for label in actor_labels["B"]) / max(1, len(actor_labels["B"]))

    assert a_walking_ratio >= 0.55
    assert b_fall_ratio == 0.0

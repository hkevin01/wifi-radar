import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from wifi_radar.api.app import AppState, create_app


def test_api_health_and_config_roundtrip():
    state = AppState()
    app = create_app(state)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    payload = {
        "system": {"simulation_mode": True},
        "dashboard": {"port": 9000},
    }
    resp = client.post("/config", json=payload)
    assert resp.status_code == 200
    assert resp.json()["config"]["dashboard"]["port"] == 9000


def test_api_ingest_and_status():
    state = AppState()
    app = create_app(state)
    client = TestClient(app)

    resp = client.post(
        "/ingest",
        json={
            "tracked_people": [{"person_id": 1, "confidence": [1.0] * 17, "keypoints": [[0, 0, 0]] * 17}],
            "gait_metrics": {"cadence_spm": 100, "stride_length": 0.6, "step_symmetry": 1.0, "speed_est": 1.0, "num_steps": 8, "window_s": 5.0},
            "csi_summary": {
                "amplitude_mean": 0.11,
                "phase_mean": -0.02,
                "per_person_hybrid": [
                    {
                        "person_id": 2,
                        "activity_label": "stationary",
                        "motion_score": 0.08,
                        "fall_risk": 0.06,
                    },
                    {
                        "person_id": 1,
                        "activity_label": "walking",
                        "motion_score": 0.42,
                        "fall_risk": 0.08,
                    }
                ],
            },
            "events": [{"message": "fall alert", "severity": 2}],
        },
    )
    assert resp.status_code == 200

    status = client.get("/status")
    assert status.status_code == 200
    body = status.json()
    assert body["tracked_count"] == 1
    assert body["event_count"] >= 1
    assert len(state.snapshot()["csi_summary"]["per_person_hybrid"]) == 2

    hybrid = client.get("/metrics/hybrid/people")
    assert hybrid.status_code == 200
    rows = hybrid.json()["per_person_hybrid"]
    assert [row["person_id"] for row in rows] == [1, 2]
    assert rows[0]["activity_label"] == "walking"
    assert rows[1]["activity_label"] == "stationary"


def test_api_hybrid_people_not_available_before_ingest():
    state = AppState()
    app = create_app(state)
    client = TestClient(app)

    resp = client.get("/metrics/hybrid/people")
    assert resp.status_code == 404

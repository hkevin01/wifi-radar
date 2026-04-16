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
            "events": [{"message": "fall alert", "severity": 2}],
        },
    )
    assert resp.status_code == 200

    status = client.get("/status")
    assert status.status_code == 200
    body = status.json()
    assert body["tracked_count"] == 1
    assert body["event_count"] >= 1

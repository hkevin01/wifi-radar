
"""
ID: WR-API-001
Requirement: Expose a lightweight REST API for health checks, config,
             events, tracked people, and gait metrics.
Purpose: Enable headless and embedded deployments to integrate with the
         WiFi-Radar pipeline over HTTP without the Dash dashboard.
Rationale: FastAPI provides typed request validation and automatic OpenAPI
           documentation with minimal runtime overhead.
Assumptions: FastAPI and uvicorn are installed when the API server is used.
Constraints: In-memory state only; no persistent DB dependency.
References: FastAPI; OpenAPI 3.1.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import uvicorn


class EventPayload(BaseModel):
    message: str
    severity: int = 0
    timestamp: Optional[float] = None
    person_id: Optional[int] = None


class PersonPayload(BaseModel):
    person_id: int
    confidence: List[float] = Field(default_factory=list)
    keypoints: List[List[float]] = Field(default_factory=list)


class IngestPayload(BaseModel):
    tracked_people: List[PersonPayload] = Field(default_factory=list)
    gait_metrics: Optional[Dict[str, Any]] = None
    csi_summary: Optional[Dict[str, Any]] = None
    events: List[EventPayload] = Field(default_factory=list)


@dataclass
class AppState:
    """Shared in-memory state exposed by the REST API.

    ID: WR-API-STATE-001
    Requirement: Hold the latest config, tracked people, gait metrics,
                 and event feed for HTTP access.
    Purpose: Provide thread-safe shared state between the processing loop
             and the REST API endpoints.
    Rationale: A dataclass with a lock is simpler than introducing a DB for
               embedded/headless deployments.
    Inputs:
        config — Optional dict; defaults to a minimal runtime config.
    Outputs:
        None — used as shared mutable state.
    Preconditions:
        None.
    Postconditions:
        State is initialised and ready to serve requests.
    Assumptions:
        Single-process deployment.
    Side Effects:
        Allocates a threading.Lock.
    Failure Modes:
        None expected.
    Error Handling:
        None required.
    Constraints:
        All data is stored in memory only.
    Verification:
        API tests assert config updates and ingest roundtrips.
    References:
        FastAPI application state patterns.
    """
    config: Dict[str, Any] = field(default_factory=lambda: {
        "system": {"simulation_mode": False},
        "dashboard": {"port": 8050},
        "api": {"enabled": False, "host": "0.0.0.0", "port": 8081},
    })
    tracked_people: List[Dict[str, Any]] = field(default_factory=list)
    gait_metrics: Optional[Dict[str, Any]] = None
    csi_summary: Optional[Dict[str, Any]] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "config": dict(self.config),
                "tracked_people": list(self.tracked_people),
                "gait_metrics": dict(self.gait_metrics) if self.gait_metrics else None,
                "csi_summary": dict(self.csi_summary) if self.csi_summary else None,
                "events": list(self.events),
                "uptime_s": round(time.time() - self.started_at, 3),
            }

    def update_config(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self.config = _deep_merge(self.config, patch)
            return dict(self.config)

    def add_events(self, events: List[Dict[str, Any]]) -> None:
        if not events:
            return
        with self._lock:
            for event in events:
                if "timestamp" not in event or event["timestamp"] is None:
                    event = {**event, "timestamp": time.time()}
                self.events.append(event)
            self.events = self.events[-100:]

    def ingest(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self.tracked_people = list(payload.get("tracked_people") or self.tracked_people)
            if payload.get("gait_metrics") is not None:
                self.gait_metrics = dict(payload["gait_metrics"])
            if payload.get("csi_summary") is not None:
                self.csi_summary = dict(payload["csi_summary"])
        self.add_events(list(payload.get("events") or []))
        return self.snapshot()


def create_app(state: Optional[AppState] = None) -> FastAPI:
    """Create and configure the headless REST API application."""
    state = state or AppState()
    app = FastAPI(title="WiFi-Radar API", version="1.0.0")

    @app.get("/")
    def root() -> Dict[str, Any]:
        return {"service": "wifi-radar", "docs": "/docs"}

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "uptime_s": round(time.time() - state.started_at, 3)}

    @app.get("/status")
    def status() -> Dict[str, Any]:
        snap = state.snapshot()
        return {
            "status": "ok",
            "tracked_count": len(snap["tracked_people"]),
            "event_count": len(snap["events"]),
            "has_gait_metrics": snap["gait_metrics"] is not None,
            "simulation_mode": bool(snap["config"].get("system", {}).get("simulation_mode", False)),
            "uptime_s": snap["uptime_s"],
        }

    @app.get("/config")
    def get_config() -> Dict[str, Any]:
        return {"config": state.snapshot()["config"]}

    @app.post("/config")
    def update_config(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"config": state.update_config(payload)}

    @app.post("/ingest")
    def ingest(payload: IngestPayload) -> Dict[str, Any]:
        snapshot = state.ingest(payload.model_dump())
        return {
            "accepted": True,
            "tracked_count": len(snapshot["tracked_people"]),
            "event_count": len(snapshot["events"]),
        }

    @app.get("/people")
    def people() -> Dict[str, Any]:
        snap = state.snapshot()
        return {"tracked_people": snap["tracked_people"]}

    @app.get("/events")
    def events(limit: int = Query(default=20, ge=1, le=100)) -> Dict[str, Any]:
        snap = state.snapshot()
        return {"events": snap["events"][-limit:]}

    @app.get("/metrics/gait")
    def gait_metrics() -> Dict[str, Any]:
        snap = state.snapshot()
        if snap["gait_metrics"] is None:
            raise HTTPException(status_code=404, detail="No gait metrics available yet")
        return {"gait_metrics": snap["gait_metrics"]}

    return app


def run_api_server(host: str = "0.0.0.0", port: int = 8081, state: Optional[AppState] = None) -> None:
    """Start the FastAPI server using uvicorn."""
    uvicorn.run(create_app(state), host=host, port=port, log_level="info")


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


app = create_app()

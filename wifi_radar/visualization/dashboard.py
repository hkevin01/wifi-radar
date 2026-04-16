"""
ID: WR-VIZ-DASH-001
Purpose: Real-time Dash dashboard with three tabs:
         1. Live Monitor   — 3-D pose, CSI signal, detection stats
         2. Events         — Fall-detection alerts and gait metrics
         3. Configuration  — Live-editable settings with YAML persistence

Thread-safety: All shared state is protected by self.data_lock.
"""
import json
import logging
import os
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import yaml
from dash import dcc, html
from dash.dependencies import Input, Output, State


class Dashboard:
    """Real-time dashboard for WiFi-based pose estimation."""

    def __init__(
        self,
        update_interval_ms: int = 100,
        max_history: int = 100,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
    ) -> None:
        self.logger = logging.getLogger("Dashboard")
        self.update_interval_ms = update_interval_ms
        self.max_history = max_history
        self._config = config or {}
        self._config_path = config_path or os.path.expanduser("~/.wifi_radar/config.yaml")

        # ── Live data store ────────────────────────────────────────────────
        self.pose_data: Optional[Dict] = None
        self.confidence_data: Optional[np.ndarray] = None
        self.csi_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.tracked_people: List[Dict] = []   # list of TrackedPerson-like dicts
        self.data_lock = threading.Lock()

        # ── History buffers ────────────────────────────────────────────────
        self.confidence_history = deque(maxlen=max_history)
        self.detected_people_history = deque(maxlen=max_history)

        # ── Events ────────────────────────────────────────────────────────
        self._fall_events: List[Dict] = []     # max 50 most recent
        self._gait_metrics: Optional[Dict] = None
        self._events_lock = threading.Lock()

        # ── Settings change callback ───────────────────────────────────────
        self._on_config_change = None   # callable(new_config) if set

        # ── Dash app ──────────────────────────────────────────────────────
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            title="WiFi-Radar",
            suppress_callback_exceptions=True,
        )
        self._setup_layout()
        self._setup_callbacks()

    # ═══════════════════════════════════════════════════════════════════════ #
    # Layout                                                                  #
    # ═══════════════════════════════════════════════════════════════════════ #

    def _setup_layout(self) -> None:
        """Build the top-level Dash layout and assign it to ``self.app.layout``.

        Structure:
            - Header row: app title + subtitle.
            - ``dcc.Tabs`` with three tabs (Monitor / Events / Configuration).
            - ``html.Div#tab-content`` populated by the ``render_tab`` callback.
            - Two ``dcc.Interval`` timers:
                ``fast-interval`` (``update_interval_ms`` ms) drives live data updates.
                ``slow-interval`` (2 000 ms) drives the events panel.
            - ``dcc.Store#config-save-result`` for inter-callback state passing.

        Side Effects:
            Sets ``self.app.layout`` (mutates the Dash application).
        """
        self.app.layout = dbc.Container(
            [
                dbc.Row([
                    dbc.Col(html.H2("📡 WiFi-Radar", className="text-primary mb-0"), width="auto"),
                    dbc.Col(html.Small("Human Pose Estimation via WiFi CSI",
                                       className="text-muted align-self-center"), width="auto"),
                ], className="mb-3 mt-2"),

                dcc.Tabs(
                    id="main-tabs",
                    value="tab-monitor",
                    className="mb-3",
                    children=[
                        dcc.Tab(label="📊 Live Monitor",  value="tab-monitor"),
                        dcc.Tab(label="🚨 Events",        value="tab-events"),
                        dcc.Tab(label="⚙️  Configuration", value="tab-config"),
                    ],
                ),

                html.Div(id="tab-content"),

                # Shared intervals
                dcc.Interval(id="fast-interval",  interval=self.update_interval_ms, n_intervals=0),
                dcc.Interval(id="slow-interval",  interval=2000,                    n_intervals=0),

                # Config save feedback store
                dcc.Store(id="config-save-result", data=""),
            ],
            fluid=True,
        )

    # ── Tab content builders ─────────────────────────────────────────────── #

    def _monitor_tab(self) -> html.Div:
        """Build the Live Monitor tab layout.

        Contains:
            - An 8-column ``dcc.Graph#pose-graph`` showing the live 3-D skeleton.
            - A 4-column sidebar with:
                ``people-counter`` — headline count of detected people.
                ``confidence-graph`` — rolling confidence + people-count sparkline.
                ``system-status``   — running/no-data text indicator.
                ``csi-graph``       — raw CSI amplitude and phase for TX0·RX0.

        Returns:
            ``html.Div`` subtree for the monitor tab.
        """
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Human Pose (3-D)"),
                        dbc.CardBody(dcc.Graph(
                            id="pose-graph",
                            style={"height": "500px"},
                            figure=self._empty_pose_fig(),
                        )),
                    ])
                ], width=8),

                # Stats sidebar
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Detection Stats"),
                        dbc.CardBody([
                            html.H6("People Detected"),
                            html.H3(id="people-counter", children="0", className="text-success text-center"),
                            html.Hr(),
                            html.H6("Avg Confidence"),
                            dcc.Graph(
                                id="confidence-graph",
                                figure=self._empty_confidence_fig(),
                                style={"height": "150px"},
                            ),
                            html.Hr(),
                            html.H6("System Status"),
                            html.P(id="system-status", children="Initialising …", className="text-warning"),
                        ]),
                    ]),
                    html.Br(),
                    dbc.Card([
                        dbc.CardHeader("CSI Signal (TX0 · RX0)"),
                        dbc.CardBody(dcc.Graph(
                            id="csi-graph",
                            figure=self._empty_csi_fig(),
                            style={"height": "200px"},
                        )),
                    ]),
                ], width=4),
            ]),
        ])

    def _events_tab(self) -> html.Div:
        """Build the Events tab layout.

        Contains:
            - Left column ``fall-events-list``: scrollable list of fall alerts,
              populated by the ``update_events`` callback every 2 seconds.
            - Right column ``gait-metrics-panel``: table of current gait metrics
              (cadence, stride length, step symmetry, speed, step count, window).

        Returns:
            ``html.Div`` subtree for the events tab.
        """
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🚨 Fall Detection Alerts"),
                        dbc.CardBody(
                            html.Div(id="fall-events-list",
                                     children=[html.P("No events yet.", className="text-muted")]),
                        ),
                    ]),
                ], width=6),

                # Gait metrics
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🚶 Gait Analysis"),
                        dbc.CardBody(html.Div(id="gait-metrics-panel",
                                              children=[html.P("Collecting data …", className="text-muted")])),
                    ]),
                ], width=6),
            ]),
        ])

    def _config_tab(self) -> html.Div:
        """Build the Configuration tab layout with live-editable settings.

        Sections:
            - Router / Source: IP, port, simulation toggle.
            - Detection Settings: confidence threshold slider, max-people input.
            - RTMP Streaming: URL and FPS.
            - Fall Detection: enable toggle, velocity threshold slider,
              angle threshold slider.
            - Save button + feedback alert + restart-required notice.

        All input widgets are pre-populated from ``self._config`` so the UI
        reflects the currently active configuration on first render.

        Returns:
            ``html.Div`` subtree for the configuration tab.
        """
        cfg = self._config
        router  = cfg.get("router", {})
        system  = cfg.get("system", {})
        dash_c  = cfg.get("dashboard", {})
        stream  = cfg.get("streaming", {})
        fall_c  = cfg.get("fall_detection", {})

        return html.Div([
            dbc.Row([
                dbc.Col([
                    # ── Router ──────────────────────────────────────────────
                    dbc.Card([
                        dbc.CardHeader("Router / Source"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col(dbc.Label("Router IP")),
                                dbc.Col(dbc.Input(id="cfg-router-ip",   value=router.get("ip", "192.168.1.1"), type="text")),
                            ], className="mb-2"),
                            dbc.Row([
                                dbc.Col(dbc.Label("Router Port")),
                                dbc.Col(dbc.Input(id="cfg-router-port", value=str(router.get("port", 5500)), type="number")),
                            ], className="mb-2"),
                            dbc.Row([
                                dbc.Col(dbc.Label("Simulation Mode")),
                                dbc.Col(dbc.Switch(id="cfg-simulation",
                                                   value=bool(system.get("simulation_mode", True)))),
                            ], className="mb-2"),
                        ]),
                    ], className="mb-3"),

                    # ── Detection ───────────────────────────────────────────
                    dbc.Card([
                        dbc.CardHeader("Detection Settings"),
                        dbc.CardBody([
                            dbc.Label(id="cfg-conf-label",
                                      children=f"Confidence Threshold: {system.get('confidence_threshold', 0.30):.2f}"),
                            dcc.Slider(
                                id="cfg-conf-threshold",
                                min=0.1, max=0.9, step=0.05,
                                value=system.get("confidence_threshold", 0.30),
                                marks={v: f"{v:.1f}" for v in [0.1, 0.3, 0.5, 0.7, 0.9]},
                            ),
                            html.Br(),
                            dbc.Row([
                                dbc.Col(dbc.Label("Max People to Track")),
                                dbc.Col(dbc.Input(id="cfg-max-people",
                                                  value=str(system.get("max_people", 4)),
                                                  type="number", min=1, max=8)),
                            ], className="mb-2"),
                        ]),
                    ], className="mb-3"),
                ], width=6),

                dbc.Col([
                    # ── Streaming ───────────────────────────────────────────
                    dbc.Card([
                        dbc.CardHeader("RTMP Streaming"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col(dbc.Label("RTMP URL")),
                                dbc.Col(dbc.Input(id="cfg-rtmp-url",
                                                  value=stream.get("rtmp_url", "rtmp://localhost/live/wifi_radar"),
                                                  type="text")),
                            ], className="mb-2"),
                            dbc.Row([
                                dbc.Col(dbc.Label("Stream FPS")),
                                dbc.Col(dbc.Input(id="cfg-stream-fps",
                                                  value=str(stream.get("fps", 30)),
                                                  type="number", min=5, max=60)),
                            ], className="mb-2"),
                        ]),
                    ], className="mb-3"),

                    # ── Fall Detection ──────────────────────────────────────
                    dbc.Card([
                        dbc.CardHeader("Fall Detection"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col(dbc.Label("Enable")),
                                dbc.Col(dbc.Switch(id="cfg-fall-enabled",
                                                   value=bool(fall_c.get("enabled", True)))),
                            ], className="mb-2"),
                            dbc.Label(id="cfg-vel-label",
                                      children=f"Velocity Threshold: {fall_c.get('velocity_threshold', -0.20):.2f}"),
                            dcc.Slider(
                                id="cfg-fall-velocity",
                                min=-0.8, max=-0.05, step=0.05,
                                value=fall_c.get("velocity_threshold", -0.20),
                                marks={v: f"{v:.2f}" for v in [-0.8, -0.5, -0.2, -0.05]},
                            ),
                            html.Br(),
                            dbc.Label(id="cfg-angle-label",
                                      children=f"Angle Threshold: {fall_c.get('angle_threshold_deg', 40.0):.0f}°"),
                            dcc.Slider(
                                id="cfg-fall-angle",
                                min=20, max=80, step=5,
                                value=fall_c.get("angle_threshold_deg", 40.0),
                                marks={v: f"{v}°" for v in [20, 40, 60, 80]},
                            ),
                        ]),
                    ], className="mb-3"),

                    # ── Save button ─────────────────────────────────────────
                    dbc.Button("💾 Save Configuration", id="cfg-save-btn",
                               color="success", className="w-100"),
                    html.Div(id="cfg-save-feedback", className="mt-2 text-center"),

                    html.Small(
                        "⚠️  Changes to Router IP / Simulation require restart.",
                        className="text-muted d-block mt-2",
                    ),
                ], width=6),
            ]),
        ])

    # ═══════════════════════════════════════════════════════════════════════ #
    # Callbacks                                                               #
    # ═══════════════════════════════════════════════════════════════════════ #

    def _setup_callbacks(self) -> None:
        """Register all Dash reactive callbacks with ``self.app``.

        Callbacks registered:
            ``render_tab``        — routes tab value to the correct layout builder.
            ``update_monitor``    — fast-interval: updates pose/CSI/confidence figures
                                    and the people counter and system status string.
            ``update_events``     — slow-interval: refreshes the fall-events list and
                                    gait-metrics table.
            ``update_conf_label`` — live label for the confidence-threshold slider.
            ``update_vel_label``  — live label for the velocity-threshold slider.
            ``update_angle_label``— live label for the angle-threshold slider.
            ``save_config``       — serialises form values to YAML and invokes
                                    ``self._on_config_change`` if registered.

        Side Effects:
            Mutates ``self.app`` by registering Dash Input/Output/State bindings.
        """

        # ── Tab routing ──────────────────────────────────────────────────── #
        @self.app.callback(
            Output("tab-content", "children"),
            Input("main-tabs", "value"),
        )
        def render_tab(tab):
            """Return the layout subtree for the selected tab.

            Args:
                tab: The ``value`` of the active ``dcc.Tab`` component.

            Returns:
                ``html.Div`` for the selected tab (monitor / events / config).
            """
            if tab == "tab-monitor":
                return self._monitor_tab()
            if tab == "tab-events":
                return self._events_tab()
            return self._config_tab()

        # ── Live monitor fast update ─────────────────────────────────────── #
        @self.app.callback(
            [
                Output("pose-graph",       "figure"),
                Output("confidence-graph", "figure"),
                Output("csi-graph",        "figure"),
                Output("people-counter",   "children"),
                Output("system-status",    "children"),
                Output("system-status",    "className"),
            ],
            Input("fast-interval", "n_intervals"),
        )
        def update_monitor(n):
            """Refresh all live-monitor widgets on each fast-interval tick.

            Reads shared state under ``self.data_lock``, then builds updated
            Plotly figures without holding the lock (avoids blocking the
            processing thread).  Appends the latest confidence value to the
            rolling history deque before building the sparkline figure.

            Args:
                n: Interval tick counter (unused; triggers reactivity only).

            Returns:
                Tuple (pose_fig, conf_fig, csi_fig, n_people_str,
                       status_text, status_css_class).
            """
            with self.data_lock:
                pose     = self.pose_data
                conf     = self.confidence_data
                csi      = self.csi_data
                n_people = len(self.tracked_people) or (1 if pose is not None else 0)

            pose_fig = self._update_pose_figure(pose)

            if conf is not None:
                self.confidence_history.append(float(np.nanmean(conf)))
            self.detected_people_history.append(n_people)

            conf_fig = self._update_confidence_figure()
            csi_fig  = self._update_csi_figure(csi)

            if pose is None and n > 10:
                status, cls = "No Data", "text-warning"
            else:
                status, cls = "Running", "text-success"

            return pose_fig, conf_fig, csi_fig, str(n_people), status, cls

        # ── Events slow update ───────────────────────────────────────────── #
        @self.app.callback(
            [
                Output("fall-events-list",  "children"),
                Output("gait-metrics-panel","children"),
            ],
            Input("slow-interval", "n_intervals"),
        )
        def update_events(n):
            """Refresh the fall-events list and gait-metrics table every 2 seconds.

            Reads ``self._fall_events`` and ``self._gait_metrics`` under
            ``self._events_lock``.  Fall events are shown newest-first (up to 20).
            Severity levels map to Bootstrap alert colours:
                1 (possible) → warning, 2 (detected) → danger, 3 (alert) → danger.

            Args:
                n: Slow-interval tick counter (triggers reactivity only).

            Returns:
                Tuple (fall_event_ui_elements, gait_metrics_ui_elements).
            """
            with self._events_lock:
                events  = list(self._fall_events)
                metrics = self._gait_metrics

            # ── Fall events list ──────────────────────────────────────────
            if not events:
                fall_ui = [html.P("No events yet.", className="text-muted")]
            else:
                severity_colors = {1: "warning", 2: "danger", 3: "danger"}
                fall_ui = []
                for ev in reversed(events[-20:]):
                    ts_str = time.strftime("%H:%M:%S", time.localtime(ev["timestamp"]))
                    color  = severity_colors.get(ev["severity"], "secondary")
                    fall_ui.append(
                        dbc.Alert(
                            [
                                html.Strong(f"[{ts_str}] Person {ev['person_id']}  —  {ev['message']}"),
                                html.Br(),
                                html.Small(f"Body angle: {ev['body_angle_deg']:.1f}°"),
                            ],
                            color=color,
                            className="mb-1 py-2",
                        )
                    )

            # ── Gait metrics ──────────────────────────────────────────────
            if metrics is None:
                gait_ui = [html.P("Collecting data …", className="text-muted")]
            else:
                gait_ui = [
                    dbc.Table([
                        html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
                        html.Tbody([
                            html.Tr([html.Td("Cadence"),         html.Td(f"{metrics['cadence_spm']:.1f} steps/min")]),
                            html.Tr([html.Td("Stride Length"),   html.Td(f"{metrics['stride_length']:.3f} (norm.)")]),
                            html.Tr([html.Td("Step Symmetry"),   html.Td(f"{metrics['step_symmetry']:.2f}")]),
                            html.Tr([html.Td("Est. Speed"),      html.Td(f"{metrics['speed_est']:.3f} units/s")]),
                            html.Tr([html.Td("Steps in window"), html.Td(str(metrics["num_steps"]))]),
                            html.Tr([html.Td("Window"),          html.Td(f"{metrics['window_s']:.1f} s")]),
                        ]),
                    ], bordered=True, hover=True, size="sm", dark=True),
                ]

            return fall_ui, gait_ui

        # ── Config slider live labels ────────────────────────────────────── #
        @self.app.callback(
            Output("cfg-conf-label",  "children"),
            Input("cfg-conf-threshold", "value"),
        )
        def update_conf_label(v):
            """Format the confidence-threshold slider label with the current value."""
            return f"Confidence Threshold: {v:.2f}" if v is not None else "Confidence Threshold"

        @self.app.callback(
            Output("cfg-vel-label", "children"),
            Input("cfg-fall-velocity", "value"),
        )
        def update_vel_label(v):
            """Format the fall-velocity-threshold slider label with the current value."""
            return f"Velocity Threshold: {v:.2f}" if v is not None else "Velocity Threshold"

        @self.app.callback(
            Output("cfg-angle-label", "children"),
            Input("cfg-fall-angle", "value"),
        )
        def update_angle_label(v):
            """Format the fall-angle-threshold slider label with the current value."""
            return f"Angle Threshold: {int(v)}°" if v is not None else "Angle Threshold"

        # ── Save config ──────────────────────────────────────────────────── #
        @self.app.callback(
            Output("cfg-save-feedback", "children"),
            Input("cfg-save-btn", "n_clicks"),
            [
                State("cfg-router-ip",      "value"),
                State("cfg-router-port",    "value"),
                State("cfg-simulation",     "value"),
                State("cfg-conf-threshold", "value"),
                State("cfg-max-people",     "value"),
                State("cfg-rtmp-url",       "value"),
                State("cfg-stream-fps",     "value"),
                State("cfg-fall-enabled",   "value"),
                State("cfg-fall-velocity",  "value"),
                State("cfg-fall-angle",     "value"),
            ],
            prevent_initial_call=True,
        )
        def save_config(n_clicks, router_ip, router_port, simulation,
                        conf_thr, max_people, rtmp_url, stream_fps,
                        fall_enabled, fall_vel, fall_angle):
            """Persist form values to YAML and invoke the config-change callback.

            Constructs a nested config dict from the form field values, writes it
            to ``self._config_path`` with ``yaml.safe_dump``, updates the in-memory
            config, and calls ``self._on_config_change(new_config)`` if registered.

            Args:
                n_clicks:    Button click count; callback does nothing when 0 or None.
                router_ip:   Router IP address string.
                router_port: Router TCP port integer.
                simulation:  Simulation-mode boolean toggle value.
                conf_thr:    Confidence threshold float (0.1–0.9).
                max_people:  Maximum simultaneous tracked people (1–8).
                rtmp_url:    RTMP destination URL string.
                stream_fps:  Streaming frame rate integer.
                fall_enabled: Fall detection enabled boolean.
                fall_vel:    Velocity threshold float (negative, m/s normalised).
                fall_angle:  Angle threshold float in degrees.

            Returns:
                ``dbc.Alert`` with success or failure message.
            """
            if not n_clicks:
                return ""
            try:
                new_config: Dict[str, Any] = {
                    "router": {
                        "ip":   str(router_ip or "192.168.1.1"),
                        "port": int(router_port or 5500),
                    },
                    "system": {
                        "simulation_mode":       bool(simulation),
                        "confidence_threshold":  float(conf_thr or 0.3),
                        "max_people":            int(max_people or 4),
                    },
                    "streaming": {
                        "rtmp_url": str(rtmp_url or ""),
                        "fps":      int(stream_fps or 30),
                    },
                    "fall_detection": {
                        "enabled":             bool(fall_enabled),
                        "velocity_threshold":  float(fall_vel or -0.2),
                        "angle_threshold_deg": float(fall_angle or 40.0),
                    },
                }
                os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
                with open(self._config_path, "w") as fh:
                    yaml.safe_dump(new_config, fh, default_flow_style=False)

                self._config.update(new_config)
                if self._on_config_change:
                    self._on_config_change(new_config)

                return dbc.Alert("✅ Configuration saved.", color="success", duration=3000)
            except Exception as exc:
                return dbc.Alert(f"❌ Save failed: {exc}", color="danger")

    # ═══════════════════════════════════════════════════════════════════════ #
    # Figure builders                                                         #
    # ═══════════════════════════════════════════════════════════════════════ #

    def _empty_pose_fig(self) -> go.Figure:
        """Return a blank 3-D scatter figure used before any pose data arrives.

        The figure shows an empty cube with labelled axes and a 'Waiting for data'
        title, preventing the graph component from displaying a grey placeholder.

        Returns:
            ``go.Figure`` with a 3-D scatter scene (no data points).
        """
        fig = go.Figure(go.Scatter3d(x=[], y=[], z=[], mode="markers",
                                     marker=dict(size=0, opacity=0)))
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[-1, 1], title="X"),
                yaxis=dict(range=[-1, 1], title="Y"),
                zaxis=dict(range=[-1, 1], title="Z"),
                aspectmode="cube",
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            title="Waiting for data …",
            paper_bgcolor="#222",
            plot_bgcolor="#222",
        )
        return fig

    def _update_pose_figure(self, pose_data: Optional[Dict]) -> go.Figure:
        """Build a 3-D scatter figure from the latest pose keypoints.

        Renders:
            - Keypoint markers coloured by confidence (Viridis colorscale).
            - Skeleton edge lines for the 15 COCO-17 limb connections.
            - Low-confidence keypoints (< 0.3) replaced with NaN so they are
              invisible without removing them from the trace index.

        Args:
            pose_data: Dict with ``keypoints`` (17, 3) and ``confidence`` (17,),
                       or None (returns ``_empty_pose_fig()``).

        Returns:
            Plotly ``go.Figure`` with scatter3d keypoints and line traces.
        """
        if pose_data is None:
            return self._empty_pose_fig()

        kp   = pose_data["keypoints"]
        conf = pose_data["confidence"]
        mask = conf > 0.3
        x, y, z = kp[:, 0].copy(), kp[:, 1].copy(), kp[:, 2].copy()
        x[~mask] = np.nan
        y[~mask] = np.nan
        z[~mask] = np.nan

        edges = [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),(8,9),
                 (7,10),(10,11),(11,12),(7,13),(13,14),(14,15)]

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="markers",
            marker=dict(size=6, color=conf, colorscale="Viridis",
                        opacity=0.85, cmin=0, cmax=1),
            text=[f"KP{i}: {c:.2f}" for i, c in enumerate(conf)],
            hoverinfo="text",
        ))
        for i, j in edges:
            if mask[i] and mask[j]:
                fig.add_trace(go.Scatter3d(
                    x=[x[i], x[j]], y=[y[i], y[j]], z=[z[i], z[j]],
                    mode="lines",
                    line=dict(color="rgba(100,180,255,0.7)", width=3),
                    hoverinfo="none",
                ))
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[-1, 1], title="X"),
                yaxis=dict(range=[-1, 1], title="Y"),
                zaxis=dict(range=[-1, 1], title="Z"),
                aspectmode="cube",
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            showlegend=False,
            paper_bgcolor="#222",
        )
        return fig

    def _empty_confidence_fig(self) -> go.Figure:
        """Return a blank confidence sparkline figure (used during initialisation)."""
        fig = go.Figure()
        fig.update_layout(
            xaxis=dict(title="Time", showticklabels=False),
            yaxis=dict(title="Confidence", range=[0, 1]),
            margin=dict(l=10, r=10, t=10, b=10),
            height=150,
            paper_bgcolor="#333",
            plot_bgcolor="#333",
            font=dict(color="#ccc"),
        )
        return fig

    def _update_confidence_figure(self) -> go.Figure:
        """Build the rolling confidence + people-count sparkline from history deques.

        Renders two overlaid line traces:
            - Confidence (solid green): mean per-keypoint confidence over time.
            - People count (dashed orange): number of detected people over time.

        Both traces share the same X axis (frame index) and Y axis [0, 1.1].

        Returns:
            Plotly ``go.Figure`` with the two line traces.
        """
        fig = go.Figure()
        if self.confidence_history:
            x = list(range(len(self.confidence_history)))
            fig.add_trace(go.Scatter(x=x, y=list(self.confidence_history),
                                     mode="lines", line=dict(color="rgba(0,200,100,0.8)", width=2),
                                     name="Confidence"))
            fig.add_trace(go.Scatter(x=x, y=list(self.detected_people_history),
                                     mode="lines", line=dict(color="rgba(200,100,0,0.6)", width=2, dash="dash"),
                                     name="People"))
        fig.update_layout(
            xaxis=dict(showticklabels=False),
            yaxis=dict(range=[0, 1.1]),
            margin=dict(l=10, r=10, t=10, b=10),
            height=150,
            showlegend=False,
            paper_bgcolor="#333",
            plot_bgcolor="#333",
            font=dict(color="#ccc"),
        )
        return fig

    def _empty_csi_fig(self) -> go.Figure:
        """Return a blank CSI subcarrier figure (used during initialisation)."""
        fig = go.Figure()
        fig.update_layout(xaxis_title="Subcarrier", yaxis_title="Amplitude",
                          margin=dict(l=10, r=10, t=10, b=10), height=200,
                          paper_bgcolor="#333", plot_bgcolor="#333",
                          font=dict(color="#ccc"))
        return fig

    def _update_csi_figure(self, csi_data: Optional[Tuple]) -> go.Figure:
        """Build the CSI subcarrier figure from the first TX-RX pair.

        Renders amplitude (left Y axis) and phase (right Y axis) for antenna
        pair (TX0, RX0) across all subcarriers so the operator can verify the
        raw signal quality.

        Args:
            csi_data: Tuple (amplitude, phase) where each array is shaped
                      (num_tx, num_rx, num_subcarriers), or None.

        Returns:
            Plotly ``go.Figure`` with dual-Y-axis line traces.
        """
        fig = go.Figure()
        if csi_data is not None:
            amp, phase = csi_data
            x = np.arange(amp.shape[2])
            fig.add_trace(go.Scatter(x=x, y=amp[0, 0],
                                     mode="lines", line=dict(color="rgba(0,100,200,0.8)", width=2),
                                     name="Amplitude"))
            fig.add_trace(go.Scatter(x=x, y=phase[0, 0],
                                     mode="lines", line=dict(color="rgba(200,0,100,0.6)", width=2),
                                     name="Phase", yaxis="y2"))
        fig.update_layout(
            xaxis_title="Subcarrier",
            yaxis_title="Amplitude",
            yaxis2=dict(title="Phase", overlaying="y", side="right",
                        range=[-np.pi, np.pi]),
            margin=dict(l=10, r=10, t=10, b=10),
            height=200,
            showlegend=True,
            legend=dict(orientation="h", y=1.1),
            paper_bgcolor="#333",
            plot_bgcolor="#333",
            font=dict(color="#ccc"),
        )
        return fig

    # ═══════════════════════════════════════════════════════════════════════ #
    # Data ingestion (thread-safe)                                            #
    # ═══════════════════════════════════════════════════════════════════════ #

    def update_data(
        self,
        pose_data: Optional[Dict] = None,
        confidence_data: Optional[np.ndarray] = None,
        csi_data: Optional[Tuple] = None,
        tracked_people: Optional[List] = None,
    ) -> None:
        """Thread-safe update of live inference results for the next dashboard refresh.

        Called by the processing thread after each forward pass.  Only non-None
        arguments overwrite the corresponding fields, so partial updates (e.g.
        updating only ``csi_data``) are safe.

        Args:
            pose_data:        Dict with ``keypoints`` (17, 3) and ``confidence`` (17,).
            confidence_data:  (17,) confidence array for the primary person.
            csi_data:         Tuple (amplitude, phase) raw CSI arrays.
            tracked_people:   List of TrackedPerson-like dicts with ``person_id``,
                              ``keypoints``, and ``confidence`` fields.

        Thread Safety:
            All writes are wrapped in ``self.data_lock``.  The fast-interval
            callback reads the same fields under the same lock.
        """
        with self.data_lock:
            if pose_data       is not None: self.pose_data       = pose_data
            if confidence_data is not None: self.confidence_data = confidence_data
            if csi_data        is not None: self.csi_data        = csi_data
            if tracked_people  is not None: self.tracked_people  = tracked_people

    def update_events(
        self,
        fall_events: Optional[List[Dict]] = None,
        gait_metrics: Optional[Dict]      = None,
    ) -> None:
        """Thread-safe append of fall events and replacement of gait metrics.

        Args:
            fall_events:  List of fall-event dicts to append.  The internal list
                          is capped at 50 entries (oldest are dropped).
            gait_metrics: Latest gait metrics dict; replaces previous value.

        Thread Safety:
            All mutations are wrapped in ``self._events_lock``.
        """
        with self._events_lock:
            if fall_events is not None:
                self._fall_events.extend(fall_events)
                self._fall_events = self._fall_events[-50:]   # keep last 50
            if gait_metrics is not None:
                self._gait_metrics = gait_metrics

    def set_config_change_callback(self, fn) -> None:
        """Register a callable invoked when the user saves config from the UI."""
        self._on_config_change = fn

    # ═══════════════════════════════════════════════════════════════════════ #
    # Start                                                                   #
    # ═══════════════════════════════════════════════════════════════════════ #

    def run(self, debug: bool = False, port: int = 8050) -> None:
        """Start the Dash development server (blocking).

        Args:
            debug: If True, enables Dash hot-reloading and error overlays.
                   Should be False in production to avoid exposing debug info.
            port:  TCP port the dashboard HTTP server listens on.

        Side Effects:
            Blocks the calling thread until the server is stopped (Ctrl-C or
            process termination).  ``use_reloader=False`` prevents the Werkzeug
            reloader from spawning a second process that would duplicate threads.
        """
        self.logger.info("Starting dashboard on port %d", port)
        self.app.run(debug=debug, port=port, use_reloader=False)

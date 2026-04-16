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
    """Real-time Dash/Plotly dashboard for WiFi-based pose estimation.

    ID: WR-VIZ-DASH-CLASS-001
    Requirement: Serve a Dash web application with three tabs (Monitor, Events,
                 Configuration) that update in real-time from inference thread data.
    Purpose: Provide operators with a browser-based interface to observe live pose
             output, review fall/gait events, and adjust system configuration
             without restarting the pipeline.
    Rationale: Dash callbacks with dcc.Interval provide a push-free polling model
               that avoids WebSocket complexity while still refreshing at 10 Hz.
    Inputs:
        update_interval_ms — int: fast-update interval in milliseconds.
        max_history        — int: rolling buffer depth for sparkline charts.
        config             — Optional[Dict]: initial configuration values.
        config_path        — Optional[str]: YAML file path for persistent config.
    Outputs:
        A running Dash HTTP server accessible at http://localhost:port/.
    Preconditions:
        dash, dash-bootstrap-components, plotly must be installed.
    Assumptions:
        Inference thread calls update_data() and update_events() concurrently.
    Constraints:
        data_lock and _events_lock must be held for all shared state access.
    References:
        Dash documentation; WR-VIZ-DASH-001 module docstring.
    """

    def __init__(
        self,
        update_interval_ms: int = 100,
        max_history: int = 100,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
    ) -> None:
        """Initialise the Dash application, layout, and all shared state.

        ID: WR-VIZ-DASH-INIT-001
        Requirement: Create the Dash app instance, call _setup_layout() and
                     _setup_callbacks(), and initialise all thread-safety
                     primitives and data buffers.
        Purpose: Produce a fully configured but not yet running Dash app so
                 the caller can start it at the appropriate point in the lifecycle.
        Rationale: Separating construction from run() allows main.py to start all
                   subsystems before the dashboard server starts consuming the port.
        Inputs:
            update_interval_ms — int > 0: fast-update poll interval in ms.
            max_history        — int > 0: rolling history depth for sparklines.
            config             — Optional[Dict]: initial config dict.
            config_path        — Optional[str]: path to YAML config file.
        Outputs:
            None — initialises self.
        Preconditions:
            None.
        Postconditions:
            self.app is a configured Dash instance.
            self.data_lock and self._events_lock are threading.Lock objects.
        Assumptions:
            run() is called after all subsystems are started.
        Side Effects:
            Creates a Dash app; calls _setup_layout() and _setup_callbacks().
        Failure Modes:
            Missing Dash dependencies: ImportError at module load time.
        Error Handling:
            None at construction; Dash validates component props at runtime.
        Constraints:
            suppress_callback_exceptions=True required for dynamic tab content.
        Verification:
            Unit test: construct; assert self.app is not None.
        References:
            dash.Dash; dbc.themes.DARKLY; WR-VIZ-DASH-CLASS-001.
        """
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
        """Build the top-level Dash layout and assign it to self.app.layout.

        ID: WR-VIZ-DASH-LAYOUT-001
        Requirement: Construct a dbc.Container with header, tab selector,
                     tab content div, two dcc.Interval timers, and a
                     dcc.Store for config save feedback; assign to app.layout.
        Purpose: Define the complete page structure once at startup so all
                 callbacks can reference component IDs by string without
                 the components being created lazily.
        Rationale: Top-level layout with a content div populated by the
                   render_tab callback enables tab isolation without loading
                   all three tab layouts simultaneously.
        Inputs:
            None — reads self.update_interval_ms.
        Outputs:
            None — sets self.app.layout.
        Preconditions:
            self.app must be a fully initialised Dash instance.
        Postconditions:
            self.app.layout is a dbc.Container with all required component IDs.
        Assumptions:
            dbc.themes.DARKLY is available.
        Side Effects:
            Mutates self.app.layout.
        Failure Modes:
            Missing component IDs: Dash raises dash.exceptions.NonExistentIdException
            when callbacks reference them.
        Error Handling:
            None required; layout is statically defined.
        Constraints:
            fast-interval: update_interval_ms; slow-interval: 2000 ms.
        Verification:
            Smoke test: app.layout is not None after __init__().
        References:
            dbc.Container; dcc.Tabs; dcc.Interval; dcc.Store; WR-VIZ-DASH-001.
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
        """Build the Live Monitor tab layout with pose graph and stats sidebar.

        ID: WR-VIZ-DASH-MONTAB-001
        Requirement: Return an html.Div containing a pose-graph card (width=8) and
                     a stats sidebar (width=4) with people-counter, confidence-graph,
                     system-status, and csi-graph component IDs.
        Purpose: Provide the live operator view showing 3-D skeleton, detection
                 confidence trend, and raw CSI signal quality side-by-side.
        Rationale: An 8/4 column split maximises the 3-D pose graph while keeping
                   the stats sidebar visible without scrolling.
        Inputs:
            None.
        Outputs:
            html.Div — complete monitor tab layout subtree.
        Preconditions:
            self._empty_pose_fig(), self._empty_confidence_fig(),
            self._empty_csi_fig() must be callable.
        Postconditions:
            Returned layout contains component IDs: pose-graph, people-counter,
            confidence-graph, system-status, csi-graph.
        Side Effects:
            None.
        Failure Modes:
            Missing component IDs: callbacks referencing them will fail at runtime.
        Verification:
            Smoke test: call _monitor_tab(); assert isinstance(result, html.Div).
        References:
            dcc.Graph; dbc.Card; WR-VIZ-DASH-001.
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
        """Build the Events tab layout with fall alerts and gait metrics panel.

        ID: WR-VIZ-DASH-EVTTAB-001
        Requirement: Return an html.Div containing fall-events-list (width=6)
                     and gait-metrics-panel (width=6) component IDs.
        Purpose: Provide operators with a chronological view of fall alerts and
                 current gait analysis metrics for clinical review.
        Rationale: A 50/50 column split gives equal prominence to fall detection
                   and gait analysis, both being primary clinical outputs.
        Inputs:
            None.
        Outputs:
            html.Div — complete events tab layout subtree.
        Preconditions:
            None.
        Postconditions:
            Returned layout contains: fall-events-list, gait-metrics-panel IDs.
        Side Effects:
            None.
        Failure Modes:
            Missing component IDs referenced by update_events callback.
        Verification:
            Smoke test: call _events_tab(); assert isinstance(result, html.Div).
        References:
            dbc.Alert; dbc.Table; WR-VIZ-DASH-001.
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

        ID: WR-VIZ-DASH-CFGTAB-001
        Requirement: Return an html.Div containing form inputs for Router IP/Port,
                     simulation toggle, confidence threshold, max-people, RTMP URL,
                     stream FPS, fall detection enable/thresholds, and a save button.
        Purpose: Allow operators to adjust system parameters at runtime without
                 editing YAML files or restarting the application.
        Rationale: Pre-populating inputs from self._config ensures the UI always
                   reflects the currently active configuration on first render.
        Inputs:
            None — reads self._config for initial widget values.
        Outputs:
            html.Div — complete configuration tab layout subtree.
        Preconditions:
            None.
        Postconditions:
            Returned layout contains all cfg-* component IDs required by
            save_config callback State declarations.
        Side Effects:
            None.
        Failure Modes:
            Missing cfg-* IDs: save_config callback will raise at runtime.
        Constraints:
            All input IDs must match State() arguments in save_config callback.
        Verification:
            Smoke test: call _config_tab(); assert isinstance(result, html.Div).
        References:
            dbc.Input; dcc.Slider; dbc.Switch; dbc.Button; WR-VIZ-DASH-001.
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
        """Register all Dash reactive callbacks with self.app.

        ID: WR-VIZ-DASH-CALLBACKS-001
        Requirement: Register render_tab, update_monitor, update_events,
                     update_conf_label, update_vel_label, update_angle_label,
                     and save_config callbacks with self.app.
        Purpose: Wire all Dash Input/Output/State bindings so the browser UI
                 reacts to user interaction and interval ticks.
        Rationale: Registering all callbacks from one method keeps them in one
                   place for auditing; inner-function closures provide access to
                   self without requiring a Dash server context variable.
        Inputs:
            None — operates on self.app.
        Outputs:
            None — mutates self.app as a side effect.
        Preconditions:
            _setup_layout() must have been called first (component IDs must exist).
        Postconditions:
            self.app has 7 registered callbacks.
        Assumptions:
            suppress_callback_exceptions=True is set (required for dynamic tabs).
        Side Effects:
            Mutates self.app by calling app.callback() decorators.
        Failure Modes:
            Duplicate callback IDs: Dash raises DuplicateCallbackOutput error.
        Error Handling:
            None; Dash raises on misconfiguration at import time.
        Constraints:
            Inner callbacks capture self via closure; must not be called
            from outside the Dashboard instance.
        Verification:
            Smoke test: construct Dashboard; run server; open browser; verify
            tabs render and monitor updates at configured interval.
        References:
            dash.dependencies.Input/Output/State; WR-VIZ-DASH-CLASS-001.
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

        ID: WR-VIZ-DASH-EPOSEFIG-001
        Requirement: Return a go.Figure with an empty Scatter3d trace, a labelled
                     [-1,1]^3 cube scene, a 'Waiting for data' title, and dark
                     background colours.
        Purpose: Prevent the pose-graph card from showing a grey placeholder
                 before the first frame arrives from the inference thread.
        Rationale: A fully configured empty figure avoids layout shifts when
                   data arrives and replaces the figure.
        Inputs:
            None.
        Outputs:
            go.Figure — empty 3-D scatter with configured layout.
        Preconditions:
            plotly.graph_objects must be available.
        Postconditions:
            Returned figure has zero data points.
        Side Effects:
            None — pure function; allocates a new Figure.
        Failure Modes:
            None.
        Verification:
            Unit test: assert len(fig.data[0].x) == 0.
        References:
            go.Scatter3d; go.Figure.update_layout; WR-VIZ-DASH-CLASS-001.
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

        ID: WR-VIZ-DASH-UPOSEFIG-001
        Requirement: Render COCO-17 keypoints as Viridis-coloured Scatter3d markers
                     and draw 15 skeleton edge lines; mask low-confidence keypoints
                     with NaN so they are invisible.
        Purpose: Provide operators with a 3-D visualisation of the estimated human
                 pose updated at the fast-interval rate.
        Rationale: NaN masking preserves keypoint index alignment for line traces
                   without removing entries from the trace arrays.
        Inputs:
            pose_data — Optional[Dict]: {'keypoints': (17,3), 'confidence': (17,)};
                        None triggers _empty_pose_fig().
        Outputs:
            go.Figure — Scatter3d figure with keypoints and skeleton edges.
        Preconditions:
            pose_data must contain 'keypoints' and 'confidence' keys if not None.
        Postconditions:
            All edge traces connect only valid (confidence > 0.3) keypoints.
        Assumptions:
            Keypoints are normalised to [-1, 1]; COCO-17 edge list is hard-coded.
        Side Effects:
            None — pure function; allocates a new Figure.
        Failure Modes:
            pose_data is None: returns _empty_pose_fig().
        Verification:
            Unit test: provide synthetic pose_data; assert figure has > 1 trace.
        References:
            go.Scatter3d; COCO-17 edge list; WR-VIZ-DASH-CLASS-001.
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
        """Return a blank confidence sparkline figure used during initialisation.

        ID: WR-VIZ-DASH-ECONFFIG-001
        Requirement: Return an empty go.Figure with Y axis [0,1] and dark
                     background, ready to be populated by _update_confidence_figure.
        Purpose: Prevent the confidence-graph card from showing a grey placeholder
                 at startup.
        Rationale: Consistent empty figure style avoids layout shifts on first update.
        Inputs:
            None.
        Outputs:
            go.Figure — empty confidence sparkline with configured layout.
        Side Effects:
            None — pure function.
        Verification:
            Unit test: assert fig.layout.yaxis.range == [0, 1].
        References:
            go.Figure.update_layout; WR-VIZ-DASH-CLASS-001.
        """
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

        ID: WR-VIZ-DASH-UCONFFIG-001
        Requirement: Render two overlaid line traces (confidence and people count)
                     from self.confidence_history and self.detected_people_history.
        Purpose: Give operators a quick visual of detection quality and occupancy
                 trends over the rolling history window.
        Rationale: Overlaid traces share one X axis (frame index) so relative
                   timing is visually clear without a secondary X axis.
        Inputs:
            None — reads self.confidence_history and self.detected_people_history.
        Outputs:
            go.Figure — line chart with confidence and people-count traces.
        Preconditions:
            Deques may be empty; handled gracefully (returns blank figure).
        Postconditions:
            Returned figure has 0 or 2 traces.
        Side Effects:
            None — read-only; allocates a new Figure.
        Failure Modes:
            Empty deques: returns a blank figure.
        Verification:
            Unit test: add 5 items to history deques; assert figure has 2 traces.
        References:
            go.Scatter; WR-VIZ-DASH-CLASS-001.
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
        """Return a blank CSI subcarrier figure used during initialisation.

        ID: WR-VIZ-DASH-ECSIFIG-001
        Requirement: Return an empty go.Figure with 'Subcarrier'/'Amplitude' axis
                     labels and dark background, ready for _update_csi_figure.
        Purpose: Prevent the CSI card from showing a grey placeholder at startup.
        Rationale: Consistent empty figure style avoids layout shifts on first update.
        Inputs:
            None.
        Outputs:
            go.Figure — empty CSI figure with configured layout.
        Side Effects:
            None — pure function.
        Verification:
            Unit test: assert fig.layout.xaxis.title.text == 'Subcarrier'.
        References:
            go.Figure.update_layout; WR-VIZ-DASH-CLASS-001.
        """
        fig = go.Figure()
        fig.update_layout(xaxis_title="Subcarrier", yaxis_title="Amplitude",
                          margin=dict(l=10, r=10, t=10, b=10), height=200,
                          paper_bgcolor="#333", plot_bgcolor="#333",
                          font=dict(color="#ccc"))
        return fig

    def _update_csi_figure(self, csi_data: Optional[Tuple]) -> go.Figure:
        """Build the CSI subcarrier figure from the first TX-RX antenna pair.

        ID: WR-VIZ-DASH-UCSIFIG-001
        Requirement: Plot amplitude (primary Y) and phase (secondary Y) for antenna
                     pair (TX0, RX0) across all subcarriers; return empty figure if
                     csi_data is None.
        Purpose: Allow operators to verify raw signal quality and check for anomalies
                 or interference patterns in the CSI measurement.
        Rationale: Dual-Y axes let amplitude and phase be shown at their natural
                   scales without normalisation.
        Inputs:
            csi_data — Optional[Tuple[np.ndarray, np.ndarray]]: (amplitude, phase)
                       each shaped (num_tx, num_rx, num_sub).
        Outputs:
            go.Figure — dual-Y line chart with amplitude and phase traces.
        Preconditions:
            csi_data arrays must be at least shape (1,1,N).
        Postconditions:
            Phase Y axis is locked to [-pi, pi].
        Assumptions:
            Only TX0/RX0 pair is plotted; multi-antenna display not implemented.
        Side Effects:
            None — pure function; allocates a new Figure.
        Failure Modes:
            csi_data is None: returns _empty_csi_fig().
        Verification:
            Unit test: provide 3x3x56 arrays; assert figure has 2 traces.
        References:
            go.Scatter dual Y axis; WR-VIZ-DASH-CLASS-001.
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

        ID: WR-VIZ-DASH-UPDATEDATA-001
        Requirement: Overwrite non-None fields in shared state under data_lock
                     so the fast-interval callback sees the most recent values.
        Purpose: Decouple the inference pipeline thread from the Dash callback
                 thread using a double-buffer pattern protected by data_lock.
        Rationale: Only non-None arguments overwrite their fields so partial
                   updates (e.g. CSI only) do not corrupt unrelated fields.
        Inputs:
            pose_data        — Optional[Dict]: {'keypoints': (17,3), 'confidence': (17,)}.
            confidence_data  — Optional[np.ndarray]: (17,) per-keypoint confidence.
            csi_data         — Optional[Tuple[ndarray, ndarray]]: (amplitude, phase).
            tracked_people   — Optional[List[Dict]]: multi-person track list.
        Outputs:
            None — updates self fields as side effects.
        Preconditions:
            data_lock must not already be held by the calling thread.
        Postconditions:
            Updated fields reflect the provided arguments.
        Assumptions:
            Called from the processing thread; not from a Dash callback.
        Side Effects:
            Acquires self.data_lock; overwrites non-None shared fields.
        Failure Modes:
            None; lock prevents data races.
        Error Handling:
            None required.
        Constraints:
            Lock held only for the duration of field assignments.
        Verification:
            Unit test: call update_data(pose_data=x); with data_lock: assert pose_data==x.
        References:
            threading.Lock; WR-VIZ-DASH-CLASS-001.
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

        ID: WR-VIZ-DASH-UPDATEEVT-001
        Requirement: Append fall_events to self._fall_events (cap at 50) and replace
                     self._gait_metrics, both under self._events_lock.
        Purpose: Provide the slow-interval callback with up-to-date event and metric
                 data without data races from the processing thread.
        Rationale: A separate _events_lock from data_lock avoids priority inversion
                   between high-frequency pose updates and low-frequency event updates.
        Inputs:
            fall_events  — Optional[List[Dict]]: events to append.
            gait_metrics — Optional[Dict]: latest metrics to replace current value.
        Outputs:
            None — updates self fields as side effects.
        Preconditions:
            _events_lock must not already be held by the calling thread.
        Postconditions:
            self._fall_events has at most 50 entries.
            self._gait_metrics reflects the latest provided value.
        Assumptions:
            Called from the processing thread.
        Side Effects:
            Acquires self._events_lock; extends and truncates self._fall_events;
            replaces self._gait_metrics.
        Failure Modes:
            None; lock prevents data races.
        Error Handling:
            None required.
        Constraints:
            Event list capped at 50 entries (oldest dropped).
        Verification:
            Unit test: append 60 events; assert len(_fall_events) == 50.
        References:
            threading.Lock; WR-VIZ-DASH-CLASS-001.
        """
        with self._events_lock:
            if fall_events is not None:
                self._fall_events.extend(fall_events)
                self._fall_events = self._fall_events[-50:]   # keep last 50
            if gait_metrics is not None:
                self._gait_metrics = gait_metrics

    def set_config_change_callback(self, fn) -> None:
        """Register a callable invoked when the user saves config from the UI.

        ID: WR-VIZ-DASH-SETCFGCB-001
        Requirement: Store fn as self._on_config_change so the save_config Dash
                     callback can invoke it with the new config dict after saving.
        Purpose: Allow main.py to register a callback that propagates config
                 changes to running subsystems without the dashboard importing them.
        Rationale: Callback registration decouples the dashboard from specific
                   subsystem classes, keeping it a pure presentation layer.
        Inputs:
            fn — callable(new_config: Dict) — invoked with the new config dict.
        Outputs:
            None — stores fn as self._on_config_change.
        Preconditions:
            fn must be callable with one dict argument.
        Postconditions:
            self._on_config_change == fn.
        Side Effects:
            Sets self._on_config_change.
        Failure Modes:
            Non-callable fn: will raise TypeError when save_config invokes it.
        Error Handling:
            None at registration time; save_config handles call exceptions.
        Verification:
            Unit test: register fn; save config; assert fn was called with new dict.
        References:
            save_config inner callback; WR-VIZ-DASH-CALLBACKS-001.
        """
        self._on_config_change = fn

    # ═══════════════════════════════════════════════════════════════════════ #
    # Start                                                                   #
    # ═══════════════════════════════════════════════════════════════════════ #

    def run(self, debug: bool = False, port: int = 8050) -> None:
        """Start the Dash development server (blocking call).

        ID: WR-VIZ-DASH-RUN-001
        Requirement: Start the Werkzeug HTTP server serving the Dash application
                     on the specified port; block the calling thread until stopped.
        Purpose: Activate the browser-accessible dashboard so operators can connect
                 to http://localhost:port/ after all subsystems are started.
        Rationale: use_reloader=False prevents Werkzeug from spawning a child
                   process that would duplicate threads and open the RTMP streamer twice.
        Inputs:
            debug — bool: if True enables Dash hot-reloading and error overlays.
            port  — int: TCP port for the HTTP server (default 8050).
        Outputs:
            None — blocks until interrupted.
        Preconditions:
            __init__() must have completed; all subsystems should be started before
            calling run() so they are ready when the first browser request arrives.
        Postconditions:
            Server is running; blocks until Ctrl-C or process termination.
        Assumptions:
            The caller has started the processing thread before calling run().
        Side Effects:
            Binds TCP port; logs INFO message.
        Failure Modes:
            Port already in use: OSError raised by Werkzeug.
        Error Handling:
            None; caller must handle OSError.
        Constraints:
            debug=True must NOT be used in production (exposes error tracebacks).
        Verification:
            Integration test: start in a thread; assert HTTP GET / returns 200.
        References:
            Dash.run; Werkzeug use_reloader=False; WR-VIZ-DASH-CLASS-001.
        """
        self.logger.info("Starting dashboard on port %d", port)
        self.app.run(debug=debug, port=port, use_reloader=False)

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
import threading
import time
import logging
import json
from collections import deque

class Dashboard:
    """Real-time dashboard for visualizing WiFi-based pose estimation."""
    
    def __init__(self, update_interval_ms=100, max_history=100):
        self.logger = logging.getLogger("Dashboard")
        self.update_interval_ms = update_interval_ms
        self.max_history = max_history
        
        # Data storage
        self.pose_data = None
        self.confidence_data = None
        self.csi_data = None
        self.data_lock = threading.Lock()
        
        # History for plots
        self.confidence_history = deque(maxlen=max_history)
        self.detected_people_history = deque(maxlen=max_history)
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, 
                           external_stylesheets=[dbc.themes.DARKLY],
                           title="WiFi-Radar")
        
        # Configure layout
        self._setup_layout()
        
        # Configure callbacks
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("WiFi-Radar: Human Pose Estimation", 
                           className="text-center text-primary mb-4")
                ], width=12)
            ]),
            
            dbc.Row([
                # 3D Pose Visualization
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Human Pose Visualization"),
                        dbc.CardBody([
                            dcc.Graph(id="pose-graph", 
                                     style={"height": "500px"},
                                     figure=self._create_empty_pose_figure())
                        ])
                    ])
                ], width=8),
                
                # Stats and controls
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Detection Stats"),
                        dbc.CardBody([
                            html.Div([
                                html.H5("People Detected:"),
                                html.H2(id="people-counter", children="0", 
                                      className="text-center text-success")
                            ]),
                            html.Hr(),
                            html.Div([
                                html.H5("Confidence:"),
                                dcc.Graph(id="confidence-graph", 
                                         figure=self._create_empty_confidence_figure(),
                                         style={"height": "150px"})
                            ]),
                            html.Hr(),
                            html.Div([
                                html.H5("System Status:"),
                                html.P(id="system-status", children="Running", 
                                      className="text-success")
                            ])
                        ])
                    ]),
                    
                    html.Br(),
                    
                    dbc.Card([
                        dbc.CardHeader("CSI Data Visualization"),
                        dbc.CardBody([
                            dcc.Graph(id="csi-graph", 
                                     figure=self._create_empty_csi_figure(),
                                     style={"height": "200px"})
                        ])
                    ])
                ], width=4)
            ]),
            
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval_ms,
                n_intervals=0
            )
        ], fluid=True)
    
    def _setup_callbacks(self):
        """Set up the dashboard callbacks."""
        @self.app.callback(
            [Output('pose-graph', 'figure'),
             Output('confidence-graph', 'figure'),
             Output('csi-graph', 'figure'),
             Output('people-counter', 'children'),
             Output('system-status', 'children'),
             Output('system-status', 'className')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_graphs(n_intervals):
            # Acquire lock to prevent data race
            with self.data_lock:
                pose_data = self.pose_data
                confidence_data = self.confidence_data
                csi_data = self.csi_data
                
            # Update pose visualization
            pose_fig = self._update_pose_figure(pose_data)
            
            # Update confidence history and graph
            if confidence_data is not None:
                avg_confidence = np.nanmean(confidence_data)
                self.confidence_history.append(avg_confidence)
                people_detected = 1 if pose_data is not None else 0
                self.detected_people_history.append(people_detected)
            
            confidence_fig = self._update_confidence_figure()
            
            # Update CSI visualization
            csi_fig = self._update_csi_figure(csi_data)
            
            # Count people
            people_count = "0"
            if pose_data is not None:
                people_count = "1"  # Simplified - actual implementation would count multiple people
                
            # System status
            system_status = "Running"
            status_class = "text-success"
            if pose_data is None and n_intervals > 10:
                system_status = "No Data"
                status_class = "text-warning"
            
            return pose_fig, confidence_fig, csi_fig, people_count, system_status, status_class
    
    def _create_empty_pose_figure(self):
        """Create an empty 3D pose visualization figure."""
        fig = go.Figure(data=[go.Scatter3d(
            x=[], y=[], z=[],
            mode='markers',
            marker=dict(
                size=0,
                color='blue',
                opacity=0
            )
        )])
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[-1, 1], title='X'),
                yaxis=dict(range=[-1, 1], title='Y'),
                zaxis=dict(range=[-1, 1], title='Z'),
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            scene_camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            title="No Pose Data"
        )
        
        return fig
    
    def _update_pose_figure(self, pose_data):
        """Update the 3D pose visualization with new data."""
        if pose_data is None:
            return self._create_empty_pose_figure()
            
        # Convert keypoints to coordinates
        keypoints = pose_data['keypoints']
        confidence = pose_data['confidence']
        
        # Filter low-confidence keypoints
        threshold = 0.3
        valid_mask = confidence > threshold
        
        # Get coordinates
        x = keypoints[:, 0]
        y = keypoints[:, 1]
        z = keypoints[:, 2]
        
        # Set invalid keypoints to NaN
        x[~valid_mask] = np.nan
        y[~valid_mask] = np.nan
        z[~valid_mask] = np.nan
        
        # Define human skeleton connections
        # This is a simplified skeleton - actual implementation would have more connections
        edges = [
            (0, 1), (1, 2), (2, 3),  # Right leg
            (0, 4), (4, 5), (5, 6),  # Left leg
            (0, 7),  # Spine
            (7, 8), (8, 9),  # Neck and head
            (7, 10), (10, 11), (11, 12),  # Right arm
            (7, 13), (13, 14), (14, 15)   # Left arm
        ]
        
        # Create figure
        fig = go.Figure()
        
        # Add keypoints
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=6,
                color=confidence,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Confidence")
            ),
            text=[f"Keypoint {i}: {conf:.2f}" for i, conf in enumerate(confidence)],
            hoverinfo="text"
        ))
        
        # Add skeleton lines
        for edge in edges:
            if valid_mask[edge[0]] and valid_mask[edge[1]]:
                fig.add_trace(go.Scatter3d(
                    x=[x[edge[0]], x[edge[1]]],
                    y=[y[edge[0]], y[edge[1]]],
                    z=[z[edge[0]], z[edge[1]]],
                    mode='lines',
                    line=dict(color='rgba(50, 50, 200, 0.7)', width=3),
                    hoverinfo='none'
                ))
        
        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[-1, 1], title='X'),
                yaxis=dict(range=[-1, 1], title='Y'),
                zaxis=dict(range=[-1, 1], title='Z'),
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            scene_camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            title="Human Pose Estimation"
        )
        
        return fig
    
    def _create_empty_confidence_figure(self):
        """Create an empty confidence history figure."""
        fig = go.Figure()
        
        fig.update_layout(
            xaxis=dict(title='Time'),
            yaxis=dict(title='Confidence', range=[0, 1]),
            margin=dict(l=10, r=10, t=10, b=10),
            height=150,
            showlegend=False
        )
        
        return fig
    
    def _update_confidence_figure(self):
        """Update the confidence history plot."""
        fig = go.Figure()
        
        if len(self.confidence_history) > 0:
            x = list(range(len(self.confidence_history)))
            
            # Confidence line
            fig.add_trace(go.Scatter(
                x=x,
                y=list(self.confidence_history),
                mode='lines',
                line=dict(color='rgba(0, 200, 100, 0.8)', width=2),
                name='Confidence'
            ))
            
            # People detected
            fig.add_trace(go.Scatter(
                x=x,
                y=list(self.detected_people_history),
                mode='lines',
                line=dict(color='rgba(200, 100, 0, 0.6)', width=2, dash='dash'),
                name='People'
            ))
        
        fig.update_layout(
            xaxis=dict(title='Time', showticklabels=False),
            yaxis=dict(title='Value', range=[0, 1.1]),
            margin=dict(l=10, r=10, t=10, b=10),
            height=150,
            showlegend=False
        )
        
        return fig
    
    def _create_empty_csi_figure(self):
        """Create an empty CSI visualization figure."""
        fig = go.Figure()
        
        fig.update_layout(
            xaxis=dict(title='Subcarrier'),
            yaxis=dict(title='Amplitude'),
            margin=dict(l=10, r=10, t=10, b=10),
            height=200,
            showlegend=False
        )
        
        return fig
    
    def _update_csi_figure(self, csi_data):
        """Update the CSI visualization with new data."""
        fig = go.Figure()
        
        if csi_data is not None:
            amplitude, phase = csi_data
            
            # For simplicity, just show one TX-RX pair
            tx_idx, rx_idx = 0, 0
            amplitude_data = amplitude[tx_idx, rx_idx]
            phase_data = phase[tx_idx, rx_idx]
            
            # X-axis is subcarrier index
            x = np.arange(len(amplitude_data))
            
            # Amplitude plot
            fig.add_trace(go.Scatter(
                x=x,
                y=amplitude_data,
                mode='lines',
                line=dict(color='rgba(0, 100, 200, 0.8)', width=2),
                name='Amplitude'
            ))
            
            # Phase plot (on secondary y-axis)
            fig.add_trace(go.Scatter(
                x=x,
                y=phase_data,
                mode='lines',
                line=dict(color='rgba(200, 0, 100, 0.6)', width=2),
                name='Phase',
                yaxis='y2'
            ))
        
        fig.update_layout(
            xaxis=dict(title='Subcarrier'),
            yaxis=dict(title='Amplitude'),
            yaxis2=dict(
                title='Phase',
                overlaying='y',
                side='right',
                range=[-np.pi, np.pi]
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            height=200,
            showlegend=True,
            legend=dict(orientation='h', y=1.1)
        )
        
        return fig
    
    def update_data(self, pose_data=None, confidence_data=None, csi_data=None):
        """Update the dashboard with new data.
        
        Args:
            pose_data: Dictionary containing keypoints and confidence
            confidence_data: Array of confidence values
            csi_data: Tuple of (amplitude, phase) arrays
        """
        with self.data_lock:
            if pose_data is not None:
                self.pose_data = pose_data
            
            if confidence_data is not None:
                self.confidence_data = confidence_data
                
            if csi_data is not None:
                self.csi_data = csi_data
    
    def run(self, debug=False, port=8050):
        """Run the dashboard server.
        
        Args:
            debug: Enable debug mode for Dash
            port: Port to run the server on
        """
        self.logger.info(f"Starting dashboard on port {port}")
        self.app.run_server(debug=debug, port=port)

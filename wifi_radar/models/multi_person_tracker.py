"""
ID: WR-MODEL-MPT-001
Purpose: Detect up to N people from a single CSI measurement using a
         dedicated multi-person hypothesis network, then track identities
         across frames with greedy nearest-centroid matching.

Architecture:
    DualBranchEncoder → BiLSTM → N independent person-hypothesis heads
    Each head predicts: existence_score (1), keypoints (17×3), confidence (17).

Tracking:
    Greedy assignment   — O(N²) per frame, acceptable for N ≤ 8.
    ID retirement       — person ID retired after ``id_timeout_frames`` frames
                          without a matching detection.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrackedPerson:
    """State of one tracked person."""
    person_id:         int
    keypoints:         np.ndarray           # (17, 3) latest estimate
    confidence:        np.ndarray           # (17,)
    centroid:          np.ndarray           # (3,)  weighted centroid
    last_seen_frame:   int
    frames_since_seen: int = 0
    active:            bool = True
    history:           List[np.ndarray] = field(default_factory=list)  # centroid trail


# ─────────────────────────────────────────────────────────────────────────────
# Neural network
# ─────────────────────────────────────────────────────────────────────────────

class MultiPersonPoseEstimator(nn.Module):
    """Multi-hypothesis pose estimator.

    Output heads are independent so the network can specialise each head for
    a different person without interference.

    Args:
        input_dim:     Feature dimension from DualBranchEncoder (default 256).
        hidden_dim:    LSTM hidden size (default 512).
        num_keypoints: Number of 3-D body keypoints (default 17, COCO format).
        max_people:    Maximum simultaneous people to predict (default 4).
        num_lstm_layers: Depth of BiLSTM backbone (default 2).
    """

    def __init__(
        self,
        input_dim:       int = 256,
        hidden_dim:      int = 512,
        num_keypoints:   int = 17,
        max_people:      int = 4,
        num_lstm_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim     = input_dim
        self.hidden_dim    = hidden_dim
        self.num_keypoints = num_keypoints
        self.max_people    = max_people

        # Shared temporal backbone (bidirectional for richer context)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_dim * 2  # bidirectional doubles output dim

        # Shared projection
        self.shared_fc = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Per-person hypothesis heads
        head_out = 1 + num_keypoints * 3 + num_keypoints  # exist + kp_coords + kp_conf
        self.person_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, head_out),
            )
            for _ in range(max_people)
        ])

        self._init_weights()

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[List[Dict[str, torch.Tensor]], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x:      (batch, input_dim) or (batch, seq_len, input_dim).
            hidden: Optional LSTM hidden state for stateful inference.

        Returns:
            people: List of dicts per hypothesis, each with keys:
                      ``existence``  (batch,)    probability person i is present
                      ``keypoints``  (batch, 17, 3)
                      ``confidence`` (batch, 17)
            hidden: Updated LSTM state.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)   # (batch, 1, input_dim)

        lstm_out, hidden = self.lstm(x, hidden)
        features = lstm_out[:, -1]           # (batch, hidden*2)
        shared   = self.shared_fc(features)  # (batch, hidden)

        people: List[Dict[str, torch.Tensor]] = []
        for head in self.person_heads:
            raw = head(shared)               # (batch, head_out)
            existence  = torch.sigmoid(raw[:, 0])
            kp_coords  = raw[:, 1 : 1 + self.num_keypoints * 3]
            kp_coords  = kp_coords.view(-1, self.num_keypoints, 3)
            kp_conf    = torch.sigmoid(raw[:, 1 + self.num_keypoints * 3:])
            people.append({"existence": existence, "keypoints": kp_coords, "confidence": kp_conf})

        return people, hidden

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, p in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(p)
                    elif "bias" in name:
                        nn.init.constant_(p, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────────────────────────────────────

class MultiPersonTracker:
    """Maintains consistent person IDs across frames.

    Works with ANY source of per-frame people dicts
    (``MultiPersonPoseEstimator`` or legacy ``PoseEstimator.detect_people``).

    Args:
        max_people:          Hard cap on simultaneous tracks (default 4).
        existence_threshold: Minimum existence score to accept a detection (default 0.40).
        max_match_distance:  Max normalised-space distance for ID assignment (default 0.40).
        id_timeout_frames:   Frames without match before retiring a track (default 10).
        confidence_threshold: Min keypoint confidence for centroid computation (default 0.3).
    """

    def __init__(
        self,
        max_people:           int   = 4,
        existence_threshold:  float = 0.40,
        max_match_distance:   float = 0.40,
        id_timeout_frames:    int   = 10,
        confidence_threshold: float = 0.30,
    ) -> None:
        self.max_people          = max_people
        self.existence_threshold = existence_threshold
        self.max_match_dist      = max_match_distance
        self.id_timeout_frames   = id_timeout_frames
        self.conf_threshold      = confidence_threshold

        self._tracks:    Dict[int, TrackedPerson] = {}
        self._next_id:   int = 0
        self._frame_idx: int = 0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def update(
        self, people: List[Dict[str, Any]], frame_id: Optional[int] = None
    ) -> List[TrackedPerson]:
        """Assign identities to a list of per-frame detections.

        Args:
            people:   List of dicts, each containing at minimum:
                        ``keypoints``  (17, 3) numpy array or Tensor
                        ``confidence`` (17,)   numpy array or Tensor
                        ``existence``  float or scalar Tensor (optional; assumed 1.0 if absent)
            frame_id: Monotonic integer; uses internal counter when None.

        Returns:
            List of active :class:`TrackedPerson` instances (existence-filtered,
            sorted by person_id).
        """
        self._frame_idx = frame_id if frame_id is not None else self._frame_idx + 1

        # ── 1. Filter detections by existence score ──────────────────────
        detections = []
        for p in people:
            existence = float(
                p["existence"].item() if hasattr(p.get("existence", 1.0), "item")
                else p.get("existence", 1.0)
            )
            if existence < self.existence_threshold:
                continue
            kp   = self._to_numpy(p["keypoints"])
            conf = self._to_numpy(p["confidence"])
            if kp.ndim == 3:
                kp   = kp[0]   # strip batch dim if present
                conf = conf[0]
            centroid = self._weighted_centroid(kp, conf)
            if np.any(np.isnan(centroid)):
                continue
            detections.append({"keypoints": kp, "confidence": conf,
                                "centroid": centroid, "existence": existence})

        # ── 2. Match detections to existing tracks ────────────────────────
        active_track_ids = [tid for tid, t in self._tracks.items() if t.active]
        unmatched_dets   = list(range(len(detections)))
        matched_pairs: List[Tuple[int, int]] = []  # (track_id, det_idx)

        if active_track_ids and detections:
            matched_pairs, unmatched_dets = self._greedy_match(
                active_track_ids, detections
            )

        # ── 3. Update matched tracks ──────────────────────────────────────
        for tid, det_idx in matched_pairs:
            det = detections[det_idx]
            t   = self._tracks[tid]
            t.keypoints         = det["keypoints"]
            t.confidence        = det["confidence"]
            t.centroid          = det["centroid"]
            t.last_seen_frame   = self._frame_idx
            t.frames_since_seen = 0
            t.history.append(det["centroid"].copy())
            if len(t.history) > 60:
                t.history = t.history[-60:]

        # ── 4. Create new tracks for unmatched detections ─────────────────
        for det_idx in unmatched_dets:
            if len(self._tracks) >= self.max_people:
                break
            det = detections[det_idx]
            new_id = self._next_id
            self._next_id += 1
            self._tracks[new_id] = TrackedPerson(
                person_id=new_id,
                keypoints=det["keypoints"],
                confidence=det["confidence"],
                centroid=det["centroid"],
                last_seen_frame=self._frame_idx,
                history=[det["centroid"].copy()],
            )
            logger.debug("New person track: id=%d", new_id)

        # ── 5. Age out lost tracks ────────────────────────────────────────
        for tid in list(self._tracks.keys()):
            t = self._tracks[tid]
            if t.last_seen_frame < self._frame_idx:
                t.frames_since_seen += 1
            if t.frames_since_seen > self.id_timeout_frames:
                t.active = False
                logger.debug("Retired track id=%d", tid)

        return sorted(
            [t for t in self._tracks.values() if t.active],
            key=lambda t: t.person_id,
        )

    @property
    def active_count(self) -> int:
        return sum(1 for t in self._tracks.values() if t.active)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id   = 0
        self._frame_idx = 0

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _greedy_match(
        self,
        active_track_ids: List[int],
        detections: List[Dict],
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Greedily assign detections to tracks by minimum centroid distance."""
        tracks   = [(tid, self._tracks[tid].centroid) for tid in active_track_ids]
        n_tracks = len(tracks)
        n_dets   = len(detections)

        # Distance matrix  (n_tracks × n_dets)
        dist_mat = np.full((n_tracks, n_dets), fill_value=np.inf)
        for i, (tid, tc) in enumerate(tracks):
            for j, det in enumerate(detections):
                dist_mat[i, j] = float(np.linalg.norm(tc - det["centroid"]))

        matched: List[Tuple[int, int]] = []
        used_dets: set = set()
        used_tracks: set = set()

        # Repeat: pick global minimum until no valid pair remains
        while True:
            if dist_mat.size == 0:
                break
            idx = int(np.argmin(dist_mat))
            row, col = divmod(idx, n_dets)
            if dist_mat[row, col] > self.max_match_dist:
                break
            track_id = active_track_ids[row]
            matched.append((track_id, col))
            used_tracks.add(row)
            used_dets.add(col)
            dist_mat[row, :] = np.inf
            dist_mat[:, col] = np.inf

        unmatched_dets = [j for j in range(n_dets) if j not in used_dets]
        return matched, unmatched_dets

    @staticmethod
    def _weighted_centroid(kp: np.ndarray, conf: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        mask = conf > threshold
        if not np.any(mask):
            return np.zeros(3, dtype=np.float32)
        w = conf[mask]
        return (kp[mask] * w[:, None]).sum(axis=0) / w.sum()

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

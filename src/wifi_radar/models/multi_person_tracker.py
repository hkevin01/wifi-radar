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
    """State snapshot for one actively tracked person.

    ID: WR-MODEL-MPT-TPERSON-001
    Requirement: Store keypoints, confidence, centroid, and lifecycle metadata
                 for a single person track across consecutive inference frames.
    Purpose: Provide a stable, ID-indexed view of an individual so downstream
             consumers (fall detector, gait analyzer) can query consistent state.
    Rationale: dataclass with mutable defaults (history list) avoids the
               overhead of a full ORM or named-tuple for a hot-path data object.
    Inputs:
        person_id         — int: unique monotonic track identifier.
        keypoints         — (17,3) float32: latest COCO-17 keypoint estimate.
        confidence        — (17,) float32: per-keypoint confidence [0,1].
        centroid          — (3,) float32: confidence-weighted body centroid.
        last_seen_frame   — int: frame index of most recent match.
        frames_since_seen — int: frames elapsed without a matching detection.
        active            — bool: False once id_timeout_frames is exceeded.
        history           — List of centroid arrays (max 60, rolling window).
    Outputs:
        N/A — data container only.
    Preconditions:
        Created by MultiPersonTracker.update() only.
    References:
        MultiPersonTracker._greedy_match; COCO-17 keypoint format.
    """
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
    """Multi-hypothesis BiLSTM pose estimator for up to N simultaneous people.

    ID: WR-MODEL-MPT-MPPE-CLASS-001
    Requirement: Accept a DualBranchEncoder feature vector and output per-person
                 existence probability, 17 COCO keypoints, and keypoint confidence
                 for up to max_people independent hypothesis heads.
    Purpose: Enable multi-person tracking from a single CSI measurement by
             predicting all hypotheses simultaneously in one forward pass.
    Rationale: Independent per-person heads prevent cross-person interference;
               a shared BiLSTM backbone provides temporal context efficiently.
    Inputs:
        input_dim       — int: feature dimension from DualBranchEncoder (default 256).
        hidden_dim      — int: BiLSTM hidden size (default 512).
        num_keypoints   — int: number of 3-D body keypoints (default 17, COCO).
        max_people      — int: maximum simultaneous person hypotheses (default 4).
        num_lstm_layers — int: depth of BiLSTM backbone (default 2).
    Outputs:
        List of max_people dicts per forward pass; each dict has:
            'existence'  — (batch,) sigmoid probability.
            'keypoints'  — (batch, num_keypoints, 3) raw coordinates.
            'confidence' — (batch, num_keypoints) sigmoid confidence.
    Preconditions:
        Encoder must produce (batch, input_dim) or (batch, seq_len, input_dim).
    Constraints:
        BiLSTM output is 2*hidden_dim; dropout disabled for single-layer LSTM.
    References:
        DualBranchEncoder; COCO-17 keypoints; nn.LSTM bidirectional.
    """

    def __init__(
        self,
        input_dim:       int = 256,
        hidden_dim:      int = 512,
        num_keypoints:   int = 17,
        max_people:      int = 4,
        num_lstm_layers: int = 2,
    ) -> None:
        """Build BiLSTM backbone, shared projection, and per-person output heads.

        ID: WR-MODEL-MPT-MPPE-INIT-001
        Requirement: Construct all nn.Module submodules, store hyperparameters,
                     and call _init_weights() to initialise parameters.
        Purpose: Separate construction from activation so the model can be
                 serialised (torch.save) before any forward pass.
        Rationale: ModuleList for person_heads ensures PyTorch registers all head
                   parameters for gradient computation and checkpointing.
        Inputs:
            input_dim       — int: encoder output feature size.
            hidden_dim      — int: BiLSTM hidden units per direction.
            num_keypoints   — int: COCO keypoint count.
            max_people      — int: number of independent hypothesis heads.
            num_lstm_layers — int: LSTM depth; dropout applied if > 1.
        Outputs:
            None — initialises self.
        Preconditions:
            torch and torch.nn must be available.
        Postconditions:
            All submodules are registered; _init_weights() has been called.
        Assumptions:
            input_dim matches DualBranchEncoder output dimension.
        Side Effects:
            Allocates all weight tensors in CPU memory.
        Failure Modes:
            hidden_dim <= 0: nn.LSTM raises ValueError.
        Error Handling:
            None; constructor relies on PyTorch validation.
        Constraints:
            Dropout is 0.0 when num_lstm_layers==1 (PyTorch requirement).
        Verification:
            Unit test: instantiate; call forward with random input; check output shapes.
        References:
            nn.LSTM; nn.ModuleList; WR-MODEL-MPT-MPPE-CLASS-001.
        """
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
        """Run one forward pass through the BiLSTM backbone and all hypothesis heads.

        ID: WR-MODEL-MPT-MPPE-FWD-001
        Requirement: Accept encoder features, run through BiLSTM and shared
                     projection, then produce existence/keypoints/confidence for
                     each person hypothesis head.
        Purpose: Produce all max_people pose hypotheses in a single batched pass
                 for downstream tracker assignment.
        Rationale: Unsqueeze for 2-D input ensures the LSTM always receives a
                   sequence; using the last time-step output captures the most
                   recent context.
        Inputs:
            x      — Tensor (batch, input_dim) or (batch, seq_len, input_dim).
            hidden — Optional LSTM hidden state for stateful inference.
        Outputs:
            people — List[Dict] of length max_people; each dict:
                'existence'  (batch,) — sigmoid person presence probability.
                'keypoints'  (batch, num_keypoints, 3) — raw 3-D coords.
                'confidence' (batch, num_keypoints) — sigmoid per-keypoint score.
            hidden — updated LSTM hidden state tuple.
        Preconditions:
            x.shape[-1] == self.input_dim.
        Postconditions:
            All existence and confidence values in [0, 1] (sigmoid applied).
        Assumptions:
            Batch size >= 1.
        Side Effects:
            Updates LSTM hidden state if called statelessly (hidden=None).
        Failure Modes:
            Wrong input_dim: LSTM raises shape mismatch RuntimeError.
        Error Handling:
            None; relies on PyTorch runtime checks.
        Constraints:
            Output keypoints are raw (not clamped); normalisation is caller's
            responsibility.
        Verification:
            Unit test: batch=2, seq_len=5; assert len(people)==max_people.
        References:
            BiLSTM last time-step; nn.Sigmoid via torch.sigmoid.
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
        """Initialise Linear and LSTM weights with orthogonal/Kaiming strategies.

        ID: WR-MODEL-MPT-MPPE-INITW-001
        Requirement: Apply Kaiming normal initialisation to Linear weights and
                     orthogonal initialisation to LSTM recurrent weights.
        Purpose: Improve training stability by providing well-conditioned initial
                 weight values rather than relying on PyTorch defaults.
        Rationale: Kaiming normal is standard for ReLU activations; orthogonal
                   initialisation prevents vanishing/exploding gradients in LSTMs.
        Inputs:
            None — iterates self.modules().
        Outputs:
            None — modifies weights in-place.
        Preconditions:
            All submodules must be registered (called after super().__init__() and
            module construction).
        Postconditions:
            All Linear.weight are Kaiming-normal; all LSTM recurrent weights are
            orthogonal; all biases are zero.
        Assumptions:
            nn.init functions are available.
        Side Effects:
            Modifies all Linear and LSTM parameter tensors in-place.
        Failure Modes:
            None expected for standard PyTorch module types.
        Error Handling:
            None required.
        Constraints:
            Must be called after all submodules are added to self.
        Verification:
            Unit test: construct model; assert no NaN in parameters.
        References:
            nn.init.kaiming_normal_; nn.init.orthogonal_; WR-MODEL-MPT-MPPE-INIT-001.
        """
        def _init(m):
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
        self.apply(_init)


# ─────────────────────────────────────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────────────────────────────────────

class MultiPersonTracker:
    """Maintains consistent person IDs across inference frames via greedy matching.

    ID: WR-MODEL-MPT-TRACKER-CLASS-001
    Requirement: Accept per-frame people dicts, assign stable integer IDs using
                 nearest-centroid greedy matching, and retire tracks that have
                 not been matched for id_timeout_frames consecutive frames.
    Purpose: Provide downstream consumers (fall detector, gait analyzer) with
             frame-to-frame consistent person identities.
    Rationale: Greedy O(N^2) matching is sufficient for N<=8 people; nearest-
               centroid is robust to partial occlusion and keypoint dropout.
    Inputs:
        max_people          — int: hard cap on simultaneous tracks.
        existence_threshold — float: min existence score to accept a detection.
        max_match_distance  — float: max normalised-space distance for ID assignment.
        id_timeout_frames   — int: frames without match before retiring a track.
        confidence_threshold — float: min keypoint confidence for centroid computation.
    Outputs:
        update() returns List[TrackedPerson] sorted by person_id.
    Preconditions:
        Works with any source providing keypoints/confidence/existence dicts.
    Assumptions:
        Keypoint coordinates are in normalised [-1, 1] space.
    Constraints:
        Not thread-safe; wrap update() in a lock if called from multiple threads.
    References:
        TrackedPerson; _greedy_match; WR-MODEL-MPT-001 module docstring.
    """

    def __init__(
        self,
        max_people:           int   = 4,
        existence_threshold:  float = 0.40,
        max_match_distance:   float = 0.40,
        id_timeout_frames:    int   = 10,
        confidence_threshold: float = 0.30,
    ) -> None:
        """Initialise tracker state and configuration parameters.

        ID: WR-MODEL-MPT-TRACKER-INIT-001
        Requirement: Store all configuration parameters and initialise the
                     empty track registry, frame counter, and ID counter.
        Purpose: Prepare the tracker for the first update() call without
                 requiring any initial frame data.
        Rationale: Using a dict (track_id -> TrackedPerson) gives O(1)
                   lookup during the greedy match loop.
        Inputs:
            max_people           — int: hard cap on simultaneous tracks.
            existence_threshold  — float [0,1]: minimum existence score.
            max_match_distance   — float: max centroid distance for ID assignment.
            id_timeout_frames    — int: frames before retiring an unmatched track.
            confidence_threshold — float [0,1]: keypoint confidence floor.
        Outputs:
            None — initialises self.
        Preconditions:
            None.
        Postconditions:
            self._tracks == {}; self._next_id == 0; self._frame_idx == 0.
        Assumptions:
            update() will be called at a consistent frame rate.
        Side Effects:
            None — pure data initialisation.
        Failure Modes:
            None expected at construction time.
        Error Handling:
            None required.
        Constraints:
            max_people <= 8 recommended for O(N^2) matching performance.
        Verification:
            Unit test: construct; assert _tracks == {} and active_count == 0.
        References:
            _greedy_match; TrackedPerson; WR-MODEL-MPT-TRACKER-CLASS-001.
        """
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

        ID: WR-MODEL-MPT-TRACKER-UPDATE-001
        Requirement: Filter detections by existence score, match to existing
                     tracks via greedy centroid distance, create new tracks for
                     unmatched detections, age out lost tracks, and return active
                     TrackedPerson list sorted by person_id.
        Purpose: Maintain stable identity assignments across frames so downstream
                 consumers see consistent person IDs rather than re-indexed lists.
        Rationale: Five-stage pipeline (filter, match, update, create, age) is the
                   minimal correct implementation of nearest-centroid tracking.
        Inputs:
            people   — List[Dict]: each must have 'keypoints' (17,3), 'confidence' (17,);
                       'existence' float is optional (assumed 1.0 if absent).
            frame_id — Optional[int]: monotonic frame counter; auto-increments if None.
        Outputs:
            List[TrackedPerson] — active tracks sorted by person_id.
        Preconditions:
            Keypoints and confidence must be numpy arrays or PyTorch Tensors.
        Postconditions:
            Tracks with frames_since_seen > id_timeout_frames have active=False.
        Assumptions:
            Called once per inference frame at a consistent rate.
        Side Effects:
            Mutates self._tracks (updates existing, creates new, retires old).
            Increments self._frame_idx.
        Failure Modes:
            NaN centroid (all-zero confidence): detection silently skipped.
        Error Handling:
            np.any(np.isnan(centroid)) guard skips malformed detections.
        Constraints:
            Not thread-safe; not idempotent (each call advances frame state).
        Verification:
            Unit test: feed 3 frames with 2 detections; assert track IDs stable.
        References:
            _greedy_match; _weighted_centroid; TrackedPerson; WR-MODEL-MPT-001.
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
        """Number of currently active (not retired) person tracks.

        ID: WR-MODEL-MPT-TRACKER-ACTCNT-001
        Requirement: Return the count of TrackedPerson instances with active==True.
        Purpose: Provide a lightweight way to query occupant count without
                 iterating the full track list externally.
        Rationale: Property avoids accidentally mutating the tracks dict.
        Inputs:
            None.
        Outputs:
            int — count of active tracks in [0, max_people].
        Preconditions:
            None.
        Postconditions:
            Return value equals len([t for t in _tracks.values() if t.active]).
        Side Effects:
            None — read-only.
        Failure Modes:
            None.
        Verification:
            Unit test: after update() with 2 valid detections, assert active_count==2.
        References:
            TrackedPerson.active field; WR-MODEL-MPT-TRACKER-CLASS-001.
        """
        return sum(1 for t in self._tracks.values() if t.active)

    def reset(self) -> None:
        """Clear all tracks and reset the frame and ID counters.

        ID: WR-MODEL-MPT-TRACKER-RESET-001
        Requirement: Empty self._tracks and reset self._next_id and
                     self._frame_idx to 0.
        Purpose: Allow the tracker to be reused for a new recording session
                 without creating a new instance.
        Rationale: Resetting counters to 0 avoids integer overflow in very long
                   sessions and provides a clean baseline for new sessions.
        Inputs:
            None.
        Outputs:
            None.
        Preconditions:
            None.
        Postconditions:
            self._tracks == {}; self._next_id == 0; self._frame_idx == 0.
        Assumptions:
            Caller is responsible for ensuring no concurrent update() calls.
        Side Effects:
            Empties self._tracks.
            Sets self._next_id = 0 and self._frame_idx = 0.
        Failure Modes:
            None.
        Error Handling:
            None required.
        Constraints:
            Not thread-safe.
        Verification:
            Unit test: update with data; reset(); assert active_count==0.
        References:
            WR-MODEL-MPT-TRACKER-CLASS-001.
        """
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
        """Greedily assign detections to existing tracks by minimum centroid distance.

        ID: WR-MODEL-MPT-TRACKER-MATCH-001
        Requirement: Build an (n_tracks x n_dets) Euclidean distance matrix and
                     repeatedly select the global minimum pair until no pair
                     satisfies the max_match_dist threshold.
        Purpose: Resolve the assignment problem between existing tracks and new
                 detections with O(N^2) complexity suitable for N<=8.
        Rationale: Greedy global-minimum selection is a fast approximation of
                   the Hungarian algorithm; it is optimal when detections and
                   tracks are well-separated (no cluster ambiguity).
        Inputs:
            active_track_ids — List[int]: IDs of active tracks in self._tracks.
            detections       — List[Dict]: candidate dicts with 'centroid' key.
        Outputs:
            Tuple (matched_pairs, unmatched_det_indices) where:
                matched_pairs      — List[(track_id, det_idx)].
                unmatched_det_idx  — List[int] of unmatched detection indices.
        Preconditions:
            active_track_ids and detections are both non-empty.
        Postconditions:
            Each track ID and each detection index appears at most once in output.
        Assumptions:
            Centroids are 3-D float32 vectors in normalised [-1,1] space.
        Side Effects:
            Allocates and modifies a (n_tracks x n_dets) float matrix.
        Failure Modes:
            All distances exceed threshold: returns empty matched list.
        Error Handling:
            None; assumes valid centroid arrays.
        Constraints:
            O(N^2); N<=8 per max_people.
        Verification:
            Unit test: two tracks, two dets, near each other; assert 2 matched pairs.
        References:
            np.linalg.norm; greedy assignment; WR-MODEL-MPT-TRACKER-CLASS-001.
        """
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
        """Compute the confidence-weighted 3-D centroid of valid keypoints.

        ID: WR-MODEL-MPT-TRACKER-CENTROID-001
        Requirement: Select keypoints with confidence > threshold, compute the
                     weighted mean position, and return a (3,) float32 centroid.
        Purpose: Provide a robust body-centre estimate that degrades gracefully
                 when only a subset of keypoints is visible.
        Rationale: Confidence weighting suppresses noisy low-confidence keypoints;
                   fallback to zero vector avoids NaN propagation downstream.
        Inputs:
            kp        — (17, 3) float32: normalised keypoint coordinates.
            conf      — (17,) float32: per-keypoint confidence scores [0,1].
            threshold — float: minimum confidence to include (default 0.3).
        Outputs:
            (3,) float32 centroid; zero vector if no keypoint exceeds threshold.
        Preconditions:
            kp and conf have matching first dimension.
        Postconditions:
            Return value has no NaN when at least one valid keypoint exists.
        Assumptions:
            Coordinates are in [-1, 1] normalised space.
        Side Effects:
            None — pure function.
        Failure Modes:
            All confidence < threshold: returns zero vector (not NaN).
        Error Handling:
            `if not np.any(mask)` guard prevents division by zero.
        Constraints:
            Called once per detection per update() call; must be fast.
        Verification:
            Unit test: conf all 0.5, kp centred at origin; assert return ~[0,0,0].
        References:
            np.linalg.norm; confidence weighting; WR-MODEL-MPT-TRACKER-UPDATE-001.
        """
        if not np.any(mask):
            return np.zeros(3, dtype=np.float32)
        w = conf[mask]
        return (kp[mask] * w[:, None]).sum(axis=0) / w.sum()

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        """Convert a PyTorch Tensor or array-like to a float32 numpy array.

        ID: WR-MODEL-MPT-TRACKER-TONUMPY-001
        Requirement: Accept a torch.Tensor or any array-like object and return
                     a float32 numpy array with the same shape.
        Purpose: Provide a single conversion point so update() does not need
                 explicit tensor type checks scattered throughout the method.
        Rationale: detach().cpu() is required for gradient-tracked or GPU tensors;
                   np.asarray avoids a copy when input is already a numpy array.
        Inputs:
            x — torch.Tensor (any device, any grad state) or array-like.
        Outputs:
            float32 numpy array with the same shape as x.
        Preconditions:
            x must be convertible to a numeric array.
        Postconditions:
            Return dtype is np.float32.
        Assumptions:
            Tensor values are finite (no special handling for NaN/Inf).
        Side Effects:
            None — pure function; may allocate a new array.
        Failure Modes:
            Non-numeric input: np.asarray raises TypeError.
        Error Handling:
            None; caller (update()) is responsible for valid input.
        Constraints:
            None.
        Verification:
            Unit test: pass torch.tensor([1.0]); assert return is np.array([1.0]).
        References:
            torch.Tensor.detach().cpu().numpy(); np.asarray.
        """
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

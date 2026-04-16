"""
ID: WR-MODEL-POSE-001
Requirement: Accept a CSI embedding from DualBranchEncoder and regress 3-D COCO-17
             keypoint coordinates and per-keypoint confidence scores.
Purpose: Provides the primary human-pose output for downstream fall detection, gait
         analysis, and dashboard visualisation.
Architecture:
    FC → Dropout → FC → (optional LSTM for temporal smoothing) → keypoint head + confidence head
Assumptions:
    - Input features are (batch_size, 256) embeddings from DualBranchEncoder.
    - Keypoint coordinates are in normalised space [-1, 1] on each axis.
    - Confidence scores are in [0, 1] (sigmoid output).
References: COCO 17-point skeleton, OpenPose (CMU, 2019)
"""
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseEstimator(nn.Module):
    """Single-person pose estimator with optional LSTM temporal smoothing.

    Processes either a single feature vector (batch_size, input_dim) or a
    sequence (batch_size, seq_len, input_dim).  In sequence mode an LSTM
    provides temporal consistency across frames.

    Args:
        input_dim:      Feature vector size from DualBranchEncoder (default 256).
        hidden_dim:     Width of the shared fully-connected backbone (default 512).
        num_keypoints:  Number of output keypoints — COCO-17 by default.
        output_dim:     Coordinate dimension per keypoint (3 for x/y/z; default 3).
    """

    def __init__(self, input_dim=256, hidden_dim=512, num_keypoints=17, output_dim=3):
        """Build all sub-modules.

        Args:
            input_dim:      Embedding size produced by DualBranchEncoder.
            hidden_dim:     Width of the two shared FC layers and LSTM hidden state.
            num_keypoints:  Number of body keypoints to regress (COCO-17 default).
            output_dim:     Spatial dimensions per keypoint (3 → x, y, z).

        Side Effects:
            Allocates Linear, LSTM, and Dropout parameter tensors on CPU.
            Calls initialize_weights() immediately after construction.
        """
        super(PoseEstimator, self).__init__()
        self.logger = logging.getLogger("PoseEstimator")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_keypoints = num_keypoints  # Number of keypoints in human pose (COCO-17)
        self.output_dim = output_dim        # 3D coordinates (x, y, z)

        # ── Shared backbone ───────────────────────────────────────────────
        # Two FC layers extract a high-level representation shared by both heads.
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # ── Output heads ──────────────────────────────────────────────────
        # Keypoint head: predicts (num_keypoints × output_dim) coordinate values.
        self.keypoint_fc = nn.Linear(hidden_dim, num_keypoints * output_dim)
        # Confidence head: predicts a score in [0,1] for each keypoint via sigmoid.
        self.confidence_fc = nn.Linear(hidden_dim, num_keypoints)

        # Dropout regularises the shared backbone (applied after fc1).
        self.dropout = nn.Dropout(0.3)

        # ── LSTM for temporal consistency ────────────────────────────────
        # When processing a sequence of frames the LSTM smooths keypoint predictions
        # across time.  For single-frame inference (most common) the LSTM is bypassed.
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.initialize_weights()

    def forward(self, x, hidden=None):
        """Regress keypoint positions and confidence scores from CSI features.

        Args:
            x:      Feature tensor.  Either:
                      • (batch_size, input_dim)                  — single-frame mode, LSTM skipped.
                      • (batch_size, sequence_length, input_dim) — sequential mode, LSTM applied.
            hidden: Optional LSTM hidden state ``(h_n, c_n)`` for stateful inference.
                    Pass ``None`` to reset the hidden state on each call.

        Returns:
            Tuple (keypoints, confidence, hidden):
              keypoints:  (batch_size, num_keypoints, output_dim) — normalised 3-D coordinates.
              confidence: (batch_size, num_keypoints)             — per-keypoint score in [0, 1].
              hidden:     Updated LSTM state or ``None`` in single-frame mode.

        Side Effects:
            None — pure functional forward pass.
        """
        batch_size = x.shape[0]

        # Determine whether input is a time sequence (3-D) or a single frame (2-D).
        is_sequence = len(x.shape) > 2

        if is_sequence:
            sequence_length = x.shape[1]

            # Process every timestep through the shared FC backbone in one batch
            # by temporarily collapsing the batch and sequence dimensions.
            x_reshaped = x.reshape(-1, self.input_dim)
            features = F.relu(self.fc1(x_reshaped))
            features = self.dropout(features)
            features = F.relu(self.fc2(features))

            # Restore the sequence shape so the LSTM can process temporal context.
            features = features.reshape(batch_size, sequence_length, self.hidden_dim)

            # LSTM returns features for every timestep; we only need the final one
            # for the output heads (the LSTM has already integrated temporal context).
            features, hidden = self.lstm(features, hidden)
            features = features[:, -1]   # (batch, hidden_dim)
        else:
            # Single-frame: skip the LSTM entirely for lower latency.
            features = F.relu(self.fc1(x))
            features = self.dropout(features)
            features = F.relu(self.fc2(features))

        # ── Output heads ──────────────────────────────────────────────────
        # Keypoint positions: flatten then reshape to (batch, num_kp, 3).
        keypoints = self.keypoint_fc(features)
        keypoints = keypoints.reshape(batch_size, self.num_keypoints, self.output_dim)

        # Confidence: sigmoid squashes raw logits to [0, 1].
        confidence = torch.sigmoid(self.confidence_fc(features))

        return keypoints, confidence, hidden

    def initialize_weights(self):
        """Apply Kaiming initialisation to all Linear layers.

        fan_in mode is used here (vs. fan_out in the CNN encoder) because the
        fully connected layers do not have a spatial fan-out component and the
        activations are consumed by a downstream ReLU on the receiving side.

        Side Effects:
            Modifies all Linear weight and bias tensors in-place.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def detect_people(self, keypoints, confidence, threshold=0.5):
        """Separate per-batch detections into a list of individual person dicts.

        This is a simplified single-person heuristic used for the legacy code path.
        For multi-person scenarios use ``MultiPersonPoseEstimator`` + ``MultiPersonTracker``.

        Algorithm:
            For each batch element, filter keypoints whose confidence exceeds
            ``threshold``.  If at least 30 % of keypoints pass the filter the
            person is considered detected and appended to the output list.

        Args:
            keypoints:  (batch_size, num_keypoints, output_dim) tensor.
            confidence: (batch_size, num_keypoints) tensor.
            threshold:  Confidence cutoff in [0, 1].  Keypoints below this are
                        replaced with NaN in the output.

        Returns:
            List of dicts, one per batch element that passes the keypoint filter.
            Each dict has keys:
              ``keypoints``  — (num_keypoints, output_dim) numpy array (low-conf → NaN).
              ``confidence`` — (num_keypoints,) numpy array.

        Side Effects:
            Detaches tensors and moves them to CPU; does not modify the originals.
        """
        batch_size = keypoints.shape[0]
        people = []

        for b in range(batch_size):
            batch_keypoints = keypoints[b].detach().cpu().numpy()
            batch_confidence = confidence[b].detach().cpu().numpy()

            # Mask identifies reliable keypoints above the confidence threshold.
            valid_mask = batch_confidence > threshold

            person_keypoints = batch_keypoints.copy()
            person_confidence = batch_confidence.copy()

            # Replace sub-threshold coordinates with NaN so downstream consumers
            # can detect unreliable joints without a separate boolean mask.
            person_keypoints[~valid_mask] = np.nan

            # Require at least 30 % of keypoints to avoid spurious detections
            # caused by noisy CSI frames with pervasive low confidence.
            if np.sum(valid_mask) > self.num_keypoints * 0.3:
                people.append(
                    {"keypoints": person_keypoints, "confidence": person_confidence}
                )

        return people

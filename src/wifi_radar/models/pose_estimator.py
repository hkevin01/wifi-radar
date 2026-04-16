"""
ID: WR-MODEL-POSE-001
Requirement: Accept a CSI embedding from DualBranchEncoder and regress 3-D COCO-17
             keypoint coordinates and per-keypoint confidence scores.
Purpose: Provides the primary human-pose output for downstream fall detection,
         gait analysis, and dashboard visualisation.
Rationale: A shared FC backbone feeding two specialised heads (keypoints + confidence)
           is parameter-efficient and allows the confidence head to learn independently
           of the coordinate regression objective.
Assumptions:
    Input features are (batch_size, 256) embeddings from DualBranchEncoder.
    Keypoint coordinates are in normalised space [-1, 1] on each axis.
    Confidence scores are in [0, 1] (sigmoid output).
Constraints: LSTM temporal smoothing requires >= 2 sequence frames to be useful.
References: COCO 17-point skeleton, OpenPose (CMU, 2019).
"""
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseEstimator(nn.Module):
    """Single-person pose estimator with optional LSTM temporal smoothing.

    ID: WR-MODEL-POSE-CLASS-001
    Requirement: Map a (batch, input_dim) feature vector or (batch, seq, input_dim)
                 sequence to COCO-17 keypoint coordinates and per-keypoint confidence.
    Purpose: Decode the compact CSI embedding from DualBranchEncoder into a
             structured human body pose usable by fall detection and gait analysis.
    Rationale: A two-head architecture (coordinate regression + confidence scoring)
               allows the model to express uncertainty per keypoint, enabling
               downstream consumers to filter unreliable joints.
    Inputs:
        x      — float tensor: (batch, input_dim) or (batch, seq_len, input_dim).
        hidden — Optional LSTM state tuple for stateful sequential inference.
    Outputs:
        Tuple (keypoints, confidence, hidden):
          keypoints  — (batch, num_keypoints, output_dim) normalised 3-D coords.
          confidence — (batch, num_keypoints) scores in [0, 1].
          hidden     — updated LSTM state or None.
    Preconditions:
        Input feature dimension must match self.input_dim (default 256).
    Postconditions:
        confidence values are in [0, 1] (sigmoid applied).
    Assumptions:
        DualBranchEncoder output_dim matches this model's input_dim.
    Constraints:
        LSTM path requires seq_len >= 2 for temporal smoothing to be effective.
    References:
        COCO 17-point skeleton; OpenPose (CMU, 2019).
    """

    def __init__(self, input_dim=256, hidden_dim=512, num_keypoints=17, output_dim=3):
        """Build all sub-modules and initialise weights.

        ID: WR-MODEL-POSE-INIT-001
        Requirement: Construct all Linear, LSTM, and Dropout sub-modules and
                     call initialize_weights() before the first forward pass.
        Purpose: Fully prepare the pose estimator for training or inference without
                 additional setup calls.
        Rationale: Calling initialize_weights() in __init__ ensures the model is
                   Kaiming-initialised even if the caller forgets to call it explicitly.
        Inputs:
            input_dim:      int >= 1: embedding size from DualBranchEncoder (default 256).
            hidden_dim:     int >= 1: width of shared FC layers and LSTM hidden state.
            num_keypoints:  int >= 1: body keypoints to regress (COCO-17 = 17).
            output_dim:     int >= 1: spatial dimensions per keypoint (3 for x,y,z).
        Outputs:
            None — initialises self.
        Preconditions:
            PyTorch nn.Module parent __init__() must succeed.
        Postconditions:
            All sub-module tensors allocated; weights Kaiming-initialised.
            self.input_dim, self.hidden_dim, self.num_keypoints, self.output_dim set.
        Assumptions:
            CPU allocation; caller moves model to target device after construction.
        Side Effects:
            Allocates Linear, LSTM, and Dropout parameter tensors on CPU.
            Calls initialize_weights() immediately.
        Failure Modes:
            ValueError if any dimension argument <= 0.
        Error Handling:
            PyTorch raises ValueError for invalid layer dimensions.
        Constraints:
            LSTM batch_first=True; LSTM hidden_size == hidden_dim.
        Verification:
            Unit test: construct defaults; call forward on (1, 256) tensor;
            assert keypoints shape (1, 17, 3) and confidence shape (1, 17).
        References:
            nn.Linear, nn.LSTM, nn.Dropout PyTorch documentation.
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

        ID: WR-MODEL-POSE-FWD-001
        Requirement: Accept a single-frame or sequential feature tensor and produce
                     normalised 3-D keypoint coordinates and per-keypoint confidence.
        Purpose: Decode the compact DualBranchEncoder embedding into a structured
                 pose representation used by fall detection, gait analysis, and the
                 RTMP visualiser.
        Rationale: Single-frame path skips the LSTM for lower latency in real-time
                   inference; sequential path applies LSTM for temporal consistency.
        Inputs:
            x      — float tensor:
                       (batch, input_dim)                  — single-frame mode.
                       (batch, sequence_length, input_dim) — sequential mode.
            hidden — Optional LSTM state (h_n, c_n); None resets the hidden state.
        Outputs:
            Tuple (keypoints, confidence, hidden):
              keypoints  — (batch, num_keypoints, output_dim) normalised 3-D coords.
              confidence — (batch, num_keypoints) scores in [0, 1].
              hidden     — updated LSTM state or None in single-frame mode.
        Preconditions:
            x.shape[-1] must equal self.input_dim.
            x must be on the same device as model parameters.
        Postconditions:
            confidence values are in [0, 1] (sigmoid applied).
            keypoints has shape (batch, self.num_keypoints, self.output_dim).
        Assumptions:
            batch_size >= 1; single-sample inference (batch=1) is valid.
        Side Effects:
            None — pure functional forward pass.
        Failure Modes:
            RuntimeError if x.shape[-1] != self.input_dim.
            Device mismatch raises RuntimeError from PyTorch.
        Error Handling:
            PyTorch raises descriptive RuntimeError on shape/device mismatches.
        Constraints:
            Dropout is active only during training (model.train() mode).
        Verification:
            Unit test: (batch=2, input_dim=256) tensor; assert output shapes
            are (2,17,3) for keypoints and (2,17) for confidence.
        References:
            COCO-17 keypoint format; nn.LSTM batch_first=True convention.
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

        ID: WR-MODEL-POSE-INIT-WEIGHTS-001
        Requirement: Set all Linear layer weights using Kaiming normal (fan_in)
                     initialisation and zero-initialise all biases.
        Purpose: Provide stable gradient flow at the start of training.
        Rationale: fan_in mode is preferred for fully-connected layers (vs. fan_out
                   used in CNN encoders) because FC layers lack a spatial fan-out.
        Inputs:
            None — operates on self.modules().
        Outputs:
            None — modifies parameter tensors in-place.
        Preconditions:
            All sub-modules must already be allocated (called after super().__init__()).
        Postconditions:
            All Linear weight tensors are Kaiming-normal (fan_in); biases are zero.
        Assumptions:
            All linear layers are followed by ReLU activations.
        Side Effects:
            Modifies all Linear weight and bias tensors in-place.
        Failure Modes:
            None expected; isinstance check handles all module types.
        Error Handling:
            Non-Linear sub-modules are silently skipped.
        Constraints:
            Must be called before training for reproducible initialisation.
        Verification:
            Unit test: call initialize_weights(); assert all Linear bias tensors are zero.
        References:
            He et al. (2015); nn.init.kaiming_normal_(mode='fan_in').
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def detect_people(self, keypoints, confidence, threshold=0.5):
        """Separate per-batch detections into a list of individual person dicts.

        ID: WR-MODEL-POSE-DETECT-001
        Requirement: Filter a batch of keypoint tensors by confidence threshold and
                     return a list of person dicts for downstream single-person consumers.
        Purpose: Provide a simple single-person heuristic for the legacy code path;
                 multi-person scenarios should use MultiPersonPoseEstimator instead.
        Rationale: Filtering sub-threshold keypoints with NaN allows downstream
                   consumers to detect unreliable joints without a separate boolean mask.
        Inputs:
            keypoints:  float tensor (batch_size, num_keypoints, output_dim).
            confidence: float tensor (batch_size, num_keypoints) in [0, 1].
            threshold:  float in (0, 1): confidence cutoff (default 0.5).
        Outputs:
            List of dicts, one per batch element passing the 30% keypoint filter.
            Each dict: {'keypoints': (num_kp, output_dim) numpy, low-conf -> NaN;
                        'confidence': (num_kp,) numpy array}.
        Preconditions:
            keypoints and confidence are tensors (not numpy); detach() is called.
        Postconditions:
            Returned keypoints have sub-threshold positions replaced with NaN.
            Only batch elements with >30% valid keypoints are included.
        Assumptions:
            Single-person inference; for multi-person use MultiPersonPoseEstimator.
        Side Effects:
            Detaches tensors and moves them to CPU; originals are not modified.
        Failure Modes:
            Empty batch returns an empty list.
            All-zero confidence returns empty list (nothing passes 30% filter).
        Error Handling:
            No exception handling; invalid input shapes raise PyTorch errors.
        Constraints:
            30% keypoint filter may be too lenient for high-noise CSI; tune threshold.
        Verification:
            Unit test: confidence=[1]*17 tensor; assert one person returned with
            all keypoints non-NaN.
        References:
            COCO-17 keypoint visibility convention; np.nan masking pattern.
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

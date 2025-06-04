import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseEstimator(nn.Module):
    """Estimates human pose from encoded CSI features."""

    def __init__(self, input_dim=256, hidden_dim=512, num_keypoints=17, output_dim=3):
        super(PoseEstimator, self).__init__()
        self.logger = logging.getLogger("PoseEstimator")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_keypoints = num_keypoints  # Number of keypoints in human pose
        self.output_dim = output_dim  # 3D coordinates (x, y, z)

        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layers for keypoints
        self.keypoint_fc = nn.Linear(hidden_dim, num_keypoints * output_dim)

        # Output layer for confidence
        self.confidence_fc = nn.Linear(hidden_dim, num_keypoints)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        # LSTM for temporal consistency
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Initialize weights
        self.initialize_weights()

    def forward(self, x, hidden=None):
        """
        Forward pass of the pose estimator.

        Args:
            x: Encoded features of shape (batch_size, input_dim) or
               (batch_size, sequence_length, input_dim) if using LSTM
            hidden: Optional hidden state for LSTM

        Returns:
            Tuple of (keypoints, confidence):
                keypoints: Tensor of shape (batch_size, num_keypoints, output_dim)
                confidence: Tensor of shape (batch_size, num_keypoints)
        """
        batch_size = x.shape[0]

        # Check if input is a sequence
        is_sequence = len(x.shape) > 2

        if is_sequence:
            # Process each timestep
            sequence_length = x.shape[1]

            # Reshape to process all timesteps at once
            x_reshaped = x.reshape(-1, self.input_dim)

            # Common feature extraction
            features = F.relu(self.fc1(x_reshaped))
            features = self.dropout(features)
            features = F.relu(self.fc2(features))

            # Reshape back to sequence
            features = features.reshape(batch_size, sequence_length, self.hidden_dim)

            # Apply LSTM for temporal consistency
            features, hidden = self.lstm(features, hidden)

            # Get final timestep for output
            features = features[:, -1]
        else:
            # Process single timestep
            features = F.relu(self.fc1(x))
            features = self.dropout(features)
            features = F.relu(self.fc2(features))

        # Predict keypoint positions
        keypoints = self.keypoint_fc(features)
        keypoints = keypoints.reshape(batch_size, self.num_keypoints, self.output_dim)

        # Predict confidence for each keypoint
        confidence = torch.sigmoid(self.confidence_fc(features))

        return keypoints, confidence, hidden

    def initialize_weights(self):
        """Initialize model weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def detect_people(self, keypoints, confidence, threshold=0.5):
        """
        Detect and separate multiple people from keypoints.

        Args:
            keypoints: Tensor of shape (batch_size, num_keypoints, output_dim)
            confidence: Tensor of shape (batch_size, num_keypoints)
            threshold: Confidence threshold for valid keypoints

        Returns:
            List of detected people, each containing keypoints and confidence
        """
        batch_size = keypoints.shape[0]
        people = []

        for b in range(batch_size):
            # Get keypoints and confidence for this batch
            batch_keypoints = keypoints[b].detach().cpu().numpy()
            batch_confidence = confidence[b].detach().cpu().numpy()

            # Filter low-confidence keypoints
            valid_mask = batch_confidence > threshold

            # Clustering approach to separate people
            # This is a simplified approach - in a real system, this would be more sophisticated
            person_keypoints = batch_keypoints.copy()
            person_confidence = batch_confidence.copy()

            # Set low-confidence keypoints to NaN
            person_keypoints[~valid_mask] = np.nan

            # Only add person if enough keypoints are detected
            if (
                np.sum(valid_mask) > self.num_keypoints * 0.3
            ):  # At least 30% of keypoints
                people.append(
                    {"keypoints": person_keypoints, "confidence": person_confidence}
                )

        return people

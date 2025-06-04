import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualBranchEncoder(nn.Module):
    """Dual-branch encoder for processing amplitude and phase CSI data."""

    def __init__(
        self, num_tx=3, num_rx=3, num_subcarriers=64, hidden_dim=128, output_dim=256
    ):
        super(DualBranchEncoder, self).__init__()
        self.logger = logging.getLogger("DualBranchEncoder")

        self.num_tx = num_tx
        self.num_rx = num_rx
        self.num_subcarriers = num_subcarriers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Amplitude branch
        self.amplitude_conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.amplitude_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.amplitude_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Phase branch (separate processing path)
        self.phase_conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.phase_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.phase_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Fusion layers
        self.fusion_conv = nn.Conv2d(128, 64, kernel_size=1)

        # Calculate flattened size after convolutions
        self.flattened_size = 64 * num_tx * num_rx

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Normalization layers
        self.amplitude_norm1 = nn.BatchNorm2d(16)
        self.amplitude_norm2 = nn.BatchNorm2d(32)
        self.amplitude_norm3 = nn.BatchNorm2d(64)

        self.phase_norm1 = nn.BatchNorm2d(16)
        self.phase_norm2 = nn.BatchNorm2d(32)
        self.phase_norm3 = nn.BatchNorm2d(64)

        self.fusion_norm = nn.BatchNorm2d(64)

    def forward(self, amplitude, phase):
        """
        Forward pass of the dual-branch encoder.

        Args:
            amplitude: Tensor of shape (batch_size, num_tx, num_rx, num_subcarriers)
            phase: Tensor of shape (batch_size, num_tx, num_rx, num_subcarriers)

        Returns:
            Encoded features of shape (batch_size, output_dim)
        """
        batch_size = amplitude.shape[0]

        # Reshape inputs to (batch_size, channels, height, width)
        # Where height=num_tx, width=num_rx*num_subcarriers
        amplitude = amplitude.view(
            batch_size, 1, self.num_tx, self.num_rx * self.num_subcarriers
        )
        phase = phase.view(
            batch_size, 1, self.num_tx, self.num_rx * self.num_subcarriers
        )

        # Amplitude branch
        a = F.relu(self.amplitude_norm1(self.amplitude_conv1(amplitude)))
        a = F.max_pool2d(a, 2)
        a = F.relu(self.amplitude_norm2(self.amplitude_conv2(a)))
        a = F.max_pool2d(a, 2)
        a = F.relu(self.amplitude_norm3(self.amplitude_conv3(a)))

        # Phase branch
        p = F.relu(self.phase_norm1(self.phase_conv1(phase)))
        p = F.max_pool2d(p, 2)
        p = F.relu(self.phase_norm2(self.phase_conv2(p)))
        p = F.max_pool2d(p, 2)
        p = F.relu(self.phase_norm3(self.phase_conv3(p)))

        # Fusion
        combined = torch.cat([a, p], dim=1)
        fused = F.relu(self.fusion_norm(self.fusion_conv(combined)))

        # Flatten and pass through fully connected layers
        flattened = fused.view(batch_size, -1)
        hidden = F.relu(self.fc1(flattened))
        output = self.fc2(hidden)

        return output

    def initialize_weights(self):
        """Initialize model weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

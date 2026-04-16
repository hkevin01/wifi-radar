"""
ID: WR-MODEL-ENC-001
Requirement: Encode a (num_tx × num_rx × num_subcarriers) CSI frame into a fixed-length
             feature vector used by downstream pose estimation and tracking modules.
Purpose: Dual-branch convolutional encoder that processes amplitude and phase tensors
         independently, fuses them at the channel level, then projects to a compact
         embedding.  Keeping the branches separate until fusion lets each branch learn
         modality-specific statistics before combining information.
Rationale: Amplitude carries multipath magnitude; phase carries time-of-flight shifts.
           A shared encoder would entangle these signals early and suppress phase-only
           or amplitude-only cues.  Adaptive average pooling instead of fixed max-pool
           makes the encoder agnostic to the subcarrier-count dimension at export time.
References: Widar3.0 (NSDI 2019), Wi-Pose (MobiSys 2022)
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualBranchEncoder(nn.Module):
    """Dual-branch convolutional encoder for amplitude and phase CSI tensors.

    Architecture::

        amplitude → Conv2d×3 (with BatchNorm + ReLU) ─┐
                                                        ├→ channel-concat → Conv1×1 → AdaptiveAvgPool → FC×2 → output
        phase     → Conv2d×3 (with BatchNorm + ReLU) ─┘

    Both branches share the same layer structure but have independent weights so
    each specialises on its modality before fusion.
    """

    def __init__(
        self, num_tx=3, num_rx=3, num_subcarriers=64, hidden_dim=128, output_dim=256
    ):
        """Build all sub-modules and store dimension hyper-parameters.

        Args:
            num_tx:          Number of transmitting antennas (spatial height).
            num_rx:          Number of receiving antennas (spatial height after reshape).
            num_subcarriers: OFDM subcarrier count (spatial width after reshape).
            hidden_dim:      Width of the first fully-connected projection layer.
            output_dim:      Embedding size returned by forward() (input to PoseEstimator).

        Side Effects:
            Allocates all Conv2d, BatchNorm2d, and Linear parameter tensors.
        """
        super(DualBranchEncoder, self).__init__()
        self.logger = logging.getLogger("DualBranchEncoder")

        # Store constructor params so model_io can serialize them for provenance.
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.num_subcarriers = num_subcarriers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # ── Amplitude branch ─────────────────────────────────────────────
        # Three 3×3 conv layers progressively expand channel depth (1→16→32→64).
        # padding=1 keeps the spatial height (num_tx) unchanged after each layer.
        self.amplitude_conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.amplitude_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.amplitude_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # ── Phase branch (separate processing path) ───────────────────────
        # Identical structure to the amplitude branch but with independent weights,
        # enabling the network to learn phase-specific features.
        self.phase_conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.phase_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.phase_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # ── Fusion layer ──────────────────────────────────────────────────
        # 1×1 convolution halves channels (128→64) after concatenation of the
        # two 64-channel branch outputs.  1×1 acts as a learned channel mixer
        # without spatial blurring.
        self.fusion_conv = nn.Conv2d(128, 64, kernel_size=1)

        # Flattened size is known at construction time; used to size fc1.
        # AdaptiveAvgPool2d below will output (num_tx, num_rx) regardless of
        # the subcarrier dimension, so this calculation is exact.
        self.flattened_size = 64 * num_tx * num_rx

        # ── Fully connected projection ────────────────────────────────────
        self.fc1 = nn.Linear(self.flattened_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # ── Batch normalisation (per branch + fusion) ─────────────────────
        # BatchNorm after each conv stabilises training and decouples layer
        # learning rates from the scale of activations.
        self.amplitude_norm1 = nn.BatchNorm2d(16)
        self.amplitude_norm2 = nn.BatchNorm2d(32)
        self.amplitude_norm3 = nn.BatchNorm2d(64)

        self.phase_norm1 = nn.BatchNorm2d(16)
        self.phase_norm2 = nn.BatchNorm2d(32)
        self.phase_norm3 = nn.BatchNorm2d(64)

        self.fusion_norm = nn.BatchNorm2d(64)

    def forward(self, amplitude, phase):
        """Encode a CSI frame into a compact feature embedding.

        Args:
            amplitude: Float tensor (batch_size, num_tx, num_rx, num_subcarriers).
                       Normalised CSI amplitude values.
            phase:     Float tensor (batch_size, num_tx, num_rx, num_subcarriers).
                       Unwrapped CSI phase values.

        Returns:
            Float tensor (batch_size, output_dim) — the embedded feature vector
            consumed by PoseEstimator and MultiPersonPoseEstimator.

        Preconditions:
            Both tensors must be on the same device as the model parameters.
        """
        batch_size = amplitude.shape[0]

        # Reshape (batch, num_tx, num_rx, num_sub) → (batch, 1, num_tx, num_rx*num_sub)
        # so the 2-D convolutions treat the num_tx dimension as spatial height and
        # the flattened (num_rx × num_sub) dimension as spatial width.
        amplitude = amplitude.view(
            batch_size, 1, self.num_tx, self.num_rx * self.num_subcarriers
        )
        phase = phase.view(
            batch_size, 1, self.num_tx, self.num_rx * self.num_subcarriers
        )

        # ── Amplitude branch ─────────────────────────────────────────────
        # No max-pool between layers: num_tx=3 is already too small for two
        # 2×2 pools (would reduce spatial height to <1).  BatchNorm+ReLU
        # after every conv is the standard residual-style normalisation pattern.
        a = F.relu(self.amplitude_norm1(self.amplitude_conv1(amplitude)))
        a = F.relu(self.amplitude_norm2(self.amplitude_conv2(a)))
        a = F.relu(self.amplitude_norm3(self.amplitude_conv3(a)))

        # ── Phase branch ──────────────────────────────────────────────────
        p = F.relu(self.phase_norm1(self.phase_conv1(phase)))
        p = F.relu(self.phase_norm2(self.phase_conv2(p)))
        p = F.relu(self.phase_norm3(self.phase_conv3(p)))

        # ── Fusion ────────────────────────────────────────────────────────
        # Concatenate branch outputs along the channel axis (64+64=128 channels),
        # then compress back to 64 channels with the 1×1 fusion conv.
        combined = torch.cat([a, p], dim=1)
        fused = F.relu(self.fusion_norm(self.fusion_conv(combined)))

        # Adaptive pooling collapses the spatial width to exactly (num_tx, num_rx),
        # making the encoder shape-agnostic at the subcarrier dimension.
        # This is critical for ONNX export with dynamic batch + subcarrier axes.
        pooled = F.adaptive_avg_pool2d(fused, (self.num_tx, self.num_rx))

        # Flatten to (batch, 64 * num_tx * num_rx) and project through two FC layers.
        flattened = pooled.view(batch_size, -1)
        hidden = F.relu(self.fc1(flattened))
        output = self.fc2(hidden)

        return output

    def initialize_weights(self):
        """Apply Kaiming (He) initialization to all Conv2d and Linear layers.

        Kaiming normal initialization is recommended for layers followed by ReLU
        because it sets the variance such that the signal neither vanishes nor
        explodes as it propagates through the network (He et al., 2015).

        Side Effects:
            Modifies parameter tensors of all Conv2d, BatchNorm2d, and Linear
            sub-modules in-place.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # fan_out mode is preferred when the forward pass drives learning
                # (vs. fan_in which is better for linear layers without a following ReLU).
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # Scale=1, bias=0 is the identity at initialisation so BatchNorm
                # has no effect until gradients update its learnable parameters.
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

"""
ID: WR-MODEL-ENC-001
Requirement: Encode a (num_tx x num_rx x num_subcarriers) CSI frame into a
             fixed-length feature vector used by downstream pose estimation.
Purpose: Dual-branch convolutional encoder that processes amplitude and phase
         tensors independently, fuses them, then projects to a compact embedding.
Rationale: Amplitude carries multipath magnitude; phase carries time-of-flight shifts.
           Separate branches until fusion prevent early entanglement of the two signals.
Assumptions: 3x3 MIMO, 64 OFDM subcarriers by default; PyTorch >= 1.10.
Constraints: Adaptive average pooling makes the encoder agnostic to subcarrier count.
References: Widar3.0 (NSDI 2019), Wi-Pose (MobiSys 2022).
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualBranchEncoder(nn.Module):
    """Dual-branch convolutional encoder for amplitude and phase CSI tensors.

    ID: WR-MODEL-ENC-CLASS-001
    Requirement: Accept paired (amplitude, phase) CSI tensors and produce a
                 fixed-size embedding vector for each frame in a batch.
    Purpose: Extract modality-specific spatial features from each CSI branch
             before fusing them into a single compact representation.
    Rationale: Independent branch weights allow each path to specialise on
               its signal statistics before a learned 1x1 fusion layer combines them.
    Inputs:
        amplitude — (batch, num_tx, num_rx, num_subcarriers) float32 tensor.
        phase     — (batch, num_tx, num_rx, num_subcarriers) float32 tensor.
    Outputs:
        (batch, output_dim) float32 feature tensor.
    Preconditions:
        Both tensors on the same device as model parameters.
    Postconditions:
        Output embedding is suitable for direct input to PoseEstimator.
    Assumptions:
        Default config: num_tx=3, num_rx=3, num_subcarriers=64.
    Constraints:
        num_tx must be >= 1; adaptive pooling requires num_tx >= 1.
    References:
        Widar3.0 (NSDI 2019); Wi-Pose (MobiSys 2022).

    Architecture::

        amplitude → Conv2d×3 (with BatchNorm + ReLU) ─┬
                                                        ├→ channel-concat → Conv1×1 → AdaptiveAvgPool → FC×2 → output
        phase     → Conv2d×3 (with BatchNorm + ReLU) ─┘
    """

    def __init__(
        self, num_tx=3, num_rx=3, num_subcarriers=64, hidden_dim=128, output_dim=256
    ):
        """Build all sub-modules and store dimension hyper-parameters.

        ID: WR-MODEL-ENC-INIT-001
        Requirement: Construct all Conv2d, BatchNorm2d, and Linear sub-modules
                     and store hyperparameter attributes for checkpoint provenance.
        Purpose: Fully initialise the encoder so it is ready for a forward pass
                 or weight loading without any additional setup calls.
        Rationale: All layers are pre-allocated at construction time so the graph
                   is static and compatible with torch.onnx.export tracing.
        Inputs:
            num_tx:          int >= 1: number of transmitting antennas (spatial height).
            num_rx:          int >= 1: number of receiving antennas.
            num_subcarriers: int >= 1: OFDM subcarrier count (spatial width factor).
            hidden_dim:      int >= 1: width of the first FC projection layer.
            output_dim:      int >= 1: embedding size returned by forward().
        Outputs:
            None — initialises self; call initialize_weights() for Kaiming init.
        Preconditions:
            PyTorch nn.Module parent __init__() must succeed.
        Postconditions:
            All sub-module parameter tensors allocated on CPU with default init.
            self.num_tx, self.num_rx, self.num_subcarriers, self.hidden_dim,
            self.output_dim are set for checkpoint provenance.
        Assumptions:
            default_rng initialisation (not Kaiming) is applied by PyTorch;
            call initialize_weights() for Kaiming init.
        Side Effects:
            Allocates all Conv2d, BatchNorm2d, and Linear parameter tensors.
        Failure Modes:
            ValueError if any dimension argument <= 0.
        Error Handling:
            PyTorch raises ValueError for invalid layer dimensions.
        Constraints:
            flattened_size = 64 * num_tx * num_rx must fit in a Linear layer.
        Verification:
            Unit test: construct with defaults; assert output_dim == forward output shape.
        References:
            nn.Conv2d, nn.BatchNorm2d, nn.Linear PyTorch documentation.
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

        ID: WR-MODEL-ENC-FWD-001
        Requirement: Process paired amplitude and phase CSI tensors through the
                     dual-branch conv network and return a (batch, output_dim) embedding.
        Purpose: Produce the feature vector consumed by PoseEstimator and
                 MultiPersonPoseEstimator on each inference step.
        Rationale: The reshape-then-conv approach treats the antenna-subcarrier
                   space as a 2-D image so standard Conv2d kernels can be applied.
        Inputs:
            amplitude — float tensor (batch, num_tx, num_rx, num_subcarriers).
                        Normalised CSI amplitude values, typically z-score normalised.
            phase     — float tensor (batch, num_tx, num_rx, num_subcarriers).
                        Unwrapped CSI phase values.
        Outputs:
            float tensor (batch, output_dim): embedded feature vector.
        Preconditions:
            Both tensors must be on the same device as model parameters.
            amplitude.shape == phase.shape.
        Postconditions:
            Output has shape (batch_size, self.output_dim).
            Output is suitable for direct input to PoseEstimator.forward().
        Assumptions:
            Batch size >= 1; single-sample inference with batch_size=1 is valid.
        Side Effects:
            None — pure functional forward pass.
        Failure Modes:
            Shape mismatch between amplitude and phase raises RuntimeError.
            Device mismatch raises RuntimeError from PyTorch.
        Error Handling:
            PyTorch raises descriptive RuntimeError on tensor shape or device issues.
        Constraints:
            num_tx must be >= 1 for AdaptiveAvgPool2d to produce a valid output.
        Verification:
            Unit test: feed random tensors of shape (2, 3, 3, 64); assert output
            shape is (2, 256) and all values are finite.
        References:
            Widar3.0 feature extraction; AdaptiveAvgPool2d PyTorch docs.
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

        ID: WR-MODEL-ENC-INIT-WEIGHTS-001
        Requirement: Set all Conv2d and Linear layer weights using Kaiming normal
                     initialisation and zero-initialise all biases and BatchNorm biases.
        Purpose: Provide stable gradient flow at the start of training by scaling
                 initial weight variance according to the fan-out of each layer.
        Rationale: Kaiming normal init is recommended for ReLU networks because it
                   prevents vanishing/exploding gradients at initialisation depth.
        Inputs:
            None — operates on self.modules().
        Outputs:
            None — modifies parameter tensors in-place.
        Preconditions:
            All sub-modules must already be allocated (called after __init__).
        Postconditions:
            Conv2d weights are Kaiming-normal; biases are zero.
            BatchNorm2d weight=1, bias=0 (identity at init).
            Linear weights are Normal(0, 0.01); biases are zero.
        Assumptions:
            All convolutional layers are followed by ReLU activations.
        Side Effects:
            Modifies all Conv2d, BatchNorm2d, and Linear parameter tensors in-place.
        Failure Modes:
            None expected; all sub-module types are handled.
        Error Handling:
            Unknown module types are silently skipped (isinstance checks only).
        Constraints:
            Must be called before training or ONNX export for reproducible initialisation.
        Verification:
            Unit test: call initialize_weights(); assert conv weight std is close to
            sqrt(2 / fan_out) per Kaiming formula.
        References:
            He et al. (2015) 'Delving Deep into Rectifiers'; nn.init.kaiming_normal_.
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

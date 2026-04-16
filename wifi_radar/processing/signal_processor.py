"""
ID: WR-PROC-SIG-001
Requirement: Convert raw (amplitude, phase) CSI tensors into clean, normalised arrays
             suitable for direct ingestion by DualBranchEncoder.
Purpose: Removes phase discontinuities, normalises amplitude to zero-mean / unit-variance
         per TX-RX link, and suppresses high-frequency noise with a Butterworth
         low-pass filter applied in both the time and subcarrier dimensions.
Algorithm (5 stages):
    1. Phase unwrapping — correct 2π wrap-around discontinuities frame-by-frame.
    2. Amplitude normalisation — per-link z-score normalisation.
    3. History buffering — accumulate frames until the filter can run.
    4. Time-domain filtering — 4th-order Butterworth low-pass via scipy.filtfilt
       (zero phase shift).  Requires ≥ (3 × filter_order + 1) buffered samples.
    5. Subcarrier smoothing — uniform 3-tap moving average across subcarriers to
       suppress inter-carrier leakage.
Constraints:
    - filtfilt requires at least 3× the filter order samples; frames returned
      unfiltered until the buffer fills to that threshold.
    - Not thread-safe: one SignalProcessor instance per processing thread.
"""
import logging

import numpy as np
from scipy import signal


class SignalProcessor:
    """Stateful CSI signal processor (phase unwrap → normalise → filter).

    Each call to ``process()`` updates internal buffers; results depend on
    prior calls.  Instantiate one processor per data stream.
    """

    def __init__(self):
        """Initialise filter coefficients and rolling buffers.

        Side Effects:
            Calls ``scipy.signal.butter`` to pre-compute the Butterworth IIR
            coefficients stored in ``self.b`` / ``self.a``.
        """
        self.logger = logging.getLogger("SignalProcessor")

        # ── Filter design ─────────────────────────────────────────────────
        # 4th-order Butterworth with normalised cutoff at 0.2 (i.e. 0.2 × Nyquist).
        # At 20 Hz sampling the –3 dB corner is at 2 Hz, removing high-frequency
        # noise (respiration ~0.2–0.5 Hz, walking ~1 Hz need to pass through).
        self.butter_order = 4
        self.butterworth_cutoff = 0.2  # Normalised cutoff frequency (fraction of Nyquist)

        # Pre-compute once at construction to avoid repeated coefficient calculation.
        self.b, self.a = signal.butter(
            self.butter_order, self.butterworth_cutoff, "low"
        )

        # ── Phase unwrapping state ─────────────────────────────────────────
        # Stores the previous frame's unwrapped phase to compute inter-frame deltas.
        self.prev_phase = None

        # ── Time-domain filter buffers ─────────────────────────────────────
        # filtfilt needs a run of samples; we accumulate until the buffer is full.
        self.amplitude_buffer = []
        self.phase_buffer = []
        self.buffer_size = 10

    def process(self, amplitude, phase):
        """Run the full 5-stage processing pipeline on one CSI frame.

        Args:
            amplitude: (num_tx, num_rx, num_subcarriers) float array — raw CSI amplitude.
            phase:     (num_tx, num_rx, num_subcarriers) float array — raw CSI phase.

        Returns:
            Tuple (processed_amplitude, processed_phase) with the same shape as input.
            Returns the partially-processed input (normalised but unfiltered) when
            the time-domain buffer has not yet accumulated enough samples.

        Side Effects:
            Updates ``self.prev_phase``, ``self.amplitude_buffer``,
            and ``self.phase_buffer``.

        Failure Modes:
            Any exception during processing is caught, logged, and the raw input
            is returned unchanged to prevent upstream thread crashes.
        """
        try:
            # Stage 1: Unwrap phase to remove 2π jump discontinuities.
            unwrapped_phase = self._unwrap_phase(phase)

            # Stage 2: Normalise amplitude per TX-RX link (zero-mean, unit-variance).
            normalized_amplitude = self._normalize_amplitude(amplitude)

            # Stage 3: Buffer frames for time-domain filtering.
            self.amplitude_buffer.append(normalized_amplitude)
            self.phase_buffer.append(unwrapped_phase)

            # Keep only the most recent buffer_size frames to bound memory usage.
            if len(self.amplitude_buffer) > self.buffer_size:
                self.amplitude_buffer.pop(0)
                self.phase_buffer.pop(0)

            # Stage 4: filtfilt requires at least 3× the filter order samples.
            # Return the unfiltered result until the buffer is sufficiently full.
            min_samples = max(self.buffer_size, 3 * self.butter_order + 1)
            if len(self.amplitude_buffer) < min_samples:
                return normalized_amplitude, unwrapped_phase

            filtered_amplitude = self._apply_time_filter(
                np.array(self.amplitude_buffer)
            )
            filtered_phase = self._apply_time_filter(np.array(self.phase_buffer))

            # Stage 5: Smooth across subcarriers to reduce inter-carrier leakage.
            processed_amplitude = self._apply_frequency_filter(filtered_amplitude[-1])
            processed_phase = self._apply_frequency_filter(filtered_phase[-1])

            return processed_amplitude, processed_phase

        except Exception as e:
            self.logger.error(f"Error in signal processing: {e}")
            # Return input data if processing fails to avoid propagating exceptions.
            return amplitude, phase

    def _unwrap_phase(self, phase):
        """Remove 2π wrap-around discontinuities between consecutive CSI frames.

        Algorithm:
            Compute the inter-frame phase delta and correct jumps whose absolute
            value exceeds π by adding or subtracting 2π.  This is a single-sample
            (frame-by-frame) unwrapping rather than numpy.unwrap (which operates
            along a spatial axis).

        Args:
            phase: (num_tx, num_rx, num_subcarriers) float array — current raw phase.

        Returns:
            Unwrapped phase array with the same shape.

        Side Effects:
            Updates ``self.prev_phase`` to the returned unwrapped array.
        """
        if self.prev_phase is None:
            # First frame: no previous reference, return as-is.
            self.prev_phase = phase
            return phase

        delta_phase = phase - self.prev_phase

        # Correct positive jumps > π (wrap from +π to −π direction).
        corrected_delta = np.where(
            delta_phase > np.pi, delta_phase - 2 * np.pi, delta_phase
        )
        # Correct negative jumps < −π (wrap from −π to +π direction).
        corrected_delta = np.where(
            corrected_delta < -np.pi, corrected_delta + 2 * np.pi, corrected_delta
        )

        unwrapped_phase = self.prev_phase + corrected_delta
        # Keep a copy so the next frame can reference it (prev_phase is mutable).
        self.prev_phase = unwrapped_phase.copy()

        return unwrapped_phase

    def _normalize_amplitude(self, amplitude):
        """Normalise CSI amplitude to zero-mean / unit-variance per TX-RX link.

        Each (TX, RX) pair is normalised independently across its subcarrier axis.
        This removes per-link path-loss differences and keeps model inputs in a
        consistent numeric range regardless of absolute signal power.

        Args:
            amplitude: (num_tx, num_rx, num_subcarriers) float array.

        Returns:
            Normalised array with the same shape.
        """
        # Compute statistics across the subcarrier axis for each (TX, RX) pair.
        mean_amp = np.mean(amplitude, axis=2, keepdims=True)
        std_amp  = np.std(amplitude,  axis=2, keepdims=True)

        # Guard: replace near-zero std with 1.0 to avoid division by zero on
        # constant-amplitude links (can occur with the zero-phase placeholder parser).
        std_amp = np.where(std_amp < 1e-10, 1.0, std_amp)

        return (amplitude - mean_amp) / std_amp

    def _apply_time_filter(self, data_buffer):
        """Apply zero-phase Butterworth low-pass filter along the time axis.

        Algorithm:
            1. Collapse the spatial dimensions (TX, RX, subcarrier) into a single
               feature axis so that ``scipy.signal.filtfilt`` can process all links
               in one vectorised loop without a triple nested loop.
            2. Apply ``filtfilt`` to each feature column.  ``filtfilt`` applies the
               IIR filter forward *and* backward, producing zero phase shift (no
               temporal delay in the output) at the cost of requiring at least
               ``3 × filter_order`` samples in the buffer.
            3. Restore the original (time, TX, RX, subcarrier) shape.

        Args:
            data_buffer: (n_frames, num_tx, num_rx, num_subcarriers) float array
                         containing the most recent buffered frames.

        Returns:
            Filtered array with the same shape as ``data_buffer``.

        Constraints:
            Caller must ensure ``n_frames >= 3 × butter_order + 1``; this is
            enforced by the ``process()`` guard before calling this method.
        """
        # Apply Butterworth filter along time dimension (axis 0)
        shape = data_buffer.shape

        # Reshape (n_frames, TX, RX, SC) → (n_frames, TX*RX*SC) so filtfilt
        # processes all antenna-subcarrier combinations in a single loop.
        reshaped = data_buffer.reshape(shape[0], -1)
        filtered = np.zeros_like(reshaped)

        # filtfilt over the time axis for each (TX, RX, subcarrier) feature.
        for i in range(reshaped.shape[1]):
            filtered[:, i] = signal.filtfilt(self.b, self.a, reshaped[:, i])

        # Restore original spatial shape before returning.
        filtered = filtered.reshape(shape)

        return filtered

    def _apply_frequency_filter(self, data):
        """Apply a uniform 3-tap moving average across the subcarrier axis.

        Algorithm:
            A rectangular (box) kernel of width 3 is convolved with the subcarrier
            amplitude (or phase) vector for each (TX, RX) antenna pair.  The
            ``mode='same'`` argument ensures the output has the same length as the
            input by zero-padding at the edges.

            Rationale: A 3-tap box filter attenuates inter-carrier interference
            without introducing the group-delay distortion of a longer FIR or the
            asymmetric response of an IIR filter applied forward-only.

        Args:
            data: (num_tx, num_rx, num_subcarriers) float array — one processed
                  CSI frame after time-domain filtering.

        Returns:
            Smoothed array with the same shape.

        Side Effects:
            None — pure functional; allocates a new output array.
        """
        # 3-tap uniform (box) kernel: [1/3, 1/3, 1/3]
        kernel_size = 3
        kernel = np.ones(kernel_size) / kernel_size

        # Collapse TX and RX axes so convolution runs over the subcarrier axis
        # for every antenna pair in a single loop.
        shape = data.shape
        reshaped = data.reshape(-1, shape[2])   # (TX*RX, num_subcarriers)
        smoothed = np.zeros_like(reshaped)

        # Convolve each (TX, RX) row with the 3-tap kernel; 'same' keeps length.
        for i in range(reshaped.shape[0]):
            smoothed[i] = np.convolve(reshaped[i], kernel, mode="same")

        # Restore (num_tx, num_rx, num_subcarriers) shape.
        smoothed = smoothed.reshape(shape)

        return smoothed

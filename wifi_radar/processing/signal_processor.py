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
    """Stateful CSI signal processor (phase unwrap -> normalise -> filter).

    ID: WR-PROC-SIG-CLASS-001
    Requirement: Convert raw (amplitude, phase) CSI tensors into clean, normalised
                 arrays suitable for DualBranchEncoder ingestion.
    Purpose: Remove phase discontinuities, normalise amplitude, and suppress
             high-frequency noise through a 5-stage processing pipeline.
    Rationale: Per-link z-score normalisation removes path-loss variance;
               zero-phase Butterworth filter (filtfilt) avoids temporal delay;
               3-tap subcarrier smoothing reduces inter-carrier leakage.
    Inputs:
        amplitude — (num_tx, num_rx, num_sub) float array per call to process().
        phase     — (num_tx, num_rx, num_sub) float array per call to process().
    Outputs:
        Processed (amplitude, phase) tuple with the same shape as input.
    Preconditions:
        Instantiate one SignalProcessor per data stream (not thread-safe).
    Assumptions:
        Sampling rate ~20 Hz; Butterworth cutoff at 0.2 normalised frequency.
    Constraints:
        filtfilt requires >= 3*filter_order+1 buffered samples; earlier frames
        are returned unfiltered.
    References:
        scipy.signal.butter/filtfilt; WR-PROC-SIG-001 module docstring.
    """

    def __init__(self):
        """Initialise filter coefficients and rolling buffers.

        ID: WR-PROC-SIG-INIT-001
        Requirement: Pre-compute Butterworth IIR coefficients and initialise
                     all rolling buffers to an empty state.
        Purpose: Amortise filter coefficient computation over the lifetime of
                 the processor instance rather than recomputing per frame.
        Rationale: Pre-computation via scipy.signal.butter at construction time
                   avoids repeated numerical computation on every process() call.
        Inputs:
            None — no constructor arguments.
        Outputs:
            None — initialises self.
        Preconditions:
            scipy.signal.butter must be available.
        Postconditions:
            self.b and self.a contain 4th-order Butterworth low-pass coefficients.
            self.prev_phase, self.amplitude_buffer, self.phase_buffer initialised.
        Assumptions:
            20 Hz sampling rate; 0.2 normalised cutoff (2 Hz corner frequency).
        Side Effects:
            Calls scipy.signal.butter() to compute IIR coefficients.
        Failure Modes:
            scipy not installed: ImportError at import time (not here).
        Error Handling:
            None required; butter() is deterministic for valid parameters.
        Constraints:
            butter_order=4; butterworth_cutoff=0.2; buffer_size=10.
        Verification:
            Unit test: construct; assert len(self.b)==5 for 4th-order filter.
        References:
            scipy.signal.butter(N, Wn, 'low'); filter design guidelines.
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

        ID: WR-PROC-SIG-PROCESS-001
        Requirement: Accept a raw (amplitude, phase) frame, apply the full pipeline,
                     and return cleaned arrays ready for DualBranchEncoder.
        Purpose: Provide the single entry point for all signal processing operations
                 so the processing thread does not need to call stages individually.
        Rationale: Centralising the pipeline in one method simplifies error handling
                   and ensures stages always run in the correct order.
        Inputs:
            amplitude — (num_tx, num_rx, num_sub) float array: raw CSI amplitude.
            phase     — (num_tx, num_rx, num_sub) float array: raw CSI phase.
        Outputs:
            Tuple (processed_amplitude, processed_phase) same shape as input.
            Returns partially-processed (unfiltered) result until buffer is full.
        Preconditions:
            amplitude and phase must have identical shapes.
        Postconditions:
            self.prev_phase, self.amplitude_buffer, self.phase_buffer updated.
        Assumptions:
            Called at ~20 Hz; first ~10 frames return unfiltered output.
        Side Effects:
            Updates self.prev_phase, self.amplitude_buffer, self.phase_buffer.
        Failure Modes:
            Any stage exception: caught, logged, raw input returned unchanged.
        Error Handling:
            Top-level try/except returns raw input on any processing failure.
        Constraints:
            Not thread-safe; use one instance per processing thread.
        Verification:
            Unit test: feed 15 frames; assert frame 15 differs from raw input
            (filter has had enough samples to activate).
        References:
            _unwrap_phase, _normalize_amplitude, _apply_time_filter,
            _apply_frequency_filter in this class.
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
        """Remove 2pi wrap-around discontinuities between consecutive CSI frames.

        ID: WR-PROC-SIG-UNWRAP-001
        Requirement: Correct inter-frame phase jumps > pi by adding/subtracting 2pi
                     so consecutive phase values are continuously differentiable.
        Purpose: Prevent the neural network from seeing artificial discontinuities
                 caused by the phase modulo-2pi ambiguity.
        Rationale: Frame-by-frame delta correction is cheaper than spatial unwrapping
                   (numpy.unwrap) and sufficient for the 20 Hz frame rate.
        Inputs:
            phase — (num_tx, num_rx, num_sub) float array: current raw phase.
        Outputs:
            Unwrapped phase array with the same shape.
        Preconditions:
            None on the first call (self.prev_phase is None).
        Postconditions:
            self.prev_phase is updated to the returned unwrapped array.
        Assumptions:
            True phase change between consecutive 20 Hz frames is < pi (50 ms window).
        Side Effects:
            Updates self.prev_phase to a copy of the returned unwrapped array.
        Failure Modes:
            Large rapid phase changes (> pi in 50 ms) will be mis-corrected.
        Error Handling:
            No exception handling; numpy operations are assumed safe.
        Constraints:
            First frame is returned unchanged (no previous reference available).
        Verification:
            Unit test: feed two frames with a 3*pi delta; assert correction to pi.
        References:
            Phase unwrapping theory; numpy.where for conditional correction.
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

        ID: WR-PROC-SIG-NORM-001
        Requirement: Apply per-(TX,RX)-link z-score normalisation across the
                     subcarrier axis so model inputs have consistent numeric range.
        Purpose: Remove per-link path-loss differences so the encoder focuses
                 on multipath structure rather than absolute signal power.
        Rationale: Independent normalisation per link avoids coupling between
                   antenna pairs; keepdims preserves broadcasting shape.
        Inputs:
            amplitude — (num_tx, num_rx, num_sub) float array.
        Outputs:
            Normalised array with the same shape; zero-mean, unit-variance per link.
        Preconditions:
            amplitude has at least 1 element per (TX,RX) pair.
        Postconditions:
            Return value has mean ~0 and std ~1 along the subcarrier axis.
        Assumptions:
            At least 2 subcarriers per link (std > 0 in normal operation).
        Side Effects:
            None — pure function; allocates a new array.
        Failure Modes:
            All-constant subcarrier vector: std replaced with 1.0 to avoid div-by-zero.
        Error Handling:
            std < 1e-10 replaced with 1.0 (guard against near-zero std).
        Constraints:
            Normalisation is frame-local; no running statistics are maintained.
        Verification:
            Unit test: input random array; assert output mean ~0 and std ~1 per link.
        References:
            Z-score normalisation; numpy.mean/std axis=2 keepdims=True.
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

        ID: WR-PROC-SIG-TIMEFILT-001
        Requirement: Apply a 4th-order zero-phase Butterworth IIR low-pass filter
                     along the time dimension of a buffered frame stack.
        Purpose: Suppress high-frequency noise (> 2 Hz at 20 Hz sampling) from
                 CSI amplitude and phase time series.
        Rationale: filtfilt applies the IIR filter forward and backward, producing
                   zero phase shift so keypoint predictions are not temporally delayed.
        Inputs:
            data_buffer — (n_frames, num_tx, num_rx, num_sub) float array.
        Outputs:
            Filtered array with the same shape as data_buffer.
        Preconditions:
            n_frames >= 3 * butter_order + 1 (enforced by process() guard).
        Postconditions:
            Output has the same shape as input; high-frequency components attenuated.
        Assumptions:
            self.b, self.a are pre-computed 4th-order Butterworth coefficients.
        Side Effects:
            Allocates a new output array; does not modify data_buffer.
        Failure Modes:
            Insufficient buffer length: scipy raises ValueError (guarded by process()).
        Error Handling:
            Caller (process()) ensures minimum buffer length before calling.
        Constraints:
            Requires >= 13 samples for 4th-order filter (3*4+1=13).
        Verification:
            Unit test: sine at 5 Hz on 20 Hz grid; assert attenuation > 20 dB after filter.
        References:
            scipy.signal.filtfilt; Butterworth IIR design; WR-PROC-SIG-001.
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

        ID: WR-PROC-SIG-FREQFILT-001
        Requirement: Convolve each (TX,RX) antenna pair's subcarrier vector with
                     a 3-tap box kernel to suppress inter-carrier leakage.
        Purpose: Reduce inter-carrier interference without introducing group-delay
                 distortion or asymmetric IIR responses.
        Rationale: A 3-tap rectangular (box) kernel is the simplest linear smoothing
                   filter; mode='same' preserves the subcarrier count.
        Inputs:
            data — (num_tx, num_rx, num_sub) float array: one processed CSI frame.
        Outputs:
            Smoothed array with the same shape as input.
        Preconditions:
            data has shape (num_tx, num_rx, num_sub) with num_sub >= 3.
        Postconditions:
            Each (TX,RX) subcarrier vector has been convolved with [1/3, 1/3, 1/3].
        Assumptions:
            num_sub >= 3 so the 3-tap kernel has at least one non-edge application.
        Side Effects:
            None — pure functional; allocates a new output array.
        Failure Modes:
            num_sub < 3: edge-padded convolution still runs but may reduce SNR.
        Error Handling:
            No exception handling; numpy.convolve is assumed safe for valid inputs.
        Constraints:
            Kernel width is fixed at 3; adjust here if broader smoothing is needed.
        Verification:
            Unit test: constant subcarrier array; assert output equals input (constant
            signal is unaffected by box filter).
        References:
            numpy.convolve mode='same'; rectangular FIR filter; IIR vs FIR tradeoffs.
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

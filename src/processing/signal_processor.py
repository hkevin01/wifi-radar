import numpy as np
from scipy import signal
import logging

class SignalProcessor:
    """Processes raw CSI data for use in the neural network models."""
    
    def __init__(self):
        self.logger = logging.getLogger("SignalProcessor")
        
        # Parameters for filtering
        self.butter_order = 4
        self.butterworth_cutoff = 0.2  # Normalized cutoff frequency
        
        # Design lowpass filter
        self.b, self.a = signal.butter(self.butter_order, self.butterworth_cutoff, 'low')
        
        # Buffer for phase unwrapping
        self.prev_phase = None
        
        # Buffer for time-domain filtering
        self.amplitude_buffer = []
        self.phase_buffer = []
        self.buffer_size = 10
        
    def process(self, amplitude, phase):
        """Process raw CSI data.
        
        Args:
            amplitude: CSI amplitude array of shape (num_tx, num_rx, num_subcarriers)
            phase: CSI phase array of shape (num_tx, num_rx, num_subcarriers)
            
        Returns:
            Tuple of (processed_amplitude, processed_phase) with the same shape as input
        """
        try:
            # 1. Sanitize and unwrap phase
            unwrapped_phase = self._unwrap_phase(phase)
            
            # 2. Normalize amplitude
            normalized_amplitude = self._normalize_amplitude(amplitude)
            
            # 3. Buffer data for time-domain filtering
            self.amplitude_buffer.append(normalized_amplitude)
            self.phase_buffer.append(unwrapped_phase)
            
            if len(self.amplitude_buffer) > self.buffer_size:
                self.amplitude_buffer.pop(0)
                self.phase_buffer.pop(0)
                
            # Skip time-domain filtering if buffer is not full
            if len(self.amplitude_buffer) < self.buffer_size:
                return normalized_amplitude, unwrapped_phase
                
            # 4. Apply time-domain filtering
            filtered_amplitude = self._apply_time_filter(np.array(self.amplitude_buffer))
            filtered_phase = self._apply_time_filter(np.array(self.phase_buffer))
            
            # 5. Apply frequency-domain filtering (across subcarriers)
            processed_amplitude = self._apply_frequency_filter(filtered_amplitude[-1])
            processed_phase = self._apply_frequency_filter(filtered_phase[-1])
            
            return processed_amplitude, processed_phase
            
        except Exception as e:
            self.logger.error(f"Error in signal processing: {e}")
            # Return input data if processing fails
            return amplitude, phase
    
    def _unwrap_phase(self, phase):
        """Unwrap phase to handle 2π discontinuities."""
        if self.prev_phase is None:
            self.prev_phase = phase
            return phase
            
        # Unwrap phase by comparing with previous phase
        delta_phase = phase - self.prev_phase
        
        # Correct jumps larger than π
        corrected_delta = np.where(delta_phase > np.pi, delta_phase - 2*np.pi, delta_phase)
        corrected_delta = np.where(corrected_delta < -np.pi, corrected_delta + 2*np.pi, corrected_delta)
        
        unwrapped_phase = self.prev_phase + corrected_delta
        self.prev_phase = unwrapped_phase.copy()
        
        return unwrapped_phase
    
    def _normalize_amplitude(self, amplitude):
        """Normalize CSI amplitude."""
        # Calculate mean and standard deviation across subcarriers for each TX-RX pair
        mean_amp = np.mean(amplitude, axis=2, keepdims=True)
        std_amp = np.std(amplitude, axis=2, keepdims=True)
        
        # Avoid division by zero
        std_amp = np.where(std_amp < 1e-10, 1.0, std_amp)
        
        # Normalize to zero mean and unit variance
        normalized = (amplitude - mean_amp) / std_amp
        
        return normalized
    
    def _apply_time_filter(self, data_buffer):
        """Apply time-domain filtering to buffered data."""
        # Apply Butterworth filter along time dimension (axis 0)
        shape = data_buffer.shape
        
        # Reshape for filtering
        reshaped = data_buffer.reshape(shape[0], -1)
        filtered = np.zeros_like(reshaped)
        
        # Apply filter to each feature
        for i in range(reshaped.shape[1]):
            filtered[:, i] = signal.filtfilt(self.b, self.a, reshaped[:, i])
            
        # Reshape back
        filtered = filtered.reshape(shape)
        
        return filtered
    
    def _apply_frequency_filter(self, data):
        """Apply frequency-domain filtering across subcarriers."""
        # Apply smoothing across subcarriers (axis 2)
        kernel_size = 3
        kernel = np.ones(kernel_size) / kernel_size
        
        # Reshape for convolution
        shape = data.shape
        reshaped = data.reshape(-1, shape[2])
        smoothed = np.zeros_like(reshaped)
        
        # Apply convolution to each TX-RX pair
        for i in range(reshaped.shape[0]):
            smoothed[i] = np.convolve(reshaped[i], kernel, mode='same')
            
        # Reshape back
        smoothed = smoothed.reshape(shape)
        
        return smoothed

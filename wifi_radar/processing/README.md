# Signal Processing

This directory contains signal processing modules for WiFi CSI data:

- `signal_processor.py`: Processes raw CSI data for neural network input

## Signal Processor

The `SignalProcessor` class applies various signal processing techniques to raw CSI data:
- Phase unwrapping
- Amplitude normalization
- Time-domain filtering
- Frequency-domain filtering

## Usage

```python
from wifi_radar.processing.signal_processor import SignalProcessor

# Initialize processor
processor = SignalProcessor()

# Process raw CSI data
processed_amplitude, processed_phase = processor.process(raw_amplitude, raw_phase)
```

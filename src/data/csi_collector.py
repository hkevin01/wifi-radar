import numpy as np
import time
import threading
import logging
from queue import Queue
import socket
import struct

class CSICollector:
    """Collects Channel State Information (CSI) from WiFi router."""
    
    def __init__(self, router_ip='192.168.1.1', port=5500, buffer_size=100):
        self.router_ip = router_ip
        self.port = port
        self.buffer_size = buffer_size
        self.csi_data_queue = Queue(maxsize=buffer_size)
        self.running = False
        self.logger = logging.getLogger("CSICollector")
        
        # 3x3 MIMO configuration
        self.num_tx = 3  # Transmitting antennas
        self.num_rx = 3  # Receiving antennas
        self.num_subcarriers = 64  # OFDM subcarriers
        
        # For simulation mode
        self.simulation_mode = True
        
    def start(self, simulation_mode=True):
        """Start CSI data collection.
        
        Args:
            simulation_mode: If True, generate simulated CSI data instead of connecting to router.
        """
        self.simulation_mode = simulation_mode
        self.running = True
        
        if simulation_mode:
            self.logger.info("Starting CSI collector in simulation mode")
            self.collection_thread = threading.Thread(target=self._simulate_csi_data)
        else:
            self.logger.info(f"Starting CSI collector connecting to {self.router_ip}:{self.port}")
            self.collection_thread = threading.Thread(target=self._collect_csi_data)
            
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
    def stop(self):
        """Stop CSI data collection."""
        self.running = False
        if hasattr(self, 'collection_thread'):
            self.collection_thread.join(timeout=1.0)
        self.logger.info("CSI collector stopped")
        
    def get_csi_data(self, block=True, timeout=None):
        """Get CSI data from the queue.
        
        Returns:
            Tuple of (amplitude, phase) arrays with shape (num_tx, num_rx, num_subcarriers)
        """
        try:
            return self.csi_data_queue.get(block=block, timeout=timeout)
        except Exception as e:
            self.logger.error(f"Error getting CSI data: {e}")
            return None
        
    def _collect_csi_data(self):
        """Collect real CSI data from WiFi router."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.router_ip, self.port))
            
            while self.running:
                # This is a simplified example - actual CSI data format depends on router firmware
                raw_data = sock.recv(8192)
                if not raw_data:
                    continue
                    
                # Parse the raw CSI data (implementation depends on router and protocol)
                amplitude, phase = self._parse_csi_data(raw_data)
                
                if not self.csi_data_queue.full():
                    self.csi_data_queue.put((amplitude, phase))
                    
        except Exception as e:
            self.logger.error(f"Error in CSI collection: {e}")
        finally:
            sock.close()
            
    def _simulate_csi_data(self):
        """Generate simulated CSI data for testing."""
        while self.running:
            # Generate random CSI data with realistic properties
            amplitude = np.random.rayleigh(scale=1.0, 
                                          size=(self.num_tx, self.num_rx, self.num_subcarriers))
            
            # Phase is uniformly distributed between -π and π
            phase = np.random.uniform(-np.pi, np.pi, 
                                     size=(self.num_tx, self.num_rx, self.num_subcarriers))
            
            # Add simulated human presence in the environment
            self._add_simulated_human_presence(amplitude, phase)
            
            if not self.csi_data_queue.full():
                self.csi_data_queue.put((amplitude, phase))
            
            time.sleep(0.05)  # 20 Hz sampling rate
            
    def _parse_csi_data(self, raw_data):
        """Parse raw CSI data from router.
        
        This is a placeholder implementation. The actual parsing depends on the
        specific format used by the router firmware.
        """
        # Placeholder implementation
        amplitude = np.zeros((self.num_tx, self.num_rx, self.num_subcarriers))
        phase = np.zeros((self.num_tx, self.num_rx, self.num_subcarriers))
        
        # Parse the raw data into amplitude and phase
        # This would be specific to the router and protocol
        
        return amplitude, phase
    
    def _add_simulated_human_presence(self, amplitude, phase):
        """Add simulated human presence effects to CSI data."""
        # Simplified human presence simulation
        # In reality, human presence creates complex patterns of reflection and absorption
        
        # Simulate a person moving
        t = time.time()
        x_pos = 0.5 + 0.3 * np.sin(t * 0.5)  # Person moving in x direction
        y_pos = 0.5 + 0.2 * np.cos(t * 0.3)  # Person moving in y direction
        
        # Apply effects to specific subcarriers based on position
        for tx in range(self.num_tx):
            for rx in range(self.num_rx):
                # Calculate distance-based effect
                effect_magnitude = 0.2 * np.exp(-((tx/self.num_tx - x_pos)**2 + 
                                                 (rx/self.num_rx - y_pos)**2) * 10)
                
                # Apply to a range of subcarriers with frequency-dependent effect
                for sc in range(self.num_subcarriers):
                    sc_factor = np.sin(sc / self.num_subcarriers * np.pi * 4)
                    
                    # Modify amplitude
                    amplitude[tx, rx, sc] *= (1 + effect_magnitude * sc_factor * 0.5)
                    
                    # Add phase shift
                    phase[tx, rx, sc] += effect_magnitude * sc_factor * 0.8

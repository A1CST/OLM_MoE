import threading
import time
import numpy as np
import sounddevice as sd
from PyQt5.QtCore import QObject, pyqtSignal
from collections import deque
from scipy.fft import fft


class AudioEngine(QObject):
    """Audio capture engine that provides chunked audio data synchronized with vision ticks."""
    
    audio_chunk_ready = pyqtSignal(np.ndarray)  # Raw audio chunk (1024 samples)
    audio_fft_ready = pyqtSignal(np.ndarray)    # FFT spectrum for visualization (64 bands)
    audio_level = pyqtSignal(float)             # RMS level for basic visualization
    
    def __init__(self, 
                 device_id: int = 12,
                 sample_rate: int = 48000,
                 chunk_size: int = 1024,
                 buffer_chunks: int = 10):
        super().__init__()
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # Audio buffer to store recent chunks
        self._audio_buffer = deque(maxlen=buffer_chunks)
        self._buffer_lock = threading.Lock()
        
        # FFT processing
        self._fft_bands = 64
        self._smoothed_fft = np.zeros(self._fft_bands)
        self._smoothing_factor = 0.7
        
        # Visual smoothing (separate from processing)
        self._visual_smoothing_factor = 0.8  # More aggressive smoothing for visuals
        self._smoothed_visual_fft = np.zeros(self._fft_bands)
        
        # Stream control
        self._stream = None
        self._running = False
        
    def _audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice stream - processes each audio chunk."""
        if status:
            print(f"Audio callback status: {status}")
            
        # Convert to mono if stereo (handle both 1D and 2D arrays)
        if indata.ndim > 1 and indata.shape[1] > 1:
            # Stereo input - average channels to mono
            audio_data = np.mean(indata, axis=1)
        else:
            # Mono input or single channel
            audio_data = indata.flatten()
            
        # Calculate RMS level
        rms_level = np.sqrt(np.mean(audio_data ** 2))
        self.audio_level.emit(float(rms_level * 20))  # Scale for visualization
        
        # Store in buffer
        with self._buffer_lock:
            self._audio_buffer.append(audio_data.copy())
            
        # Emit raw chunk for processing
        self.audio_chunk_ready.emit(audio_data.copy())
        
        # Process FFT for visualization
        self._process_fft(audio_data)
        
    def _process_fft(self, audio_data: np.ndarray):
        """Process FFT spectrum for visualization."""
        if len(audio_data) == 0:
            return
            
        # Apply FFT
        fft_data = np.abs(fft(audio_data))
        fft_data = fft_data[:len(fft_data)//2]  # Take positive frequencies only
        
        # Normalize
        if np.max(fft_data) > 0:
            fft_data = fft_data / np.max(fft_data)
            
        # Downsample to desired number of bands
        band_width = len(fft_data) // self._fft_bands
        if band_width > 0:
            bands = np.array([
                np.mean(fft_data[i*band_width:(i+1)*band_width]) 
                for i in range(self._fft_bands)
            ])
        else:
            bands = np.zeros(self._fft_bands)
            
        # Apply smoothing for processing (less aggressive)
        self._smoothed_fft = (self._smoothing_factor * self._smoothed_fft + 
                             (1 - self._smoothing_factor) * bands)
        
        # Apply additional smoothing for visuals only (more aggressive)
        self._smoothed_visual_fft = (self._visual_smoothing_factor * self._smoothed_visual_fft + 
                                    (1 - self._visual_smoothing_factor) * bands)
        
        # Emit smoothed visual data for GUI visualization
        self.audio_fft_ready.emit(self._smoothed_visual_fft.copy())
        
    def get_latest_chunk(self) -> np.ndarray | None:
        """Get the most recent audio chunk for processing."""
        with self._buffer_lock:
            if len(self._audio_buffer) > 0:
                return self._audio_buffer[-1].copy()
            return None
            
    def get_audio_history(self, num_chunks: int = 5) -> list[np.ndarray]:
        """Get recent audio chunks for sequence processing."""
        with self._buffer_lock:
            if len(self._audio_buffer) == 0:
                return []
            return list(self._audio_buffer)[-num_chunks:]
            
    def get_raw_fft(self) -> np.ndarray:
        """Get raw FFT data for processing (less smoothed)."""
        return self._smoothed_fft.copy()
        
    def get_visual_fft(self) -> np.ndarray:
        """Get smoothed FFT data for visualization only."""
        return self._smoothed_visual_fft.copy()
        
    def set_visual_smoothing(self, factor: float):
        """Set visual smoothing factor (0.0 = no smoothing, 1.0 = max smoothing)."""
        self._visual_smoothing_factor = max(0.0, min(1.0, factor))
        print(f"Visual smoothing factor set to: {self._visual_smoothing_factor}")
            
    def start(self):
        """Start audio capture."""
        if self._running:
            return
            
        # Try different channel configurations
        channel_configs = [1, 2]  # Try mono first, then stereo
        device_configs = [self.device_id, None]  # Try specified device, then default
            
        for device in device_configs:
            for channels in channel_configs:
                try:
                    print(f"Trying device {device} with {channels} channel(s)...")
                    self._stream = sd.InputStream(
                        device=device,
                        channels=channels,
                        samplerate=self.sample_rate,
                        blocksize=self.chunk_size,
                        callback=self._audio_callback
                    )
                    self._stream.start()
                    self._running = True
                    self.device_id = device
                    self.channels = channels
                    print(f"Audio engine started on device {device} with {channels} channel(s)")
                    return
                    
                except Exception as e:
                    print(f"Failed with device {device}, {channels} channels: {e}")
                    if self._stream:
                        try:
                            self._stream.close()
                        except:
                            pass
                        self._stream = None
                    continue
        
        # If all configurations fail, print available devices
        print("Failed to start audio engine with any configuration")
        print("Available audio devices:")
        print(sd.query_devices())
            
    def stop(self):
        """Stop audio capture."""
        if not self._running:
            return
            
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            
        with self._buffer_lock:
            self._audio_buffer.clear()
            
        print("Audio engine stopped")
        
    def is_running(self) -> bool:
        """Check if audio capture is active."""
        return self._running
        
    @staticmethod
    def list_audio_devices():
        """List all available audio devices with their capabilities."""
        print("Available audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(f"Device {i}: {device['name']}")
            print(f"  - Max input channels: {device['max_input_channels']}")
            print(f"  - Max output channels: {device['max_output_channels']}")
            print(f"  - Default sample rate: {device['default_samplerate']}")
            print(f"  - Host API: {device['hostapi']}")
            print()
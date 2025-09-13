from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import Qt, pyqtSlot
import numpy as np


class AudioVisualizerWidget(QWidget):
    """Audio spectrum visualizer widget showing FFT bars."""
    
    def __init__(self, num_bars=64, parent=None):
        super().__init__(parent)
        self.num_bars = num_bars
        self.spectrum_data = np.zeros(num_bars)
        self.audio_level = 0.0
        
        # Colors
        self.bg_color = QColor(255, 255, 255)  # White background
        self.bar_color_low = QColor(0, 0, 0)    # Black for low levels
        self.bar_color_high = QColor(255, 0, 0) # Red for high levels
        self.level_color = QColor(0, 200, 0)    # Green for level indicator
        
        # Thresholds
        self.red_threshold = 0.3  # Magnitude threshold for red bars
        
        self.setMinimumSize(400, 150)
        
    @pyqtSlot(np.ndarray)
    def update_spectrum(self, spectrum_data):
        """Update the spectrum data and repaint."""
        if len(spectrum_data) == self.num_bars:
            self.spectrum_data = spectrum_data.copy()
            self.update()
            
    @pyqtSlot(float)
    def update_level(self, level):
        """Update the audio level and repaint."""
        self.audio_level = level
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), self.bg_color)
        
        if len(self.spectrum_data) == 0:
            return
            
        # Draw spectrum bars
        width = self.width()
        height = self.height()
        
        # Reserve space for level indicator at bottom
        spectrum_height = height - 30
        spectrum_y_center = spectrum_height // 2
        
        bar_width = max(1, width // self.num_bars)
        
        # Center the spectrogram
        total_spectrum_width = self.num_bars * bar_width
        spectrum_x_start = (width - total_spectrum_width) // 2
        
        for i in range(self.num_bars):
            magnitude = self.spectrum_data[i]
            
            # Calculate bar height - increased scale for taller bars
            bar_height = int(magnitude * spectrum_height * 1.4)
            
            # Position - centered
            x = spectrum_x_start + i * bar_width
            y_top = spectrum_y_center - bar_height
            
            # Choose color based on magnitude
            if magnitude > self.red_threshold:
                color = self.bar_color_high
            else:
                color = self.bar_color_low
                
            painter.setBrush(color)
            painter.setPen(QPen(color, 1))
            
            # Draw symmetric bar (up and down from center) - thinner bars
            actual_bar_width = max(1, bar_width - 2)  # Leave more space between bars
            painter.drawRect(x, y_top, actual_bar_width, bar_height * 2)
            
        # Draw audio level indicator at bottom - much thinner
        level_y = spectrum_height + 10
        level_height = 6  # Much thinner level indicator
        level_width = int((self.audio_level / 20.0) * width)  # Scale by 20 to match AudioEngine
        level_width = max(0, min(level_width, width))
        
        painter.setBrush(self.level_color)
        painter.setPen(QPen(self.level_color, 1))
        painter.drawRect(0, level_y, level_width, level_height)
        
        # Draw level indicator border
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.drawRect(0, level_y, width, level_height)
        
        painter.end()
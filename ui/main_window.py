
import sys
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QPlainTextEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, pyqtSlot
from engine import Engine
from dual_vae import FeatureEngine
from utils import convert_cv_qt, convert_rgb_qt, wipe_state
import numpy as np
import cv2
from .heatmap_window import HeatmapWindow
from .audio_widget import AudioVisualizerWidget
from .display_windows import ScreenCaptureWindow, HashMapWindow, LiveVAEWindow, DPipeWindow
from .worker_windows import WorkerRegistryWindow, IILSTMRoutingWindow

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Vision'
        self.left = 100
        self.top = 100
        self.width = 1280
        self.height = 720
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        # Set window transparency and styling
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowOpacity(0.95)
        self.setStyleSheet("""
            QMainWindow {
                background-color: rgba(100, 149, 237, 150);
                border-radius: 10px;
            }
            QWidget {
                background-color: rgba(135, 206, 250, 80);
                border-radius: 5px;
            }
            QPushButton {
                background-color: rgba(70, 130, 180, 180);
                border: 1px solid rgba(0, 0, 100, 100);
                border-radius: 5px;
                color: white;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(100, 149, 237, 200);
            }
            QPlainTextEdit {
                background-color: rgba(255, 255, 255, 200);
                border: 1px solid rgba(70, 130, 180, 150);
                border-radius: 3px;
            }
            QLabel {
                color: white;
                font-weight: bold;
            }
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self._create_display_layout()
        self._create_feature_layout()
        self._create_audio_layout()
        self._create_stats_layout()
        self._create_button_layout()

        self.layout.addLayout(self.display_layout)
        self.layout.addLayout(self.feature_layout)
        self.layout.addLayout(self.audio_layout)
        self.layout.addLayout(self.stats_layout)
        self.layout.addLayout(self.button_layout)

        self.engine = Engine()
        self.feature_engine = FeatureEngine()
        
        self._connect_signals()

    def _create_display_layout(self):
        self.display_layout = QHBoxLayout()
        
        # Create buttons for display windows
        self.screen_capture_button = QPushButton('Screen Capture', self)
        self.hash_map_button = QPushButton('Hash Map (3D)', self)
        self.live_vae_button = QPushButton('Live VAE Prediction', self)
        self.dpipe_button = QPushButton('D-pipe Prediction', self)
        
        # Make buttons larger and more prominent
        button_style = "QPushButton { padding: 10px; font-size: 12px; }"
        self.screen_capture_button.setStyleSheet(button_style)
        self.hash_map_button.setStyleSheet(button_style)
        self.live_vae_button.setStyleSheet(button_style)
        self.dpipe_button.setStyleSheet(button_style)
        
        self.display_layout.addWidget(self.screen_capture_button)
        self.display_layout.addWidget(self.hash_map_button)
        self.display_layout.addWidget(self.live_vae_button)
        self.display_layout.addWidget(self.dpipe_button)
        
        # Add worker monitoring buttons
        self.worker_registry_button = QPushButton('Worker Registry', self)
        self.iilstm_routing_button = QPushButton('IILSTM Routing', self)
        
        self.worker_registry_button.setStyleSheet(button_style + "background-color: rgba(255, 140, 0, 180);")  # Orange
        self.iilstm_routing_button.setStyleSheet(button_style + "background-color: rgba(220, 20, 60, 180);")   # Crimson
        
        self.display_layout.addWidget(self.worker_registry_button)
        self.display_layout.addWidget(self.iilstm_routing_button)
        
        # Initialize display windows (but don't show them)
        self.screen_capture_window = None
        self.hash_map_window = None
        self.live_vae_window = None
        self.dpipe_window = None
        self.worker_registry_window = None
        self.iilstm_routing_window = None

    def _create_feature_layout(self):
        self.feature_layout = QHBoxLayout()
        # Frozen
        self.frozen_text = QPlainTextEdit(self)
        self.frozen_text.setReadOnly(True)
        self.frozen_text.setFixedSize(640, 120)
        self.frozen_text.setStyleSheet("border: 1px solid black;")
        self.frozen_label = QLabel('Frozen VAE features', self)
        self.frozen_label.setAlignment(Qt.AlignCenter)
        self.frozen_hash_label = QLabel('Hash: —', self)
        self.frozen_hash_label.setAlignment(Qt.AlignLeft)
        self.frozen_col = QVBoxLayout()
        self.frozen_col.addWidget(self.frozen_label)
        self.frozen_col.addWidget(self.frozen_text)
        self.frozen_col.addWidget(self.frozen_hash_label)
        # Live
        self.live_text = QPlainTextEdit(self)
        self.live_text.setReadOnly(True)
        self.live_text.setFixedSize(640, 120)
        self.live_text.setStyleSheet("border: 1px solid black;")
        self.live_label = QLabel('Live VAE features (training)', self)
        self.live_label.setAlignment(Qt.AlignCenter)
        self.live_hash_label = QLabel('Hash: —', self)
        self.live_hash_label.setAlignment(Qt.AlignLeft)
        self.live_col = QVBoxLayout()
        self.live_col.addWidget(self.live_label)
        self.live_col.addWidget(self.live_text)
        self.live_col.addWidget(self.live_hash_label)
        # Add to row
        self.feature_layout.addLayout(self.frozen_col)
        self.feature_layout.addLayout(self.live_col)

    def _create_audio_layout(self):
        self.audio_layout = QHBoxLayout()
        
        # Audio visualizer
        self.audio_label = QLabel('Audio Spectrum (48kHz)', self)
        self.audio_label.setAlignment(Qt.AlignCenter)
        self.audio_visualizer = AudioVisualizerWidget(num_bars=64, parent=self)
        self.audio_visualizer.setFixedHeight(150)
        self.audio_visualizer.setStyleSheet("border: 1px solid black;")
        
        self.audio_col = QVBoxLayout()
        self.audio_col.addWidget(self.audio_label)
        self.audio_col.addWidget(self.audio_visualizer)
        
        # Audio info panel
        self.audio_info_text = QPlainTextEdit(self)
        self.audio_info_text.setReadOnly(True)
        self.audio_info_text.setFixedSize(400, 150)
        self.audio_info_text.setStyleSheet("border: 1px solid black;")
        self.audio_info_text.setPlainText("Audio Engine Status:\n- Device: 105 (48kHz mono)\n- Chunk size: 1024 samples\n- Processing: Frozen VAE (16D latents)\n- Combined with vision: 48D total")
        self.audio_info_label = QLabel('Audio Processing Info', self)
        self.audio_info_label.setAlignment(Qt.AlignCenter)
        
        self.audio_info_col = QVBoxLayout()
        self.audio_info_col.addWidget(self.audio_info_label)
        self.audio_info_col.addWidget(self.audio_info_text)
        
        self.audio_layout.addLayout(self.audio_col)
        self.audio_layout.addLayout(self.audio_info_col)

    def _create_stats_layout(self):
        self.stats_layout = QVBoxLayout()
        self.stats_label = QLabel('Hash statistics', self)
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_text = QPlainTextEdit(self)
        self.stats_text.setReadOnly(True)
        self.stats_text.setFixedHeight(140)
        self.stats_text.setStyleSheet("border: 1px solid black;")
        self.stats_layout.addWidget(self.stats_label)
        self.stats_layout.addWidget(self.stats_text)
        # Tick/TPS row
        self.metrics_row = QHBoxLayout()
        self.tick_label = QLabel('Tick: 0', self)
        self.tps_label = QLabel('TPS: 0.0', self)
        self.pred_acc_label = QLabel('Pred acc (live): —', self)
        self.metrics_row.addWidget(self.tick_label)
        self.metrics_row.addWidget(self.tps_label)
        self.metrics_row.addWidget(self.pred_acc_label)
        self.dpipe_acc_label = QLabel('D acc: —', self)
        self.metrics_row.addWidget(self.dpipe_acc_label)
        self.stats_layout.addLayout(self.metrics_row)
        
        # Novelty and drivers row
        self.novelty_row = QHBoxLayout()
        self.novelty_label = QLabel('Novelty: 0.000', self)
        self.novelty_label.setStyleSheet("color: orange; font-weight: bold;")
        self.energy_label = QLabel('Energy: 1.00', self)
        self.energy_label.setStyleSheet("color: green; font-weight: bold;")
        self.sleep_label = QLabel('Sleep: 0.00', self)
        self.sleep_label.setStyleSheet("color: purple; font-weight: bold;")
        self.workers_label = QLabel('Workers: 0', self)
        self.workers_label.setStyleSheet("color: cyan; font-weight: bold;")
        self.novelty_row.addWidget(self.novelty_label)
        self.novelty_row.addWidget(self.energy_label)
        self.novelty_row.addWidget(self.sleep_label)
        self.novelty_row.addWidget(self.workers_label)
        self.stats_layout.addLayout(self.novelty_row)

        # Warnings panel
        self.warn_label = QLabel('Diagnostics warnings', self)
        self.warn_label.setAlignment(Qt.AlignCenter)
        self.warn_text = QPlainTextEdit(self)
        self.warn_text.setReadOnly(True)
        self.warn_text.setFixedHeight(120)
        self.warn_text.setStyleSheet("border: 1px solid black;")
        self.stats_layout.addWidget(self.warn_label)
        self.stats_layout.addWidget(self.warn_text)

    def _create_button_layout(self):
        self.button_layout = QHBoxLayout()
        self.start_button = QPushButton('Start', self)
        self.stop_button = QPushButton('Stop', self)
        self.button_layout.addWidget(self.start_button)
        self.button_layout.addWidget(self.stop_button)
        self.wipe_button = QPushButton('Wipe State', self)
        self.button_layout.addWidget(self.wipe_button)
        self.open_heatmap_button = QPushButton('Open Heatmap', self)
        self.button_layout.addWidget(self.open_heatmap_button)

    def _connect_signals(self):
        self.start_button.clicked.connect(self.start_all)
        self.stop_button.clicked.connect(self.stop_all)
        self.wipe_button.clicked.connect(self.on_wipe)
        self.open_heatmap_button.clicked.connect(self.on_open_heatmap)
        
        # Connect display window buttons
        self.screen_capture_button.clicked.connect(self.on_open_screen_capture)
        self.hash_map_button.clicked.connect(self.on_open_hash_map)
        self.live_vae_button.clicked.connect(self.on_open_live_vae)
        self.dpipe_button.clicked.connect(self.on_open_dpipe)
        
        # Connect worker monitoring buttons
        self.worker_registry_button.clicked.connect(self.on_open_worker_registry)
        self.iilstm_routing_button.clicked.connect(self.on_open_iilstm_routing)
        # Connect signals to main GUI components (not display windows)
        self.feature_engine.frozen_features.connect(self.update_frozen_text)
        self.feature_engine.live_features.connect(self.update_live_text)
        self.feature_engine.lsh_chained_hash.connect(self.update_hash_labels)
        self.feature_engine.hashmap_stats.connect(self.update_stats)
        self.feature_engine.tick_update.connect(self.update_tick)
        self.feature_engine.tps_update.connect(self.update_tps)
        self.feature_engine.prediction_accuracy_text.connect(self.update_pred_acc)
        self.feature_engine.diag_warning.connect(self.append_warning)
        self.feature_engine.diag_info.connect(self.append_info)
        self.feature_engine.dpipe_accuracy_text.connect(self.update_dpipe_acc)
        # Audio engine signals
        self.feature_engine.audio_engine.audio_fft_ready.connect(self.audio_visualizer.update_spectrum)
        self.feature_engine.audio_engine.audio_level.connect(self.audio_visualizer.update_level)
        
        # Novelty and worker monitoring signals
        self.feature_engine.novelty_value.connect(self.update_novelty)
        self.feature_engine.novelty_components_text.connect(self.update_novelty_components)
        self.feature_engine.worker_registry_data.connect(self.update_worker_count)
        self.feature_engine.energy_value.connect(self.update_energy)
        # Heatmap hit coordinates
        try:
            self.feature_engine.heatmap_hit.connect(self.on_heatmap_hit)
        except Exception:
            pass

    # Display window opening methods
    def on_open_screen_capture(self):
        if self.screen_capture_window is None:
            self.screen_capture_window = ScreenCaptureWindow()
            self.engine.new_frame.connect(self.screen_capture_window.update_frame)
        self.screen_capture_window.show()
        self.screen_capture_window.raise_()
        
    def on_open_hash_map(self):
        if self.hash_map_window is None:
            self.hash_map_window = HashMapWindow()
            self.feature_engine.hashmap_frame.connect(self.hash_map_window.update_frame)
        self.hash_map_window.show()
        self.hash_map_window.raise_()
        
    def on_open_live_vae(self):
        if self.live_vae_window is None:
            self.live_vae_window = LiveVAEWindow()
            self.feature_engine.live_prediction_frame.connect(self.live_vae_window.update_frame)
        self.live_vae_window.show()
        self.live_vae_window.raise_()
        
    def on_open_dpipe(self):
        if self.dpipe_window is None:
            self.dpipe_window = DPipeWindow()
            self.feature_engine.dpipe_prediction_frame.connect(self.dpipe_window.update_frame)
        self.dpipe_window.show()
        self.dpipe_window.raise_()
        
    def on_open_worker_registry(self):
        if self.worker_registry_window is None:
            self.worker_registry_window = WorkerRegistryWindow()
            # Connect worker registry data signal
            self.feature_engine.worker_registry_data.connect(self.worker_registry_window.update_workers_json)
        self.worker_registry_window.show()
        self.worker_registry_window.raise_()
        
    def on_open_iilstm_routing(self):
        if self.iilstm_routing_window is None:
            self.iilstm_routing_window = IILSTMRoutingWindow()
            # Connect novelty components signal for routing info
            self.feature_engine.novelty_components_text.connect(self.iilstm_routing_window.update_routing_data)
        self.iilstm_routing_window.show()
        self.iilstm_routing_window.raise_()

    @pyqtSlot(str)
    def update_frozen_text(self, text):
        self.frozen_text.setPlainText(text)

    @pyqtSlot(str)
    def update_live_text(self, text):
        self.live_text.setPlainText(text)

    @pyqtSlot(str)
    def update_hash_labels(self, hash_hex):
        self.frozen_hash_label.setText(f"Hash: {hash_hex}")
        self.live_hash_label.setText(f"Hash: {hash_hex}")

    @pyqtSlot(str)
    def update_stats(self, text):
        self.stats_text.setPlainText(text)

    @pyqtSlot(int)
    def update_tick(self, tick):
        self.tick_label.setText(f"Tick: {tick}")

    @pyqtSlot(float)
    def update_tps(self, tps):
        self.tps_label.setText(f"TPS: {tps:.1f}")

    @pyqtSlot(str)
    def update_pred_acc(self, text):
        self.pred_acc_label.setText(text)

    @pyqtSlot(str)
    def update_dpipe_acc(self, text):
        self.dpipe_acc_label.setText(text)
        
    @pyqtSlot(float)
    def update_novelty(self, novelty: float):
        """Update novelty counter on main GUI."""
        self.novelty_label.setText(f"Novelty: {novelty:.3f}")
        
        # Update color based on novelty level
        if novelty > 0.7:
            self.novelty_label.setStyleSheet("color: red; font-weight: bold;")
        elif novelty > 0.4:
            self.novelty_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.novelty_label.setStyleSheet("color: lightgreen; font-weight: bold;")
            
        # If IILSTM window is open, update its drivers
        if hasattr(self, 'iilstm_routing_window') and self.iilstm_routing_window:
            # We'll extract energy/sleep from the log data or estimate
            # For now, just update novelty
            pass
    
    @pyqtSlot(str)
    def update_novelty_components(self, components_text: str):
        """Parse novelty components and update driver displays."""
        try:
            # Parse the components text to extract energy and sleep pressure
            # Format: "novelty= 0.123 (latent=0.456, hashΔ=0.789, predHashWrong=0.012; mse=0.000123)"
            if "novelty=" in components_text:
                # Extract energy and sleep from feature engine if available
                # For now, we'll simulate or extract from available data
                
                # Extract worker count from the components text - for now, update when registry signal comes
                # This will be updated by the worker registry signal
                
                # If worker windows are open, send updates
                if hasattr(self, 'iilstm_routing_window') and self.iilstm_routing_window:
                    self.iilstm_routing_window.update_routing_data(components_text)
                    
        except Exception as e:
            print(f"Error parsing novelty components: {e}")

    @pyqtSlot(str)
    def update_worker_count(self, workers_json: str):
        """Update worker count from registry data."""
        try:
            workers_data = json.loads(workers_json) if workers_json.strip() else []
            worker_count = len(workers_data)
            self.workers_label.setText(f"Workers: {worker_count}")
        except Exception as e:
            print(f"Error updating worker count: {e}")

    @pyqtSlot(float)
    def update_energy(self, energy: float):
        """Update energy display and color."""
        self.energy_label.setText(f"Energy: {energy:.2f}")
        
        # Update color based on energy level for better visibility
        if energy > 0.7:
            self.energy_label.setStyleSheet("color: lightgreen; font-weight: bold;")
        elif energy > 0.4:
            self.energy_label.setStyleSheet("color: yellow; font-weight: bold;")
        elif energy > 0.2:
            self.energy_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.energy_label.setStyleSheet("color: red; font-weight: bold;")

    @pyqtSlot(str)
    def append_warning(self, text):
        # append with newline
        prev = self.warn_text.toPlainText()
        if prev:
            self.warn_text.setPlainText(prev + "\n" + text)
        else:
            self.warn_text.setPlainText(text)
        # scroll to bottom
        self.warn_text.verticalScrollBar().setValue(self.warn_text.verticalScrollBar().maximum())

    @pyqtSlot(str)
    def append_info(self, text):
        # append with newline to same text area as warnings
        prev = self.warn_text.toPlainText()
        if prev:
            self.warn_text.setPlainText(prev + "\n" + text)
        else:
            self.warn_text.setPlainText(text)
        # scroll to bottom
        self.warn_text.verticalScrollBar().setValue(self.warn_text.verticalScrollBar().maximum())

    def start_all(self):
        self.engine.start()
        self.feature_engine.start()

    def stop_all(self):
        self.engine.stop()
        self.feature_engine.stop()

    def on_wipe(self):
        try:
            self.engine.stop()
            self.feature_engine.stop()
        except Exception:
            pass
        wipe_state()

    def on_open_heatmap(self):
        try:
            if not hasattr(self, 'heatmap_window') or self.heatmap_window is None:
                self.heatmap_window = HeatmapWindow(self)
            self.heatmap_window.show()
            self.heatmap_window.raise_()
            self.heatmap_window.activateWindow()
        except Exception:
            pass

    def on_heatmap_hit(self, nx: float, ny: float):
        try:
            if hasattr(self, 'heatmap_window') and self.heatmap_window is not None:
                self.heatmap_window.on_hash_hit(nx, ny)
        except Exception:
            pass



    def closeEvent(self, event):
        self.engine.stop()
        self.feature_engine.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())

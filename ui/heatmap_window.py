from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QThread
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
import cv2
import os
import json
import time
import threading
import webbrowser
from flask import Flask, render_template, jsonify


class HeatmapWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Hash Activity Heatmap')

        # Load configuration (overridable via state/heatmap_config.json)
        self._config = self._load_config()
        self.grid_size = int(self._config.get('grid_size', 256))
        # Time-based decay: half-life in seconds for activity layer
        self.activity_half_life_s = float(self._config.get('activity_half_life_s', 2.0))
        self.hit_boost = float(self._config.get('hit_boost', 1.0))
        self.sigma_px = int(self._config.get('splat_sigma_px', 6))
        # Persistent layer config
        self.persist_boost = float(self._config.get('persist_boost', 0.4))
        self.persist_sigma_px = int(self._config.get('persist_sigma_px', 4))
        self.persist_alpha = float(self._config.get('persist_alpha', 0.35))
        # Redraw cadence only; decay uses real time
        self.fps = int(self._config.get('fps', 20))
        self.output_width = int(self._config.get('window_width', 640))
        self.output_height = int(self._config.get('window_height', 480))

        # UI
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Add control buttons
        button_layout = QHBoxLayout()
        self.open_3d_button = QPushButton('Open 3D Web Viewer', self)
        self.open_3d_button.clicked.connect(self.open_3d_viewer)
        button_layout.addWidget(self.open_3d_button)
        layout.addLayout(button_layout)
        
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFixedSize(self.output_width, self.output_height)
        self.label.setStyleSheet('border: 1px solid black;')
        layout.addWidget(self.label)
        
        # Flask server setup
        self.flask_app = None
        self.flask_thread = None
        self.server_running = False
        self.port = 5000

        # Heat state: activity (decays) and persist (no decay)
        self.activity = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.persist = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.kernel_activity = self._make_gaussian_kernel(self.sigma_px)
        self.kernel_persist = self._make_gaussian_kernel(self.persist_sigma_px)

        # Timer for decay/redraw
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(int(1000 / max(1, self.fps)))
        self._last_time_s = time.time()

    def _load_config(self) -> dict:
        try:
            # project root is parent of ui/
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cfg_path = os.path.join(root, 'state', 'heatmap_config.json')
            if os.path.exists(cfg_path):
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _make_gaussian_kernel(self, sigma_px: int) -> np.ndarray:
        sigma = max(1, int(sigma_px))
        radius = int(3 * sigma)
        size = 2 * radius + 1
        ax = np.arange(-radius, radius + 1, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * (sigma * sigma)))
        kernel = kernel / np.max(kernel)
        return kernel.astype(np.float32)

    @pyqtSlot(float, float)
    def on_hash_hit(self, norm_x: float, norm_y: float):
        # Clamp to [0,1]
        nx = float(min(max(norm_x, 0.0), 1.0))
        ny = float(min(max(norm_y, 0.0), 1.0))
        cx = int(nx * (self.grid_size - 1))
        cy = int(ny * (self.grid_size - 1))
        # Activity (decaying trail)
        self._splat(self.activity, self.kernel_activity, cy, cx, self.hit_boost)
        # Persist (all hashes ever seen, faint)
        self._splat(self.persist, self.kernel_persist, cy, cx, self.persist_boost, mode='add_clip')

    def _splat(self, target: np.ndarray, kernel: np.ndarray, center_row: int, center_col: int, strength: float, mode: str = 'add'):
        k = kernel
        kr, kc = k.shape
        rr = kr // 2
        rc = kc // 2
        r0 = max(0, center_row - rr)
        c0 = max(0, center_col - rc)
        r1 = min(self.grid_size, center_row + rr + 1)
        c1 = min(self.grid_size, center_col + rc + 1)

        k_r0 = rr - (center_row - r0)
        k_c0 = rc - (center_col - c0)
        k_r1 = k_r0 + (r1 - r0)
        k_c1 = k_c0 + (c1 - c0)

        try:
            if mode == 'add_clip':
                target[r0:r1, c0:c1] = np.clip(target[r0:r1, c0:c1] + strength * k[k_r0:k_r1, k_c0:k_c1], 0.0, 1.0)
            else:
                target[r0:r1, c0:c1] += strength * k[k_r0:k_r1, k_c0:k_c1]
        except Exception:
            pass

    def _to_qpixmap(self) -> QPixmap:
        # Activity layer image (thermal)
        a = self.activity
        ma = float(np.max(a))
        if ma > 1e-6:
            a_norm = (a / ma)
            img_activity = self._apply_colormap(a_norm)
        else:
            img_activity = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)

        # Persistent layer image (dim)
        p = self.persist
        mp = float(np.max(p))
        if mp > 1e-6:
            p_norm = (p / mp)
            img_persist = self._apply_colormap(p_norm)
        else:
            img_persist = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)

        # Blend: persistent faint underneath activity
        blended = cv2.addWeighted(img_persist, float(self.persist_alpha), img_activity, 1.0, 0.0)

        # Resize for display
        img_resized = cv2.resize(blended, (self.output_width, self.output_height), interpolation=cv2.INTER_NEAREST)
        # Convert BGR to RGB for Qt
        rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        hgt, wdt, ch = rgb.shape
        qimg = QImage(rgb.data, wdt, hgt, ch * wdt, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _apply_colormap(self, norm01: np.ndarray) -> np.ndarray:
        # Custom blue->cyan->yellow->red gradient
        x = np.clip(norm01, 0.0, 1.0)
        r = np.clip(4.0 * (x - 0.75), 0.0, 1.0)
        g = np.clip(4.0 * (x - 0.50), 0.0, 1.0)
        b = np.clip(4.0 * (0.25 - x), 0.0, 1.0)
        # Convert to BGR uint8 image
        img = np.stack([
            (b * 255.0).astype(np.uint8),
            (g * 255.0).astype(np.uint8),
            (r * 255.0).astype(np.uint8)
        ], axis=2)
        return img

    def _on_tick(self):
        try:
            # Time-based exponential decay for activity layer
            now = time.time()
            dt = max(0.0, now - self._last_time_s)
            self._last_time_s = now
            hl = max(1e-3, float(self.activity_half_life_s))
            # decay factor = 0.5 ** (dt / half_life)
            factor = float(pow(0.5, dt / hl))
            self.activity *= factor
            # Redraw
            self.label.setPixmap(self._to_qpixmap())
        except Exception:
            pass

    def open_3d_viewer(self):
        """Start Flask server and open 3D web viewer"""
        if not self.server_running:
            self.start_flask_server()
        
        # Open browser to the 3D viewer
        url = f"http://localhost:{self.port}"
        webbrowser.open(url)
    
    def start_flask_server(self):
        """Start Flask server in a separate thread"""
        if self.server_running:
            return
            
        self.flask_app = Flask(__name__, 
                             template_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates'),
                             static_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static'))
        
        # Setup routes
        self.setup_flask_routes()
        
        # Start server in thread
        self.flask_thread = threading.Thread(target=self.run_flask_server, daemon=True)
        self.flask_thread.start()
        self.server_running = True
        
    def run_flask_server(self):
        """Run Flask server"""
        try:
            self.flask_app.run(host='localhost', port=self.port, debug=False, use_reloader=False)
        except Exception as e:
            print(f"Flask server error: {e}")
    
    def setup_flask_routes(self):
        """Setup Flask routes for the 3D viewer"""
        
        @self.flask_app.route('/')
        def index():
            return render_template('heatmap_3d.html')
        
        @self.flask_app.route('/api/heatmap_data')
        def get_heatmap_data():
            """API endpoint to get current heatmap data"""
            # Convert heatmap data to JSON format
            activity_data = self.activity.tolist()
            persist_data = self.persist.tolist()
            
            # Create clusters based on activity
            clusters = self.create_clusters()
            
            return jsonify({
                'activity': activity_data,
                'persist': persist_data,
                'clusters': clusters,
                'grid_size': self.grid_size,
                'timestamp': time.time()
            })
        
        @self.flask_app.route('/api/update')
        def update_data():
            """API endpoint for real-time updates"""
            return jsonify({
                'activity_max': float(np.max(self.activity)),
                'persist_max': float(np.max(self.persist)),
                'timestamp': time.time()
            })
    
    def create_clusters(self):
        """Create cluster data from heatmap for 3D visualization"""
        clusters = []
        
        # Find local maxima in activity data
        from scipy.ndimage import maximum_filter
        try:
            # Find peaks in activity data
            local_maxima = maximum_filter(self.activity, size=10) == self.activity
            peaks = np.where((local_maxima) & (self.activity > 0.1))
            
            for i in range(len(peaks[0])):
                y, x = peaks[0][i], peaks[1][i]
                intensity = float(self.activity[y, x])
                persist_intensity = float(self.persist[y, x])
                
                # Convert grid coordinates to normalized coordinates
                norm_x = x / self.grid_size
                norm_y = y / self.grid_size
                
                clusters.append({
                    'id': i,
                    'x': norm_x,
                    'y': norm_y,
                    'z': intensity * 2.0,  # Scale for 3D height
                    'intensity': intensity,
                    'persist_intensity': persist_intensity,
                    'size': max(0.1, intensity * 0.5)  # Size based on intensity
                })
        except ImportError:
            # Fallback if scipy not available
            pass
        
        return clusters
    
    def stop_flask_server(self):
        """Stop Flask server"""
        self.server_running = False
        # Note: Flask doesn't have a clean shutdown method, 
        # the thread will end when the window closes

    def closeEvent(self, event):
        try:
            if self.timer.isActive():
                self.timer.stop()
            self.stop_flask_server()
        except Exception:
            pass
        event.accept()



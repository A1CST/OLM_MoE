from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, pyqtSlot
from utils import convert_cv_qt, convert_rgb_qt
import numpy as np
import threading
import webbrowser
import os
import json
import time
from flask import Flask, render_template, jsonify


class DisplayWindow(QMainWindow):
    """Generic display window for showing frames."""
    
    def __init__(self, title: str, width: int = 640, height: int = 480):
        super().__init__()
        self.setWindowTitle(title)
        self.setGeometry(100, 100, width + 40, height + 60)  # Extra space for borders
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QVBoxLayout(self.central_widget)
        
        self.display_label = QLabel(self)
        self.display_label.setFixedSize(width, height)
        self.display_label.setStyleSheet("border: 1px solid black;")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setText("No data yet...")
        
        self.layout.addWidget(self.display_label)
        
    @pyqtSlot(np.ndarray)
    def update_frame(self, frame):
        """Update the displayed frame."""
        try:
            qt_img = convert_cv_qt(frame)
            self.display_label.setPixmap(qt_img)
        except Exception as e:
            print(f"Error updating display frame: {e}")


class ScreenCaptureWindow(DisplayWindow):
    """Window for showing screen capture."""
    
    def __init__(self):
        super().__init__("Screen Capture", 640, 480)


class HashMapWindow(DisplayWindow):
    """Window for showing hash map visualization with 3D web viewer."""
    
    def __init__(self):
        super().__init__("Hash Map (Hamming in 3D)", 640, 480)
        
        # Add 3D web viewer button
        button_layout = QHBoxLayout()
        self.open_3d_button = QPushButton('Open 3D Web Viewer', self)
        self.open_3d_button.clicked.connect(self.open_3d_viewer)
        button_layout.addWidget(self.open_3d_button)
        self.layout.addLayout(button_layout)
        
        # Flask server setup
        self.flask_app = None
        self.flask_thread = None
        self.server_running = False
        self.port = 5001  # Different port from heatmap
        self.current_frame = None
        self.hash_data = None
        
    @pyqtSlot(np.ndarray)
    def update_frame(self, frame):
        """Update with RGB frame and store for 3D viewer."""
        try:
            qt_img = convert_rgb_qt(frame)
            self.display_label.setPixmap(qt_img)
            
            # Store current frame for 3D viewer
            self.current_frame = frame.copy()
            
            # Extract hash data from frame (assuming it contains hash visualization)
            self.extract_hash_data(frame)
            
        except Exception as e:
            print(f"Error updating hash map frame: {e}")
    
    def extract_hash_data(self, frame):
        """Extract hash data from the visualization frame for 3D display."""
        try:
            # Convert frame to grayscale for analysis
            if len(frame.shape) == 3:
                gray = np.mean(frame, axis=2)
            else:
                gray = frame
            
            # Debug: Print frame statistics
            print(f"Frame shape: {frame.shape}, Gray range: {gray.min():.2f}-{gray.max():.2f}")
            
            # Try multiple methods to find hash points
            points = []
            intensities = []
            
            # Method 1: Bright spots (original method)
            threshold = np.percentile(gray, 85)  # Top 15% brightest pixels
            bright_spots = np.where(gray > threshold)
            if len(bright_spots[0]) > 0:
                points.extend(list(zip(bright_spots[1], bright_spots[0])))
                intensities.extend(gray[bright_spots].tolist())
            
            # Method 2: Edge detection for hash-like patterns
            from scipy import ndimage
            edges = ndimage.sobel(gray)
            edge_threshold = np.percentile(edges, 80)
            edge_points = np.where(edges > edge_threshold)
            if len(edge_points[0]) > 0:
                # Sample every 10th edge point to avoid too many points
                sampled_indices = np.arange(0, len(edge_points[0]), 10)
                points.extend(list(zip(edge_points[1][sampled_indices], edge_points[0][sampled_indices])))
                intensities.extend(edges[edge_points][sampled_indices].tolist())
            
            # Method 3: If no points found, create some test points
            if len(points) == 0:
                print("No hash points detected, creating test points")
                h, w = gray.shape
                # Create a grid of test points
                for i in range(0, h, 50):
                    for j in range(0, w, 50):
                        points.append((j, i))
                        intensities.append(128 + (i + j) % 127)  # Varying intensity
            
            print(f"Extracted {len(points)} hash points")
            
            self.hash_data = {
                'points': points,
                'intensities': intensities,
                'frame_shape': frame.shape,
                'timestamp': time.time()
            }
            
        except Exception as e:
            print(f"Error extracting hash data: {e}")
            # Create fallback test data
            self.hash_data = {
                'points': [(100, 100), (200, 150), (300, 200), (150, 250), (250, 300)],
                'intensities': [200, 150, 180, 120, 160],
                'frame_shape': frame.shape if 'frame' in locals() else [480, 640, 3],
                'timestamp': time.time()
            }
    
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
        """Setup Flask routes for the 3D hash map viewer"""
        
        @self.flask_app.route('/')
        def index():
            return render_template('hashmap_3d.html')
        
        @self.flask_app.route('/api/hashmap_data')
        def get_hashmap_data():
            """API endpoint to get current hash map data"""
            print(f"API called: hash_data is {self.hash_data}")
            if self.hash_data is None:
                print("Returning empty data")
                return jsonify({
                    'points': [],
                    'intensities': [],
                    'frame_shape': [640, 480, 3],
                    'timestamp': time.time()
                })
            
            print(f"Returning {len(self.hash_data.get('points', []))} points")
            return jsonify(self.hash_data)
        
        @self.flask_app.route('/api/update')
        def update_data():
            """API endpoint for real-time updates"""
            point_count = len(self.hash_data['points']) if self.hash_data else 0
            return jsonify({
                'point_count': point_count,
                'timestamp': time.time()
            })
    
    def stop_flask_server(self):
        """Stop Flask server"""
        self.server_running = False
    
    def closeEvent(self, event):
        """Handle window close event"""
        try:
            self.stop_flask_server()
        except Exception:
            pass
        event.accept()


class LiveVAEWindow(DisplayWindow):
    """Window for showing live VAE predictions."""
    
    def __init__(self):
        super().__init__("Live VAE Prediction", 640, 480)
        
    @pyqtSlot(np.ndarray)
    def update_frame(self, frame):
        """Update with RGB frame."""
        try:
            qt_img = convert_rgb_qt(frame)
            self.display_label.setPixmap(qt_img)
        except Exception as e:
            print(f"Error updating live VAE frame: {e}")


class DPipeWindow(DisplayWindow):
    """Window for showing D-pipe predictions."""
    
    def __init__(self):
        super().__init__("D-pipe Prediction", 640, 480)
        
    @pyqtSlot(np.ndarray)
    def update_frame(self, frame):
        """Update with RGB frame."""
        try:
            qt_img = convert_rgb_qt(frame)
            self.display_label.setPixmap(qt_img)
        except Exception as e:
            print(f"Error updating D-pipe frame: {e}")
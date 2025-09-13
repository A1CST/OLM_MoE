from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPlainTextEdit, QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, pyqtSlot
import json


class WorkerRegistryWindow(QMainWindow):
    """Window for monitoring WorkerRegistry - tracks spawned workers and their stats."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Worker Registry Monitor")
        self.setGeometry(150, 150, 800, 600)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Title
        self.title_label = QLabel("Worker Registry Status")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: blue;")
        self.layout.addWidget(self.title_label)
        
        # Worker count summary
        self.summary_label = QLabel("Workers: 0 | Active: 0 | Types: predictor")
        self.summary_label.setAlignment(Qt.AlignCenter)
        self.summary_label.setStyleSheet("font-size: 12px; color: green; font-weight: bold;")
        self.layout.addWidget(self.summary_label)
        
        # Workers table
        self.workers_table = QTableWidget()
        self.workers_table.setColumnCount(5)
        self.workers_table.setHorizontalHeaderLabels(["Name", "Type", "Handle", "Control Dim", "Stats"])
        self.workers_table.setAlternatingRowColors(True)
        self.workers_table.setStyleSheet("""
            QTableWidget { 
                gridline-color: gray; 
                background-color: rgba(255, 255, 255, 200);
            }
            QHeaderView::section { 
                background-color: rgba(100, 149, 237, 150);
                color: white;
                font-weight: bold;
                padding: 4px;
            }
        """)
        self.layout.addWidget(self.workers_table)
        
        # Raw JSON display
        self.json_label = QLabel("Raw Worker Snapshot JSON:")
        self.json_label.setStyleSheet("font-weight: bold;")
        self.layout.addWidget(self.json_label)
        
        self.json_text = QPlainTextEdit()
        self.json_text.setReadOnly(True)
        self.json_text.setFixedHeight(150)
        self.json_text.setStyleSheet("""
            QPlainTextEdit {
                background-color: rgba(0, 0, 0, 200);
                color: lime;
                font-family: 'Courier New';
                font-size: 10px;
            }
        """)
        self.layout.addWidget(self.json_text)
        
    @pyqtSlot(str)
    def update_workers_json(self, workers_json: str):
        """Update the workers display from JSON snapshot."""
        try:
            workers_data = json.loads(workers_json) if workers_json.strip() else []
            
            # Update summary
            total_workers = len(workers_data)
            types = set(w.get('wtype', 'unknown') for w in workers_data)
            active_workers = sum(1 for w in workers_data if w.get('handle') is not None)
            
            self.summary_label.setText(f"Workers: {total_workers} | Active: {active_workers} | Types: {', '.join(types)}")
            
            # Update table
            self.workers_table.setRowCount(total_workers)
            for i, worker in enumerate(workers_data):
                self.workers_table.setItem(i, 0, QTableWidgetItem(str(worker.get('name', 'N/A'))))
                self.workers_table.setItem(i, 1, QTableWidgetItem(str(worker.get('wtype', 'N/A'))))
                self.workers_table.setItem(i, 2, QTableWidgetItem('Active' if worker.get('handle') else 'None'))
                self.workers_table.setItem(i, 3, QTableWidgetItem(str(worker.get('control_dim', 'N/A'))))
                stats = worker.get('stats', {})
                stats_str = f"calls: {stats.get('calls', 0)}, errors: {stats.get('errors', 0)}"
                self.workers_table.setItem(i, 4, QTableWidgetItem(stats_str))
                
            self.workers_table.resizeColumnsToContents()
            
            # Update raw JSON
            self.json_text.setPlainText(json.dumps(workers_data, indent=2))
            
        except Exception as e:
            self.json_text.setPlainText(f"Error parsing workers data: {e}")


class IILSTMRoutingWindow(QMainWindow):
    """Window for monitoring IILSTM executive routing decisions."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IILSTM Executive Routing")
        self.setGeometry(200, 200, 900, 700)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Title
        self.title_label = QLabel("IILSTM Executive Dashboard")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: red;")
        self.layout.addWidget(self.title_label)
        
        # Current routing status
        self.routing_row = QHBoxLayout()
        self.k_label = QLabel("k: 0")
        self.k_label.setStyleSheet("color: orange; font-size: 14px; font-weight: bold;")
        self.selected_label = QLabel("Selected: None")
        self.selected_label.setStyleSheet("color: green; font-size: 12px; font-weight: bold;")
        self.routing_row.addWidget(self.k_label)
        self.routing_row.addWidget(self.selected_label)
        self.layout.addLayout(self.routing_row)
        
        # Drivers panel
        self.drivers_label = QLabel("Driver Values:")
        self.drivers_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.layout.addWidget(self.drivers_label)
        
        self.drivers_row = QHBoxLayout()
        self.nov_driver_label = QLabel("Novelty: 0.000")
        self.nov_driver_label.setStyleSheet("color: orange; font-weight: bold;")
        self.energy_driver_label = QLabel("Energy: 1.000")
        self.energy_driver_label.setStyleSheet("color: green; font-weight: bold;")
        self.sleep_driver_label = QLabel("Sleep: 0.000")
        self.sleep_driver_label.setStyleSheet("color: purple; font-weight: bold;")
        self.reserve_driver_label = QLabel("Reserve: 0.000")
        self.reserve_driver_label.setStyleSheet("color: gray; font-weight: bold;")
        
        self.drivers_row.addWidget(self.nov_driver_label)
        self.drivers_row.addWidget(self.energy_driver_label)
        self.drivers_row.addWidget(self.sleep_driver_label)
        self.drivers_row.addWidget(self.reserve_driver_label)
        self.layout.addLayout(self.drivers_row)
        
        # Routing logits display
        self.logits_label = QLabel("Routing Logits (Raw):")
        self.logits_label.setStyleSheet("font-weight: bold;")
        self.layout.addWidget(self.logits_label)
        
        self.routing_logits_text = QPlainTextEdit()
        self.routing_logits_text.setReadOnly(True)
        self.routing_logits_text.setFixedHeight(80)
        self.routing_logits_text.setStyleSheet("""
            QPlainTextEdit {
                background-color: rgba(0, 50, 0, 200);
                color: lightgreen;
                font-family: 'Courier New';
                font-size: 10px;
            }
        """)
        self.layout.addWidget(self.routing_logits_text)
        
        # K logits display
        self.k_logits_label = QLabel("K Selection Logits:")
        self.k_logits_label.setStyleSheet("font-weight: bold;")
        self.layout.addWidget(self.k_logits_label)
        
        self.k_logits_text = QPlainTextEdit()
        self.k_logits_text.setReadOnly(True)
        self.k_logits_text.setFixedHeight(60)
        self.k_logits_text.setStyleSheet("""
            QPlainTextEdit {
                background-color: rgba(50, 0, 50, 200);
                color: lightblue;
                font-family: 'Courier New';
                font-size: 10px;
            }
        """)
        self.layout.addWidget(self.k_logits_text)
        
        # Selected workers details
        self.selected_workers_label = QLabel("Selected Workers Details:")
        self.selected_workers_label.setStyleSheet("font-weight: bold;")
        self.layout.addWidget(self.selected_workers_label)
        
        self.selected_workers_text = QPlainTextEdit()
        self.selected_workers_text.setReadOnly(True)
        self.selected_workers_text.setFixedHeight(200)
        self.selected_workers_text.setStyleSheet("""
            QPlainTextEdit {
                background-color: rgba(50, 50, 0, 200);
                color: yellow;
                font-family: 'Courier New';
                font-size: 10px;
            }
        """)
        self.layout.addWidget(self.selected_workers_text)
        
    @pyqtSlot(str)
    def update_routing_data(self, components_text: str):
        """Update routing display from novelty components text with IILSTM data."""
        try:
            # This gets called with the novelty components text that contains routing info
            # For now, just display it - we'll parse it properly later
            self.selected_workers_text.setPlainText(f"Debug: {components_text}")
        except Exception as e:
            self.selected_workers_text.setPlainText(f"Error: {e}")
            
    def update_drivers(self, novelty: float, energy: float, sleep_pressure: float):
        """Update driver values display."""
        self.nov_driver_label.setText(f"Novelty: {novelty:.3f}")
        self.energy_driver_label.setText(f"Energy: {energy:.3f}")
        self.sleep_driver_label.setText(f"Sleep: {sleep_pressure:.3f}")
        
    def update_routing_logits(self, logits: list):
        """Update routing logits display."""
        logits_str = ", ".join(f"{x:.3f}" for x in logits)
        self.routing_logits_text.setPlainText(f"[{logits_str}]")
        
    def update_k_logits(self, k_logits: list):
        """Update k selection logits."""
        k_logits_str = ", ".join(f"{x:.3f}" for x in k_logits)
        self.k_logits_text.setPlainText(f"[{k_logits_str}]")
        
    def update_selected_workers(self, k: int, selected: list):
        """Update selected workers display."""
        self.k_label.setText(f"k: {k}")
        if selected:
            selected_names = ", ".join(w.get('name', 'unknown') for w in selected)
            self.selected_label.setText(f"Selected: {selected_names}")
            
            details = []
            for w in selected:
                ctrl_preview = str(w.get('ctrl', []))[:50] + "..." if len(str(w.get('ctrl', []))) > 50 else str(w.get('ctrl', []))
                details.append(f"Worker {w.get('idx', '?')}: {w.get('name', 'unknown')}")
                details.append(f"  Control: {ctrl_preview}")
                details.append("")
            self.selected_workers_text.setPlainText("\n".join(details))
        else:
            self.selected_label.setText("Selected: None")
            self.selected_workers_text.setPlainText("No workers selected")
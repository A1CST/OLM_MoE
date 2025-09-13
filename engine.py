
import threading
import time
from PyQt5.QtCore import QObject, pyqtSignal, QThread
import mss
import numpy as np

class Engine(QObject):
    new_frame = pyqtSignal(np.ndarray)

    def __init__(self, tps=20):
        super().__init__()
        self.tps = tps
        self.running = False
        self._thread = None
        

    def start(self):
        if not self.running:
            self.running = True
            self._thread = threading.Thread(target=self._run)
            self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join()

    def _run(self):
        self._sct = mss.mss()
        while self.running:
            start_time = time.time()
            
            # Take a screenshot
            monitor = self._sct.monitors[1]
            sct_img = self._sct.grab(monitor)
            frame = np.array(sct_img)
            
            self.new_frame.emit(frame)

            elapsed_time = time.time() - start_time
            sleep_time = 1/self.tps - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    def __del__(self):
        self.stop()

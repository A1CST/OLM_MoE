
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import cv2
import os
import shutil

def convert_cv_qt(cv_img):
    """Convert from an opencv image to QPixmap"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
    return QPixmap.fromImage(p)


def wipe_state():
    """Delete logs, __pycache__, and model/checkpoint/hash files under project root/state.

    Removes:
    - state/logs/*
    - state/*.npz, *.pt, *.json (predictors, vae, lsh, hash db)
    - top-level *.npz related to checkpoints (legacy)
    - all __pycache__ directories recursively
    """
    root = os.path.dirname(os.path.abspath(__file__))
    state_dir = os.path.join(root, 'state')

    # Remove state logs directory
    logs_dir = os.path.join(state_dir, 'logs')
    try:
        if os.path.isdir(logs_dir):
            shutil.rmtree(logs_dir, ignore_errors=True)
    except Exception:
        pass

    # Remove known checkpoint files in state
    state_patterns = [
        'frozen_encoder.npz',
        'live_vae.npz',
        'lsh_state.npz',
        'hash_db.npz',
        'pc_lstm.pt',
        'dpipe_lstm.pt',
        'hash_predictor.json',
        'live_latent_predictor.npz',
        'project_summary.json',  # keep? remove upon request
    ]
    for name in state_patterns:
        p = os.path.join(state_dir, name)
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    # Remove legacy top-level checkpoints
    top_level = [
        'frozen_encoder.npz',
        'live_vae.npz',
        'lsh_state.npz',
        'hash_db.npz',
    ]
    for name in top_level:
        p = os.path.join(root, name)
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    # Remove all __pycache__ directories recursively
    for dirpath, dirnames, _ in os.walk(root):
        for d in list(dirnames):
            if d == '__pycache__':
                try:
                    shutil.rmtree(os.path.join(dirpath, d), ignore_errors=True)
                except Exception:
                    pass

def convert_rgb_qt(rgb_img):
    import numpy as np
    arr = np.ascontiguousarray(rgb_img, dtype=np.uint8)
    h, w, ch = arr.shape
    bytes_per_line = ch * w
    # Use tobytes() to avoid memoryview issues on some PyQt5 builds
    convert_to_Qt_format = QImage(arr.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
    p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
    return QPixmap.fromImage(p)

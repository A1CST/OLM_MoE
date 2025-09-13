import threading
import time
from typing import Tuple
import numpy as np
import mss
from PyQt5.QtCore import QObject, pyqtSignal
from lsh_system import LSHSystem
import cv2
import os
import json
import time
import sys
from datetime import datetime
from collections import deque
import torch
import torch.nn as nn
import copy
from typing import Dict, Any, List, Tuple
import torch.optim as optim
from p_C_pipe import PCPipe
from D_pipe import DPipe
from audio_engine import AudioEngine
from iilstm import IILSTM
from workers.registry import WorkerRegistry
from action_bus import ActionBus
from workers.predictor_worker import PredictorWorker
from workers.mouse_worker import MouseWorker
from workers.key_worker import KeyWorker
from workers.audio_worker import AudioWorker
from experience_buffer import ExperienceBuffer
from sleep_manager import SleepManager
import torch.nn.functional as F
from vae_sup import VaeSup


class SimpleEncoder:
    """Encoder-only model: linear projection to latent space (no decoder)."""

    def __init__(self, input_dim: int, latent_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.encoder_weight = (rng.standard_normal((latent_dim, input_dim)) * 0.01).astype(np.float32)
        self.encoder_bias = np.zeros((latent_dim,), dtype=np.float32)

    def encode(self, x: np.ndarray) -> np.ndarray:
        return self.encoder_weight @ x + self.encoder_bias


class SimpleVAE:
    """A tiny VAE-like model for demonstration/feature extraction.

    This is NOT a full VAE; it's a lightweight linear encoder/decoder stub
    suitable for fast, dependency-free demo. The "features" are the encoder
    latent vector. The train() method performs a single step of SGD to
    minimize simple reconstruction error.
    """

    def __init__(self, input_dim: int, latent_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # Encoder: input_dim -> latent_dim
        self.encoder_weight = (rng.standard_normal((latent_dim, input_dim)) * 0.01).astype(np.float32)
        self.encoder_bias = np.zeros((latent_dim,), dtype=np.float32)
        # Decoder: latent_dim -> input_dim
        self.decoder_weight = (rng.standard_normal((input_dim, latent_dim)) * 0.01).astype(np.float32)
        self.decoder_bias = np.zeros((input_dim,), dtype=np.float32)
        # Training hyperparameters
        self.learning_rate = 1e-3

    def encode(self, x: np.ndarray) -> np.ndarray:
        # x: (input_dim,)
        return self.encoder_weight @ x + self.encoder_bias

    def decode(self, z: np.ndarray) -> np.ndarray:
        return self.decoder_weight @ z + self.decoder_bias

    def reconstruct(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat

    def train_step(self, x: np.ndarray, frozen_z: np.ndarray | None = None, beta: float = 0.0) -> Tuple[np.ndarray, float, float]:
        """One SGD step on total loss = recon + beta * consistency.
        Returns (z, recon_loss, total_loss).
        """
        # Forward
        z = self.encode(x)  # (latent_dim,)
        x_hat = self.decode(z)  # (input_dim,)
        diff = x_hat - x
        recon_loss = float(np.mean(diff ** 2))

        # Backprop manually for the simple linear autoencoder
        # dL/dx_hat = 2*(x_hat - x)/N
        N = x.shape[0]
        grad_x_hat = (2.0 / N) * diff  # (input_dim,)

        # x_hat = W_dec z + b_dec
        grad_W_dec = np.outer(grad_x_hat, z)  # (input_dim, latent_dim)
        grad_b_dec = grad_x_hat  # (input_dim,)

        # z = W_enc x + b_enc
        grad_z = self.decoder_weight.T @ grad_x_hat  # (latent_dim,)
        if frozen_z is not None and beta > 0.0:
            Nz = z.shape[0]
            grad_z = grad_z + (2.0 * beta / Nz) * (z - frozen_z)
        grad_W_enc = np.outer(grad_z, x)  # (latent_dim, input_dim)
        grad_b_enc = grad_z  # (latent_dim,)

        # SGD update
        lr = self.learning_rate
        self.decoder_weight -= lr * grad_W_dec
        self.decoder_bias -= lr * grad_b_dec
        self.encoder_weight -= lr * grad_W_enc
        self.encoder_bias -= lr * grad_b_enc

        total_loss = recon_loss
        if frozen_z is not None and beta > 0.0:
            total_loss += float(beta * np.mean((z - frozen_z) ** 2))
        return z, recon_loss, total_loss


class FeatureEngine(QObject):
    """Captures the screen and emits features for a frozen and a live VAE."""

    frozen_features = pyqtSignal(str)
    live_features = pyqtSignal(str)
    frozen_latent_np = pyqtSignal(np.ndarray)
    lsh_code_bits = pyqtSignal(str)
    lsh_chained_hash = pyqtSignal(str)
    hashmap_frame = pyqtSignal(np.ndarray)
    hashmap_stats = pyqtSignal(str)
    # Normalized 2D coords (x,y) in [0,1] for current hash position in the map
    heatmap_hit = pyqtSignal(float, float)
    tick_update = pyqtSignal(int)
    tps_update = pyqtSignal(float)
    live_prediction_frame = pyqtSignal(np.ndarray)
    predicted_hash_frozen = pyqtSignal(str)
    predicted_hash_live = pyqtSignal(str)
    prediction_accuracy_text = pyqtSignal(str)
    dpipe_prediction_frame = pyqtSignal(np.ndarray)
    dpipe_accuracy_text = pyqtSignal(str)
    diag_warning = pyqtSignal(str)
    novelty_value = pyqtSignal(float)               # scalar novelty (EMA-smoothed)
    novelty_components_text = pyqtSignal(str)       # debug string of subcomponents
    worker_registry_data = pyqtSignal(str)          # JSON string of worker registry snapshot
    energy_value = pyqtSignal(float)                # current energy level
    sleep_pressure_value = pyqtSignal(float)        # 0..1 sleep pressure for the GUI
    sleep_state_text = pyqtSignal(str)              # "awake" | "nrem" | "rem"
    diag_info = pyqtSignal(str)                     # informational status lines

    def __init__(self, tps: int = 10, downsample_width: int = 64, downsample_height: int = 36,
                 latent_dim: int = 32, audio_latent_dim: int = 16):
        super().__init__()
        self.tps = tps
        self.downsample_width = downsample_width
        self.downsample_height = downsample_height
        self.latent_dim = latent_dim
        self.audio_latent_dim = audio_latent_dim
        self._thread = None
        self.running = False

        # Vision processing
        vision_input_dim = 3 * self.downsample_width * self.downsample_height
        self.frozen_vision_encoder = SimpleEncoder(input_dim=vision_input_dim, latent_dim=self.latent_dim, seed=42)
        self.live_vae = SimpleVAE(input_dim=vision_input_dim, latent_dim=self.latent_dim, seed=7)
        
        # Audio processing
        audio_input_dim = 1024  # Audio chunk size from AudioEngine
        self.frozen_audio_encoder = SimpleEncoder(input_dim=audio_input_dim, latent_dim=self.audio_latent_dim, seed=84)
        
        # Ensure all encoders are initialized with proper float32 types
        self.frozen_vision_encoder.encoder_weight = self.frozen_vision_encoder.encoder_weight.astype(np.float32)
        self.frozen_vision_encoder.encoder_bias = self.frozen_vision_encoder.encoder_bias.astype(np.float32)
        self.frozen_audio_encoder.encoder_weight = self.frozen_audio_encoder.encoder_weight.astype(np.float32)  
        self.frozen_audio_encoder.encoder_bias = self.frozen_audio_encoder.encoder_bias.astype(np.float32)
        self.live_vae.encoder_weight = self.live_vae.encoder_weight.astype(np.float32)
        self.live_vae.encoder_bias = self.live_vae.encoder_bias.astype(np.float32)
        self.live_vae.decoder_weight = self.live_vae.decoder_weight.astype(np.float32)
        self.live_vae.decoder_bias = self.live_vae.decoder_bias.astype(np.float32)
        
        # Audio engine
        self.audio_engine = AudioEngine()
        self.audio_engine.audio_chunk_ready.connect(self._on_audio_chunk)
        self._latest_audio_latent = np.zeros(self.audio_latent_dim)
        
        # Combined latent dimension for downstream processing
        self.combined_latent_dim = self.latent_dim + self.audio_latent_dim

        # LSH subsystem - now operates on combined vision+audio latents
        self.lsh_bits = 128
        self.lsh = LSHSystem(latent_dim=self.combined_latent_dim, num_bits=self.lsh_bits, seed=123)

        # Hash map state
        self._codes_bits: list[np.ndarray] = []
        self._codes_seen: set[str] = set()
        self._code_to_index: dict[str, int] = {}
        self._coords_3d: np.ndarray | None = None  # shape (n, 3)
        self._D: np.ndarray | None = None  # normalized Hamming distance matrix
        self._tick_counter: int = 0
        self._current_index: int = -1
        self._current_bits: np.ndarray | None = None
        # Clustering state
        self.cluster_threshold: float = 0.10  # normalized Hamming distance threshold
        self._clusters: list[list[int]] = []
        self._cluster_colors: list[tuple[int, int, int]] = []  # BGR
        # Persistence directory (fixed relative to this file)
        self.persist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'state')
        try:
            os.makedirs(self.persist_dir, exist_ok=True)
        except Exception:
            pass
        # Logs directory and session file
        self.logs_dir = os.path.join(self.persist_dir, 'logs')
        try:
            os.makedirs(self.logs_dir, exist_ok=True)
        except Exception:
            pass
        self._log_buffer: list[dict] = []
        self._log_context: dict = {}
        self._log_file_path = os.path.join(
            self.logs_dir,
            f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        # Separate sleep log file
        self._sleep_log_file_path = os.path.join(
            self.logs_dir,
            f"sleep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        self._sleep_log_buffer: list[dict] = []
        # TPS estimation
        self._ema_tps: float = 0.0
        # Prediction state
        self._last_bits_frozen: np.ndarray | None = None
        self._last_bits_live: np.ndarray | None = None
        self._last_live_z: np.ndarray | None = None
        self._hash_predictor = _HashPredictor()
        self._latent_predictor = _LiveLatentPredictor(self.latent_dim)  # Still predicts only vision latents
        # PC/D pipes (PyTorch)
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pc_pipe = PCPipe(latent_dim=self.combined_latent_dim, device=self.torch_device).to(self.torch_device)  # Use combined latent for PC pipe
        self.d_pipe = DPipe(latent_dim=self.combined_latent_dim, hash_bits=self.lsh_bits, device=self.torch_device).to(self.torch_device)
        self.pc_opt = self.pc_pipe.optimizer(lr=1e-4)
        self.d_opt = self.d_pipe.optimizer(lr=1e-4)
        # PC window
        self.pc_window_len = 16
        self._pc_seq: deque = deque(maxlen=self.pc_window_len)
        self._last_live_pred_for_pc: np.ndarray | None = None
        # D-train buffers (one-step delay supervision)
        self._pending_train_sample = None  # dict with inputs at t, target set at t+1
        self._last_d_loss: float = 0.0
        self._last_d_hash_sim: float = 0.0
        # D rolling buffer
        self.d_seq_len = 8
        self._d_seq: deque = deque(maxlen=self.d_seq_len)
        # Live prediction accuracy tracking
        self._last_pred_live_bits_str: str | None = None
        self._acc_live_total: int = 0
        self._acc_live_correct: int = 0
        self._acc_live_sim_sum: float = 0.0
        # Diagnostics thresholds and history
        self.diag_thresholds = {
            'recon_loss': {'high': 0.1, 'low': 1e-4},
            'latent_mse': {'high': 0.05, 'low': 1e-4},
            'd_hash_sim': {'low': 0.7, 'high': 0.98},
            'predictor': {'exact_low_pct': 0.5, 'sim_high': 0.95},
            'uniq_window': 1000,
        }
        self._active_count_history: deque = deque(maxlen=64)  # store (tick, active_count) at 100-tick cadence

        # -------- Novelty state --------
        # Components:
        #   - latent_pred_err: error of last-tick latent prediction vs current live latent
        #   - hash_delta_live: normalized Hamming distance between consecutive live hashes
        #   - pred_hash_wrong: 1 - similarity(predicted_hash_live, actual_live_hash)  [optional]
        self._prev_live_bits_str: str | None = None
        self._last_pred_live_z: np.ndarray | None = None
        self._last_latent_pred_mse: float = 0.0
        self._nov_ema: float = 0.0
        self._nov_last_hash_delta: float = 0.0
        self.nov_params = {
            'w_latent': 0.6,        # weight for latent prediction error
            'w_hash_delta': 0.4,    # weight for tick-to-tick hash delta
            'w_pred_hash': 0.0,     # optional: weight for predicted-hash wrongness
            'ema_alpha': 0.2,       # EMA smoothing for novelty
            'mse_scale': 1.0        # scale for mapping MSE -> [0,1] via 1 - exp(-scale*mse)
        }

        # -------- Executive (IILSTM) & Worker Registry (dry-run) --------
        self._drivers_dim = 4     # [novelty, energy, sleep_pressure, reserve]
        self._control_dim = 32
        self._kmax = 2
        self._registry = WorkerRegistry()
        # Seed with a single predictor slot (handle wired later when executing actions)
        self._predictor_worker_idx = self._registry.register(name="predictor_v1", wtype="predictor", handle=None, control_dim=self._control_dim)
        self._mouse_worker_idx = self._registry.register(name="mouse_v1", wtype="mouse", handle=None, control_dim=self._control_dim)
        self._key_worker_idx   = self._registry.register(name="key_v1",   wtype="key",   handle=None, control_dim=self._control_dim)
        self._audio_worker_idx = self._registry.register(name="audio_v1", wtype="audio", handle=None, control_dim=self._control_dim)
        
        # Ensure the predictor exists before any sleep cloning/training
        try:
            # Use the same bit-width as the engine LSH to avoid silent shape drift
            self._ensure_predictor_worker(latent_dim=self.latent_dim, hash_bits=self.lsh_bits)
            # Ensure the registry handle stays bound and world predictor is discoverable
            if self._predictor_worker is not None:
                if self._predictor_worker_idx is None:
                    self._predictor_worker_idx = 0
                try:
                    self._registry._workers[self._predictor_worker_idx].handle = self._predictor_worker
                except Exception:
                    pass
                # Make discovery trivial for NREM world training
                if getattr(self, "world_predictor", None) is None:
                    self.world_predictor = self._predictor_worker
        except Exception:
            self._predictor_worker = None  # sleep will fall back to latent predictor
        
        self._iilstm = IILSTM(
            drivers_dim=self._drivers_dim,
            num_workers=self._registry.size(),
            control_dim=self._control_dim,
            hidden=128,
            kmax=self._kmax,
            device=torch.device('cpu')
        )
        self._action_bus = ActionBus()
        # Last routing snapshot (for logging)
        self._iilstm_last_k: int = 0
        self._iilstm_last_selected: list[dict] = []
        self._iilstm_last_workers: list[dict] = []
        self._iilstm_last_routing_logits: list[float] = []
        self._iilstm_last_k_logits: list[float] = []

        # -------- Basic drivers besides novelty (initial scaffolding) --------
        self._energy = 1.0
        self._sleep_pressure = 0.0
        self.energy_params = {
            'drain_base': 0.001,   # per-tick drain while awake; action-cost added later
            'sleep_low': 0.15,     # threshold for pressure accumulation
            'pressure_alpha': 0.05 # EMA for sleep pressure
        }

        # ---- Reward target for homeostasis (used in sleep training) ----
        self.reward_params = {
            "target": {"novelty": 0.5, "energy": 0.7, "sleep_pressure": 0.2},
            "weights": {"novelty": 1.0, "energy": 0.5, "sleep_pressure": 0.2}
        }

        # ---- Experience buffer and sleep manager ----
        self._exp = getattr(self, "_exp", ExperienceBuffer(maxlen=50000))
        self._last_rec = None
        self._sleep = SleepManager(nrem_steps=1, rem_steps=1)  # hooks decide actual work
        self._mode = "awake"
        self._exp.start_episode()

        # ---- Sleep switches (merged defaults live in VaeSup.ensure_sleep_defaults) ----
        # Do NOT overwrite here; let ensure_sleep_defaults() seed the dict and only
        # set missing keys. Then explicitly prefer training the world model in NREM.
        self.sleep_params = {"rem_steps": 6, "rem_batch": 4, "min_replay": 5}

        # ---- Sleep reentrancy guard ----
        self._sleep_active = False

        # ---- Sleep diagnostics container (cleared each cycle) ----
        self._sleep_diag: Dict[str, Any] = {}
        
        # internal scratch during one sleep
        self._sleep_log_ctx = {}

        # Support utilities for sleep/replay/worker consolidation
        self._sup = VaeSup(self)
        # Initialize energy/sleep defaults once (no silent swallow)
        self._sup.ensure_sleep_defaults()
        self._sup.world_epochs = 2  # tiny bump for stability

        # Ensure predictor worker is created so NREM can discover it
        try:
            self._ensure_predictor_worker(latent_dim=self.latent_dim, hash_bits=self.lsh_bits)
        except Exception:
            pass
            
        # Ensure world predictor is discoverable for NREM
        self.world_predictor = getattr(self, "world_predictor", None) or self._resolve_world_predictor()
        if self.world_predictor is None:
            try:
                self.diag_warning.emit("World predictor not found; NREM world training will be skipped until available")
            except Exception:
                pass
        else:
            try:
                self.diag_info.emit(f"World predictor discovered: {type(self.world_predictor).__name__}")
            except Exception:
                pass

        # Be explicit: enable world predictor NREM consolidation by default.
        # Users can toggle this off later in the UI or config.
        self.sleep_switches["update_world_nrem"] = True
        self.sleep_switches["update_workers_nrem"] = True
        self.sleep_switches["maintain_lsh_nrem"] = True

        # ---- Emergency stop (END key) ----
        self._emergency_last_trigger = 0.0
        self._emergency_cooldown_s = 1.0  # don't spam toggles

        # -------- Worker execution controls (kill-switches) --------
        self.worker_exec_enabled = True            # flip to True to actually run workers
        self.worker_use_predictor_output = True    # if True, use worker's latent as next-pred buffer
        # Foundational workers
        self.enable_mouse_worker = True
        self.enable_key_worker = True
        self.enable_audio_worker = True
        # KEEP the indices set by registry.register(...) above
        # self._mouse_worker_idx is assigned when registering 'mouse_v1'
        # self._key_worker_idx   is assigned when registering 'key_v1'
        self._mouse_worker = None
        self._key_worker = None
        self._audio_worker = None
        self.action_params = {
            "mouse_enabled": True,
            "clicks_enabled": True,          # enable for worker imitation training
            "mouse_max_px": 5.0,
            "click_cooldown_ticks": 20,
            "keys_enabled": True,
            "key_cooldown_ticks": 10,
            "audio_enabled": True,
            "audio_cooldown_ticks": 4,
            "energy_min": 0.02,
        }
        
    def _on_audio_chunk(self, audio_chunk: np.ndarray):
        """Process incoming audio chunk through frozen audio encoder."""
        try:
            # Process through frozen audio encoder
            audio_latent = self.frozen_audio_encoder.encode(audio_chunk)
            self._latest_audio_latent = audio_latent.astype(np.float32)
        except Exception as e:
            print(f"Audio processing error: {e}")
            self._latest_audio_latent = np.zeros(self.audio_latent_dim)

    def start(self):
        if self.running:
            return
        # Attempt to load checkpoints before starting
        try:
            lsh_path = os.path.join(self.persist_dir, 'lsh_state.npz')
            if os.path.exists(lsh_path):
                self.lsh.load_state(lsh_path)
        except Exception:
            pass
        try:
            enc_path = os.path.join(self.persist_dir, 'frozen_encoder.npz')
            if os.path.exists(enc_path):
                data = np.load(enc_path)
                # Try to load new format first (vision + audio)
                if 'vision_encoder_weight' in data and 'audio_encoder_weight' in data:
                    try:
                        self.frozen_vision_encoder.encoder_weight = np.array(data['vision_encoder_weight'], dtype=np.float32)
                        self.frozen_vision_encoder.encoder_bias = np.array(data['vision_encoder_bias'], dtype=np.float32)
                        self.frozen_audio_encoder.encoder_weight = np.array(data['audio_encoder_weight'], dtype=np.float32)
                        self.frozen_audio_encoder.encoder_bias = np.array(data['audio_encoder_bias'], dtype=np.float32)
                    except Exception:
                        # Fallback: re-init if corrupted
                        rng_v = np.random.default_rng(42)
                        rng_a = np.random.default_rng(84)
                        self.frozen_vision_encoder.encoder_weight = (rng_v.standard_normal(self.frozen_vision_encoder.encoder_weight.shape) * 0.01).astype(np.float32)
                        self.frozen_vision_encoder.encoder_bias = np.zeros_like(self.frozen_vision_encoder.encoder_bias, dtype=np.float32)
                        self.frozen_audio_encoder.encoder_weight = (rng_a.standard_normal(self.frozen_audio_encoder.encoder_weight.shape) * 0.01).astype(np.float32)
                        self.frozen_audio_encoder.encoder_bias = np.zeros_like(self.frozen_audio_encoder.encoder_bias, dtype=np.float32)
                # Legacy format compatibility (vision only)
                elif 'encoder_weight' in data and 'encoder_bias' in data:
                    try:
                        self.frozen_vision_encoder.encoder_weight = np.array(data['encoder_weight'], dtype=np.float32)
                        self.frozen_vision_encoder.encoder_bias = np.array(data['encoder_bias'], dtype=np.float32)
                        # Initialize audio encoder with defaults
                        rng_a = np.random.default_rng(84)
                        self.frozen_audio_encoder.encoder_weight = (rng_a.standard_normal(self.frozen_audio_encoder.encoder_weight.shape) * 0.01).astype(np.float32)
                        self.frozen_audio_encoder.encoder_bias = np.zeros_like(self.frozen_audio_encoder.encoder_bias, dtype=np.float32)
                    except Exception:
                        pass
        except Exception:
            pass
        # Load predictors
        try:
            hp_path = os.path.join(self.persist_dir, 'hash_predictor.json')
            if os.path.exists(hp_path):
                self._hash_predictor.load(hp_path)
        except Exception:
            pass
        try:
            lp_path = os.path.join(self.persist_dir, 'live_latent_predictor.npz')
            if os.path.exists(lp_path):
                self._latent_predictor.load(lp_path)
        except Exception:
            pass
        # Load PC/D pipes
        try:
            pc_path = os.path.join(self.persist_dir, 'pc_lstm.pt')
            if os.path.exists(pc_path):
                self.pc_pipe.load(pc_path, map_location=self.torch_device)
        except Exception:
            pass
        try:
            dp_path = os.path.join(self.persist_dir, 'dpipe_lstm.pt')
            if os.path.exists(dp_path):
                self.d_pipe.load(dp_path, map_location=self.torch_device)
        except Exception:
            pass
        try:
            live_path = os.path.join(self.persist_dir, 'live_vae.npz')
            if os.path.exists(live_path):
                data = np.load(live_path)
                # Ensure proper data types when loading
                self.live_vae.encoder_weight = np.array(data['encoder_weight'], dtype=np.float32)
                self.live_vae.encoder_bias = np.array(data['encoder_bias'], dtype=np.float32)
                self.live_vae.decoder_weight = np.array(data['decoder_weight'], dtype=np.float32)
                self.live_vae.decoder_bias = np.array(data['decoder_bias'], dtype=np.float32)
        except Exception:
            # If loading fails, ensure we still have proper float32 types
            pass
        # Load hash database (codes and optional coords)
        try:
            hash_db_path = os.path.join(self.persist_dir, 'hash_db.npz')
            if os.path.exists(hash_db_path):
                data = np.load(hash_db_path, allow_pickle=True)
                if 'codes_bits' in data.files:
                    codes_bits_arr = data['codes_bits']  # shape (n, k)
                    if codes_bits_arr.ndim == 2 and codes_bits_arr.shape[1] == self.lsh.num_bits:
                        self._codes_bits = [row.astype(np.uint8).copy() for row in codes_bits_arr]
                        self._codes_seen = set()
                        self._code_to_index = {}
                        for i, row in enumerate(self._codes_bits):
                            s = self.lsh.bits_to_str(row)
                            self._codes_seen.add(s)
                            self._code_to_index[s] = i
                        # load coords if present, otherwise recompute later
                        if 'coords_3d' in data.files:
                            coords = data['coords_3d']
                            if coords.ndim == 2 and coords.shape[0] == len(self._codes_bits) and coords.shape[1] == 3:
                                self._coords_3d = coords.astype(np.float32)
                        # Recompute embedding & clusters to ensure consistency
                        self._recompute_embedding_if_needed()
                        # Reset current pointer
                        self._current_index = -1
                        self._current_bits = None
                        # Emit an initial image so GUI reflects restored map
                        try:
                            img = self._render_hashmap_image(640, 480)
                            if img is not None:
                                self.hashmap_frame.emit(img)
                            stats = self._format_hash_stats()
                            self.hashmap_stats.emit(stats)
                        except Exception:
                            pass
        except Exception:
            pass
        
        # Final safety check: ensure all weights are still float32 after loading
        self.frozen_vision_encoder.encoder_weight = self.frozen_vision_encoder.encoder_weight.astype(np.float32)
        self.frozen_vision_encoder.encoder_bias = self.frozen_vision_encoder.encoder_bias.astype(np.float32)
        self.frozen_audio_encoder.encoder_weight = self.frozen_audio_encoder.encoder_weight.astype(np.float32)  
        self.frozen_audio_encoder.encoder_bias = self.frozen_audio_encoder.encoder_bias.astype(np.float32)
        self.live_vae.encoder_weight = self.live_vae.encoder_weight.astype(np.float32)
        self.live_vae.encoder_bias = self.live_vae.encoder_bias.astype(np.float32)
        self.live_vae.decoder_weight = self.live_vae.decoder_weight.astype(np.float32)
        self.live_vae.decoder_bias = self.live_vae.decoder_bias.astype(np.float32)
        
        # Start audio engine
        self.audio_engine.start()
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        # Stop audio engine
        self.audio_engine.stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        # Save checkpoints
        try:
            lsh_path = os.path.join(self.persist_dir, 'lsh_state.npz')
            self.lsh.save_state(lsh_path)
        except Exception:
            pass
        try:
            enc_path = os.path.join(self.persist_dir, 'frozen_encoder.npz')
            np.savez(enc_path,
                     vision_encoder_weight=self.frozen_vision_encoder.encoder_weight,
                     vision_encoder_bias=self.frozen_vision_encoder.encoder_bias,
                     audio_encoder_weight=self.frozen_audio_encoder.encoder_weight,
                     audio_encoder_bias=self.frozen_audio_encoder.encoder_bias)
        except Exception:
            pass
        # Save predictors
        try:
            hp_path = os.path.join(self.persist_dir, 'hash_predictor.json')
            self._hash_predictor.save(hp_path)
        except Exception:
            pass
        try:
            lp_path = os.path.join(self.persist_dir, 'live_latent_predictor.npz')
            self._latent_predictor.save(lp_path)
        except Exception:
            pass
        try:
            pc_path = os.path.join(self.persist_dir, 'pc_lstm.pt')
            self.pc_pipe.save(pc_path)
        except Exception:
            pass
        try:
            dp_path = os.path.join(self.persist_dir, 'dpipe_lstm.pt')
            self.d_pipe.save(dp_path)
        except Exception:
            pass
        try:
            live_path = os.path.join(self.persist_dir, 'live_vae.npz')
            np.savez(live_path,
                     encoder_weight=self.live_vae.encoder_weight,
                     encoder_bias=self.live_vae.encoder_bias,
                     decoder_weight=self.live_vae.decoder_weight,
                     decoder_bias=self.live_vae.decoder_bias)
        except Exception:
            pass
        # Save hash database (codes and coords)
        try:
            hash_db_path = os.path.join(self.persist_dir, 'hash_db.npz')
            if len(self._codes_bits) > 0:
                codes_bits_arr = np.stack(self._codes_bits, axis=0).astype(np.uint8)
            else:
                codes_bits_arr = np.zeros((0, self.lsh.num_bits), dtype=np.uint8)
            coords_save = self._coords_3d if self._coords_3d is not None else np.zeros((0, 3), dtype=np.float32)
            np.savez(hash_db_path, codes_bits=codes_bits_arr, coords_3d=coords_save)
        except Exception:
            pass

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        # frame is BGRA from mss; convert to RGB and downsample
        # Safeguard shape
        if frame.ndim == 3 and frame.shape[2] >= 3:
            rgb = frame[:, :, :3][:, :, ::-1]  # BGRA/BGR -> RGB
        else:
            # If unexpected shape, flatten to zeros of expected size
            rgb = np.zeros((self.downsample_height, self.downsample_width, 3), dtype=np.uint8)

        # Simple nearest-neighbor downsampling via slicing
        H, W, _ = rgb.shape
        step_y = max(H // self.downsample_height, 1)
        step_x = max(W // self.downsample_width, 1)
        small = rgb[::step_y, ::step_x, :]
        small = small[:self.downsample_height, :self.downsample_width, :]
        x = small.astype(np.float32) / 255.0
        x = x.reshape(-1)
        # If shape mismatch due to unexpected screen size ratios, pad or crop
        input_dim = 3 * self.downsample_width * self.downsample_height
        if x.shape[0] < input_dim:
            x = np.pad(x, (0, input_dim - x.shape[0]))
        elif x.shape[0] > input_dim:
            x = x[:input_dim]
        return x

    def _resolve_world_predictor(self) -> Any:
        # 1) explicit attribute already set
        obj = getattr(self, "world_predictor", None)
        if obj is not None:
            return obj

        # 2) common fields on engine
        for name in ("predictor_worker", "predictor", "predictor_v1", "_predictor_worker"):
            w = getattr(self, name, None)
            if w is None: 
                continue
            m = getattr(w, "as_world_module", None)
            if callable(m):
                mod = m()
                if mod is not None:
                    return mod
            for attr in ("model", "net", "module", "core"):
                mod = getattr(w, attr, None)
                if mod is not None:
                    return mod
            if hasattr(w, "forward") or callable(w):
                return w

        # 3) scan registered workers  
        registry = getattr(self, "_registry", None)
        if registry is not None:
            try:
                workers_list = registry.all()
            except Exception:
                workers_list = []
                
            for worker_entry in workers_list:
                wtype = getattr(worker_entry, "wtype", None)
                w = getattr(worker_entry, "handle", None)
                
                if wtype == "predictor" or (isinstance(wtype, str) and "predictor" in wtype.lower()):
                    if w is None:
                        continue
                    m = getattr(w, "as_world_module", None)
                    if callable(m):
                        mod = m()
                        if mod is not None:
                            return mod
                    for attr in ("model", "net", "module", "core"):
                        mod = getattr(w, attr, None)
                        if mod is not None:
                            return mod
                    if hasattr(w, "forward") or callable(w):
                        return w

        return None

    def _normalize_world_result(self, out):
        # Accept: bool | (bool,) | (bool, dict) | (bool, dict, *ignored) | dict{"promoted": bool, ...}
        try:
            if isinstance(out, dict):
                return bool(out.get("promoted", False)), out
            if isinstance(out, tuple):
                if len(out) == 0:
                    return False, {}
                if len(out) == 1:
                    return bool(out[0]), {}
                # len >= 2 → take first two, ignore the rest
                return bool(out[0]), (out[1] or {})
            # fallback: scalar truthiness
            return bool(out), {}
        except Exception as e:
            return False, {"error": f"normalize_error: {e!s}"}

    def _format_features(self, vector: np.ndarray, loss: float = None) -> str:
        # Full latent vector output
        vec_text = ", ".join(f"{v:.6f}" for v in vector)
        if loss is not None:
            return f"z=[{vec_text}]  |  loss={loss:.6f}"
        return f"z=[{vec_text}]"

    def _mse_to_unit(self, mse: float) -> float:
        # Map MSE to [0,1] with a saturating curve: 1 - exp(-scale*mse)
        scale = float(self.nov_params.get('mse_scale', 1.0))
        try:
            val = 1.0 - float(np.exp(-scale * max(0.0, float(mse))))
            return max(0.0, min(1.0, val))
        except Exception:
            return 0.0

    def _driver_vector(self) -> np.ndarray:
        # Update energy and sleep pressure in awake mode. Action-cost comes later.
        self._energy = max(0.0, min(1.0, self._energy - float(self.energy_params['drain_base'])))
        self.energy_value.emit(float(self._energy))
        low_th = float(self.energy_params['sleep_low'])
        p_alpha = float(self.energy_params['pressure_alpha'])
        if self._energy < low_th:
            self._sleep_pressure = (1 - p_alpha) * self._sleep_pressure + p_alpha * 1.0
        else:
            self._sleep_pressure = (1 - p_alpha) * self._sleep_pressure
        try:
            self.sleep_pressure_value.emit(float(self._sleep_pressure))
        except Exception:
            pass
        # Assemble drivers: [novelty, energy, sleep_pressure, reserve(0)]
        v = np.array([float(self._nov_ema), float(self._energy), float(self._sleep_pressure), 0.0], dtype=np.float32)
        return v

    def _log_sleep_metric(self, **kv):
        payload = {
            "type": "sleep_metric",
            "time": datetime.utcnow().isoformat() + "Z",
            "tick": int(getattr(self, "_tick_counter", 0))
        }
        payload.update(kv)
        
        # Add to sleep log buffer for separate file
        self._sleep_log_buffer.append(payload.copy())
        
        # Also emit to GUI for immediate visibility
        line = "SLEEP " + json.dumps(payload, separators=(",", ":"))
        try:
            self.diag_info.emit(line)
        except Exception:
            try:
                print(line)  # fallback to stdout
            except Exception:
                pass
        
        # Flush sleep log periodically (every 10 entries or immediately on important events)
        important_events = ['enter', 'complete', 'skip', 'recharge']
        if len(self._sleep_log_buffer) >= 10 or kv.get('event') in important_events:
            self._flush_sleep_log_buffer()

    def _emergency_stop(self, source: str = "hotkey"):
        now = time.time()
        if now - getattr(self, "_emergency_last_trigger", 0.0) < self._emergency_cooldown_s:
            return
        self._emergency_last_trigger = now
        # Flip all execution flags off
        self.worker_exec_enabled = False
        self.enable_mouse_worker = False
        self.enable_key_worker = False
        self.enable_audio_worker = False
        self.action_params["clicks_enabled"] = False
        self.action_params["keys_enabled"] = False
        self.action_params["audio_enabled"] = False
        try:
            self.diag_warning.emit(f"EMERGENCY STOP ({source}): mouse/keyboard disabled")
        except Exception:
            pass

    def _check_emergency_hotkey(self):
        """
        If the physical END key is pressed, disable workers immediately.
        Priority: 'keyboard' lib if present; else Windows GetAsyncKeyState; else no-op.
        """
        # Try optional 'keyboard' module (if installed)
        try:
            import keyboard  # type: ignore
            if keyboard.is_pressed("end"):
                self._emergency_stop("keyboard.is_pressed")
                return
        except Exception:
            pass
        
        # Windows fallback via WinAPI
        try:
            if sys.platform.startswith("win"):
                import ctypes  # lazy import
                VK_END = 0x23
                # 0x8000 bit set means key is currently down
                if ctypes.windll.user32.GetAsyncKeyState(VK_END) & 0x8000:
                    self._emergency_stop("GetAsyncKeyState")
        except Exception:
            pass

    def _reward_from_next(self, prev_rec: dict, curr_rec: dict) -> float:
        t = self.reward_params["target"]; w = self.reward_params["weights"]
        d_next = (curr_rec.get("drivers") or {})
        err = 0.0
        for k in ("novelty", "energy", "sleep_pressure"):
            if k in d_next:
                diff = float(d_next[k]) - float(t.get(k, 0.0))
                err += float(w.get(k, 1.0)) * (diff * diff)
        return -err

    def _nrem_hook(self, replay):
        # Essential logging only

        res = {"world": None, "workers": None}

        # World predictor NREM (enable by default unless explicitly disabled)
        if self.sleep_switches.get("update_world_nrem", True):
            try:
                out = self._sup.nrem_update_world(replay)
            except Exception as e:
                world_summary = {"promoted": False, "error": f"nrem_update_world_exception: {e!s}"}
            else:
                promoted, metrics = (out if isinstance(out, tuple) else (bool(out), {}))
                
                # force sanity: no promotion without steps>0 and finite metrics
                if promoted and (int(metrics.get("steps", 0)) <= 0
                                 or not self._sup._is_finite(metrics.get("world_mse_baseline"))
                                 or not self._sup._is_finite(metrics.get("world_mse_val"))):
                    promoted = False

                world_summary = {"promoted": bool(promoted)}
                # merge detailed metrics if present
                for k in ("world_mse_baseline","world_mse_val","pairs_train","pairs_val","steps","accept_rel","accept_abs","trainable","error"):
                    if isinstance(metrics, dict) and k in metrics:
                        world_summary[k] = metrics[k]
                        
            # log essential world metrics only
            essential_world = {
                "world_mse_baseline": world_summary.get("world_mse_baseline"),
                "world_mse_val": world_summary.get("world_mse_val"), 
                "steps": world_summary.get("steps")
            }
            self._log_sleep_metric(event="nrem_world", **essential_world)
        else:
            essential_world = {"world_mse_baseline": None, "world_mse_val": None, "steps": 0}
            self._log_sleep_metric(event="nrem_world", **essential_world)

        # summary
        sd = getattr(self, "_sleep_diag", {}) or {}
        sd.setdefault("nrem", {}).setdefault("world", {}).update({"promoted": bool(world_summary.get("promoted", False)), **{k: world_summary[k] for k in world_summary if k != "promoted"}})
        # only set cloned.predictor when we actually trained
        if int(world_summary.get("steps", 0)) > 0:
            sd.setdefault("cloned", {})["predictor"] = True
        self._sleep_diag = sd

        # Workers imitation (opportunistic - only run if dataset has samples)
        if self.sleep_switches.get("update_workers_nrem", False):
            try:
                ds_info = self._sup.peek_worker_dataset_info()
                if ds_info.get("total", 0) == 0:
                    essential_workers = {"samples": 0}
                    self._log_sleep_metric(event="nrem_workers", **essential_workers)
                else:
                    if 'train' not in locals():
                        train, val = self._sup.build_replay(replay)
                        cand = self._sup.clone_components_for_sleep()
                    wr = self._sup.train_workers(train, val, cand)
                    essential_workers = {"samples": ds_info.get("total", 0)}
                    self._log_sleep_metric(event="nrem_workers", **essential_workers)
            except Exception as e:
                essential_workers = {"samples": 0}
                self._log_sleep_metric(event="nrem_workers", **essential_workers)
        else:
            essential_workers = {"samples": 0}
            self._log_sleep_metric(event="nrem_workers", **essential_workers)

        sd = getattr(self, "_sleep_diag", {}) or {}
        sd.setdefault("nrem", {})
        sd["nrem"]["workers"] = workers_summary
        self._sleep_diag = sd

        return sd.get("nrem", {})

    def _rem_hook(self, replay: list[dict]) -> None:
        # Safe REM: sandbox rollouts only; no promotion here.
        steps = int(self.sleep_params.get("rem_steps", 6))
        batch = int(self.sleep_params.get("rem_batch", 4))
        dreams = self._sup.sample_dream_rollouts(replay, steps=steps, batch=batch)
        rem_world = self._sup.train_world_rem(dreams)
        rem_workers = self._sup.train_workers_rem(dreams)
        rem_router = self._sup.train_router_rem(dreams)
        self._sleep_diag["rem"] = {
            "dreams_stats": dreams.get("stats", {}),
            "world": rem_world,
            "workers": rem_workers,
            "router": rem_router
        }
        # Log essential REM router metrics if training is on
        router_metrics = rem_router or {}
        if router_metrics.get("steps", 0) > 0:
            essential_rem = {
                "route_learn_steps": router_metrics.get("steps"),
                "route_eval_return": router_metrics.get("eval_return"), 
                "route_kl": router_metrics.get("kl")
            }
            self._log_sleep_metric(event="rem_router", **essential_rem)

    def _recharge(self):
        prev_energy = float(getattr(self, "_energy", 0.0))
        prev_pressure = float(getattr(self, "_sleep_pressure", 0.0))
        
        # reset state
        self._energy = 1.0
        try:
            if hasattr(self, "_sup") and hasattr(self._sup, "set_sleep_pressure"):
                self._sup.set_sleep_pressure(0.0)  # hard reset pressure
        except Exception:
            pass
        
        # emit to GUI first so UI matches logs
        try:
            self.energy_value.emit(float(self._energy))
            self.sleep_pressure_value.emit(float(getattr(self, "_sleep_pressure", 0.0)))
        except Exception:
            pass
        
        # now log the post-reset values
        try:
            self._log_sleep_metric(
                event="recharge",
                energy_before=prev_energy,
                energy_after=float(self._energy),
                pressure_after=float(getattr(self, "_sleep_pressure", 0.0))
            )
            self.diag_info.emit("Sleep recharge complete")
        except Exception:
            pass

    # ---------------- Sleep helpers: replay builder and cloning ----------------

    def _ensure_predictor_worker(self, latent_dim: int, hash_bits: int | None = None):
        if hash_bits is None:
            hash_bits = getattr(self, "lsh_bits", 64)
        if self._predictor_worker is not None:
            # Already constructed; make sure it's bound to the registry
            try:
                if self._predictor_worker_idx is None:
                    self._predictor_worker_idx = 0
                self._registry._workers[self._predictor_worker_idx].handle = self._predictor_worker
            except Exception:
                pass
            return
        try:
            self._predictor_worker = PredictorWorker(
                latent_dim=latent_dim,
                hash_bits=int(hash_bits),
                packet_dim=64,
                hidden=256,
                max_layers=3,
                device=torch.device('cpu')
            )
            if self._predictor_worker_idx is None:
                self._predictor_worker_idx = 0
            try:
                self._registry._workers[self._predictor_worker_idx].handle = self._predictor_worker
            except Exception:
                pass
            try:
                self.diag_info.emit(f"Predictor worker ready (latent_dim={latent_dim}, hash_bits={hash_bits})")
            except Exception:
                pass
        except Exception as e:
            self._predictor_worker = None
            try:
                self.diag_warning.emit(f"Predictor init failed: {e}")
            except Exception:
                pass

    def _ensure_mouse_worker(self):
        if self._mouse_worker is None:
            self._mouse_worker = MouseWorker(control_dim=self._control_dim, hidden=256, max_layers=2, device=torch.device('cpu'))
            self._registry._workers[self._mouse_worker_idx].handle = self._mouse_worker

    def _ensure_key_worker(self, key_count: int = 8):
        if self._key_worker is None:
            self._key_worker = KeyWorker(key_count=key_count, control_dim=self._control_dim, hidden=256, max_layers=2, device=torch.device('cpu'))
            self._registry._workers[self._key_worker_idx].handle = self._key_worker
    
    def _ensure_audio_worker(self):
        if self._audio_worker is None:
            from workers.audio_worker import AudioWorkerConfig
            cfg = AudioWorkerConfig(control_dim=self._control_dim, hidden_dim=64, vocab_size=64)
            self._audio_worker = AudioWorker(name="audio_v1", cfg=cfg, device=torch.device('cpu'))
            self._registry._workers[self._audio_worker_idx].handle = self._audio_worker

    def _run(self):
        sct = mss.mss()
        monitor = sct.monitors[1]
        while self.running:
            tick_start = time.time()
            # Emergency STOP: physical END key
            try:
                self._check_emergency_hotkey()
            except Exception:
                pass
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)

            x = self._prepare_frame(frame)

            # Frozen features (encoder-only, no training) - Vision + Audio combined
            frozen_vision_z = self.frozen_vision_encoder.encode(x)
            # Combine vision and audio latents
            frozen_combined_z = np.concatenate([frozen_vision_z, self._latest_audio_latent])
            frozen_text = self._format_features(frozen_combined_z)
            self.frozen_features.emit(frozen_text)
            self.frozen_latent_np.emit(frozen_combined_z.astype(np.float32))

            # LSH from combined frozen latent with pruning/bucketing
            code, action, centroid = self.lsh.prune_or_bucket(frozen_combined_z)
            bits = np.frombuffer(code.encode('ascii'), dtype=np.uint8)
            # Convert ASCII '0'/'1' to 0/1
            bits = (bits == ord('1')).astype(np.uint8)
            chained = self.lsh.compute_chained_hash(bits)
            bits_str = self.lsh.bits_to_str(bits)
            self.lsh_code_bits.emit(bits_str)
            self.lsh_chained_hash.emit(chained)

            # Update 3D hash map occasionally
            self._update_hash_collection(bits)
            if (self._tick_counter % 5) == 0:
                self._recompute_embedding_if_needed()
                img = self._render_hashmap_image(640, 480)
                if img is not None:
                    self.hashmap_frame.emit(img)
                # Emit stats
                stats = self._format_hash_stats()
                self.hashmap_stats.emit(stats)

            # Emit current position for heatmap window every tick (if available)
            try:
                self._emit_heatmap_hit()
            except Exception:
                pass

            # Live features (with training step)
            # Consistency against frozen vision latent only (beta small)
            live_z, recon_loss, total_loss = self.live_vae.train_step(x, frozen_z=frozen_vision_z.astype(np.float32), beta=0.1)
            live_text = self._format_features(live_z, loss=recon_loss)
            self.live_features.emit(live_text)
            # Compute live hash bits (for prediction/tracking only) - combine with audio
            live_combined_z = np.concatenate([live_z, self._latest_audio_latent])
            bits_live = self.lsh.compute_code(live_combined_z)
            bits_live_str = self.lsh.bits_to_str(bits_live)

            # Evaluate last live prediction (pred for t vs actual t)
            try:
                if self._last_pred_live_bits_str is not None:
                    self._acc_live_total += 1
                    if self._last_pred_live_bits_str == bits_live_str:
                        self._acc_live_correct += 1
                    # similarity as 1 - normalized Hamming distance
                    sim = 1.0 - self._normalized_hamming_between_bitstrings(self._last_pred_live_bits_str, bits_live_str)
                    self._acc_live_sim_sum += float(sim)
                    exact_pct = (100.0 * self._acc_live_correct / max(1, self._acc_live_total))
                    sim_pct = (100.0 * self._acc_live_sim_sum / max(1, self._acc_live_total))
                    self.prediction_accuracy_text.emit(f"Pred acc (live): exact {exact_pct:.1f}% | sim {sim_pct:.1f}%")
            except Exception:
                pass

            # -------- Novelty computation (before predicting the next z) --------
            try:
                # 1) Latent prediction error (compare last predicted z to current live_z)
                if self._last_pred_live_z is not None:
                    dz = (self._last_pred_live_z.astype(np.float32) - live_z.astype(np.float32))
                    self._last_latent_pred_mse = float(np.mean(dz * dz))
                else:
                    self._last_latent_pred_mse = 0.0
                novelty_latent = self._mse_to_unit(self._last_latent_pred_mse)

                # 2) Tick-to-tick hash delta on LIVE bits
                if self._prev_live_bits_str is not None:
                    hash_delta_live = self._normalized_hamming_between_bitstrings(self._prev_live_bits_str, bits_live_str)
                else:
                    hash_delta_live = 0.0
                self._nov_last_hash_delta = float(hash_delta_live)

                # 3) Wrongness of last predicted live hash vs actual (1 - similarity)
                #    Reuse "sim" above if available; otherwise recompute safely.
                try:
                    pred_hash_wrong = 1.0 - float(sim)
                except Exception:
                    try:
                        if self._last_pred_live_bits_str is not None:
                            pred_hash_wrong = self._normalized_hamming_between_bitstrings(self._last_pred_live_bits_str, bits_live_str)
                        else:
                            pred_hash_wrong = 0.0
                    except Exception:
                        pred_hash_wrong = 0.0

                # Weighted aggregate with EMA smoothing
                wL = float(self.nov_params.get('w_latent', 0.6))
                wH = float(self.nov_params.get('w_hash_delta', 0.4))
                wP = float(self.nov_params.get('w_pred_hash', 0.0))
                raw_novelty = wL * novelty_latent + wH * float(hash_delta_live) + wP * float(pred_hash_wrong)
                raw_novelty = max(0.0, min(1.0, raw_novelty))
                alpha = float(self.nov_params.get('ema_alpha', 0.2))
                self._nov_ema = alpha * raw_novelty + (1.0 - alpha) * self._nov_ema

                # Emit signals
                self.novelty_value.emit(float(self._nov_ema))
                comp_text = (f"novelty= {self._nov_ema:.3f} "
                             f"(latent={novelty_latent:.3f}, hashΔ={hash_delta_live:.3f}, predHashWrong={pred_hash_wrong:.3f}; "
                             f"mse={self._last_latent_pred_mse:.6f})")
                self.novelty_components_text.emit(comp_text)
            except Exception as _:
                pass

            # Keep previous live bits string for next-tick delta
            self._prev_live_bits_str = bits_live_str

            # TPS pacing
            elapsed = time.time() - tick_start
            delay = max(0.0, 1.0 / self.tps - elapsed)
            if delay > 0:
                time.sleep(delay)

            self._tick_counter += 1
            # Update TPS (EMA over instantaneous TPS)
            tick_elapsed = elapsed + delay
            inst_tps = 1.0 / max(1e-6, tick_elapsed)
            alpha = 0.2
            self._ema_tps = alpha * inst_tps + (1.0 - alpha) * self._ema_tps
            self.tick_update.emit(self._tick_counter)
            self.tps_update.emit(self._ema_tps)

            # Per-tick logging moved below after action aggregation

            # ---------------------- Prediction updates ----------------------
            # Update hash predictor with transitions
            try:
                if self._last_bits_frozen is not None:
                    self._hash_predictor.update(self.lsh.bits_to_str(self._last_bits_frozen), bits_str)
                if self._last_bits_live is not None:
                    self._hash_predictor.update(self.lsh.bits_to_str(self._last_bits_live), bits_live_str)
            except Exception:
                pass
            self._last_bits_frozen = bits.copy()
            self._last_bits_live = bits_live.copy()

            # Predict next hashes from current
            try:
                pred_frozen = self._hash_predictor.predict(bits_str)
                if pred_frozen is not None:
                    self.predicted_hash_frozen.emit(pred_frozen)
            except Exception:
                pass
            try:
                pred_live = self._hash_predictor.predict(bits_live_str)
                if pred_live is not None:
                    self.predicted_hash_live.emit(pred_live)
                    # store for next tick evaluation
                    self._last_pred_live_bits_str = pred_live
            except Exception:
                pass

            # Live latent predictor: update with (prev_z -> current_z) then predict next
            try:
                if self._last_live_z is not None:
                    self._latent_predictor.update(self._last_live_z.astype(np.float32), live_z.astype(np.float32))
                self._last_live_z = live_z.copy()
                z_next_pred = self._latent_predictor.predict(live_z.astype(np.float32))
                # Decode predicted z to frame using live decoder
                pred_frame_flat = self.live_vae.decode(z_next_pred)
                # reshape back to small RGB frame
                w, h = self.downsample_width, self.downsample_height
                img = (np.clip(pred_frame_flat, 0.0, 1.0) * 255.0).astype(np.uint8).reshape(h * w, 3)
                img = img.reshape(h, w, 3)
                # scale up for visibility
                img_large = cv2.resize(img, (640, 480), interpolation=cv2.INTER_NEAREST)
                self.live_prediction_frame.emit(img_large[:, :, ::-1])  # emit RGB
            except Exception:
                pass

            # ---------------------- PC/D pipeline ----------------------
            try:
                # Build current PC sequence step: [frozen_combined_z, live_combined_z, live_pred_prev]
                if self._last_live_pred_for_pc is None:
                    self._last_live_pred_for_pc = np.zeros_like(live_combined_z, dtype=np.float32)
                pc_step_np = np.concatenate([frozen_combined_z.astype(np.float32), live_combined_z.astype(np.float32), self._last_live_pred_for_pc.astype(np.float32)], axis=0)
                self._pc_seq.append(pc_step_np)

                # Train D-pipe on previous sample if available (target is current live_z)
                if self._pending_train_sample is not None and len(self._pc_seq) >= self.pc_window_len and len(self._d_seq) >= self.d_seq_len:
                    self.d_opt.zero_grad()
                    self.pc_opt.zero_grad()
                    # Reset states for sequence evaluation
                    self.pc_pipe.reset_state()
                    self.d_pipe.reset_state()
                    seq_prev = self._pending_train_sample['pc_seq_tensor']  # (L,1,3*latent)
                    d_seq_prev = self._pending_train_sample['d_seq_tensor']  # (Ld,1,64+latent+k)
                    frozen_lat_prev = self._pending_train_sample['frozen_lat']  # (latent,)
                    target_lat = torch.tensor(live_combined_z.astype(np.float32), device=self.torch_device)

                    # Forward through PC->packet and D-pipe
                    pkt = self.pc_pipe(seq_prev)  # (64,)
                    # Use the stored D sequence for prediction
                    pred_lat = self.d_pipe(d_seq_prev)
                    loss_d = self.d_pipe.loss(pred_lat, target_lat, frozen_lat_prev, reg_w=0.10)
                    loss_d.backward()
                    # Gradient clipping
                    try:
                        nn.utils.clip_grad_norm_(self.d_pipe.parameters(), max_norm=1.0)
                        nn.utils.clip_grad_norm_(list(self.pc_pipe.c_lstm.parameters()) + list(self.pc_pipe.proj.parameters()), max_norm=1.0)
                    except Exception:
                        pass
                    self.d_opt.step()
                    self.pc_opt.step()
                    self._last_d_loss = float(loss_d.detach().cpu().item())

                    # Hash similarity between predicted and actual next (via LSH on latents)
                    pred_lat_np = pred_lat.detach().cpu().numpy()
                    pred_bits = self.lsh.compute_code(pred_lat_np)
                    actual_bits = self.lsh.compute_code(live_combined_z)
                    sim = 1.0 - (np.sum(pred_bits != actual_bits) / float(pred_bits.shape[0]))
                    self._last_d_hash_sim = float(sim)
                    self.dpipe_accuracy_text.emit(f"D acc: latent MSE {self._last_d_loss:.4f} | hash sim {self._last_d_hash_sim:.2f}")

                # Build inputs for new D prediction based on current state
                # Packet from current PC window
                if len(self._pc_seq) > 0:
                    seq_np = np.stack(self._pc_seq, axis=0)  # (L, 3*latent)
                    seq_t = torch.tensor(seq_np, dtype=torch.float32, device=self.torch_device).view(-1, 1, 3 * self.combined_latent_dim)
                    # Reset PC states before using full sequence window
                    self.pc_pipe.reset_state()
                    pkt_now = self.pc_pipe(seq_t)  # (64,)
                    # -------- IILSTM dry-run routing --------
                    try:
                        packet_np = pkt_now.detach().cpu().numpy().astype(np.float32)
                        drivers_np = self._driver_vector()
                        packet_t = torch.from_numpy(packet_np)
                        drivers_t = torch.from_numpy(drivers_np)
                        routing_logits, k_logits, control_tokens = self._iilstm(packet_t, drivers_t)
                        k = int(torch.argmax(k_logits).item())
                        k = max(0, min(k, self._kmax))
                        probs = torch.softmax(routing_logits, dim=0)
                        topk = torch.topk(probs, k=k).indices.tolist() if k > 0 else []
                        ctrl = control_tokens.detach().cpu().numpy()
                        selected = [{"idx": int(i), "name": self._registry.get(int(i)).name, "ctrl": ctrl[int(i)].tolist()} for i in topk]
                        self._iilstm_last_k = k
                        self._iilstm_last_selected = selected
                        self._iilstm_last_workers = self._registry.snapshot()
                        self.worker_registry_data.emit(json.dumps(self._iilstm_last_workers))
                        self._iilstm_last_routing_logits = routing_logits.detach().cpu().numpy().astype(float).tolist()
                        self._iilstm_last_k_logits = k_logits.detach().cpu().numpy().astype(float).tolist()
                    except Exception:
                        self._iilstm_last_k = 0
                        self._iilstm_last_selected = []
                        self._iilstm_last_workers = self._registry.snapshot()
                        self.worker_registry_data.emit(json.dumps(self._iilstm_last_workers))
                    # ---- Optional: execute PredictorWorker (safe mode) ----
                    executed = []
                    proposals = []
                    if self.worker_exec_enabled and self._predictor_worker_idx is not None and (self._predictor_worker_idx in [e["idx"] for e in selected]):
                        try:
                            latent_dim = int(live_combined_z.shape[0])
                            self._ensure_predictor_worker(latent_dim=latent_dim, hash_bits=64)
                            if self._predictor_worker is not None:
                                if self._last_pred_live_z is not None and len(self._last_pred_live_z) == latent_dim:
                                    live_pred_vec = self._last_pred_live_z.astype(np.float32)
                                else:
                                    live_pred_vec = np.zeros((latent_dim,), dtype=np.float32)
                                try:
                                    src_bits = frozen_pred_str
                                except NameError:
                                    src_bits = bits_live_str
                                fb = np.frombuffer(src_bits.encode('ascii'), dtype=np.uint8)
                                bits_vec = (fb == ord('1')).astype(np.float32)
                                step_np = np.concatenate([packet_np, live_pred_vec, bits_vec], axis=0).astype(np.float32)
                                step_t = torch.from_numpy(step_np).view(1, 1, -1)
                                pred_lat = self._predictor_worker(step_t)
                                pred_lat_np = pred_lat.detach().cpu().numpy().astype(np.float32)
                                executed.append({
                                    "idx": int(self._predictor_worker_idx),
                                    "name": self._registry.get(int(self._predictor_worker_idx)).name,
                                    "out": pred_lat_np.tolist()
                                })
                                if self.worker_use_predictor_output:
                                    self._last_pred_live_z = pred_lat_np.copy()
                                self._registry.update_stats(self._predictor_worker_idx, used=True, marginal_gain=None, conflicted=False)
                        except Exception:
                            pass
                    else:
                        if self._predictor_worker_idx is not None:
                            self._registry.update_stats(self._predictor_worker_idx, used=False, marginal_gain=None, conflicted=None)

                    # Mouse worker
                    if self.worker_exec_enabled and self.enable_mouse_worker and any(e["idx"] == self._mouse_worker_idx for e in selected):
                        try:
                            self._ensure_mouse_worker()
                            ctrl_vec = next(e["ctrl"] for e in selected if e["idx"] == self._mouse_worker_idx)
                            step_np = np.concatenate([packet_np, np.array(ctrl_vec, dtype=np.float32)], axis=0).astype(np.float32)
                            step_t = torch.from_numpy(step_np).view(1,1,-1)
                            out = self._mouse_worker(step_t)          # (6,)
                            out_np = out.detach().cpu().numpy().astype(np.float32)
                            dx_hat, dy_hat = float(out_np[0]), float(out_np[1])
                            proposals.append({
                                "type":"mouse",
                                "dx": np.tanh(dx_hat),
                                "dy": np.tanh(dy_hat),
                                "click": "left" if out_np[2] > 0.8 else ("right" if out_np[3] > 0.85 else "none"),
                                "down": bool(out_np[4] > 0.8),
                                "up":   bool(out_np[5] > 0.8),
                            })
                            self._registry.update_stats(self._mouse_worker_idx, used=True, marginal_gain=None, conflicted=False)
                            executed.append({"idx": int(self._mouse_worker_idx), "name": "mouse_v1", "out": out_np.tolist()})
                        except Exception:
                            self._registry.update_stats(self._mouse_worker_idx, used=False, marginal_gain=None, conflicted=True)

                    # Key worker
                    if self.worker_exec_enabled and self.enable_key_worker and any(e["idx"] == self._key_worker_idx for e in selected):
                        try:
                            self._ensure_key_worker(key_count=8)
                            ctrl_vec = next(e["ctrl"] for e in selected if e["idx"] == self._key_worker_idx)
                            step_np = np.concatenate([packet_np, np.array(ctrl_vec, dtype=np.float32)], axis=0).astype(np.float32)
                            step_t = torch.from_numpy(step_np).view(1,1,-1)
                            out = self._key_worker(step_t)            # (K+3,)
                            out_np = out.detach().cpu().numpy().astype(np.float32)
                            K = out_np.shape[0] - 3
                            key_idx = int(np.argmax(out_np[:K]))
                            p_down, p_up, p_noop = float(out_np[K]), float(out_np[K+1]), float(out_np[K+2])
                            if p_noop <= 0.5:
                                proposals.append({
                                    "type":"key",
                                    "code": key_idx,
                                    "down": bool(p_down > 0.6 and p_up < 0.5),
                                    "up":   bool(p_up   > 0.6 and p_down < 0.5),
                                })
                            self._registry.update_stats(self._key_worker_idx, used=True, marginal_gain=None, conflicted=False)
                            executed.append({"idx": int(self._key_worker_idx), "name":"key_v1", "out": out_np.tolist()})
                        except Exception:
                            self._registry.update_stats(self._key_worker_idx, used=False, marginal_gain=None, conflicted=True)
                    
                    # Audio worker
                    if self.worker_exec_enabled and self.enable_audio_worker and any(e["idx"] == self._audio_worker_idx for e in selected):
                        try:
                            self._ensure_audio_worker()
                            ctrl_vec = next(e["ctrl"] for e in selected if e["idx"] == self._audio_worker_idx)
                            # Audio worker uses the propose method like other workers
                            result = self._audio_worker.propose(int(self._tick_counter), np.array(ctrl_vec, dtype=np.float32))
                            if result["proposal"].get("type") == "audio" and result["proposal"].get("event") != "noop":
                                proposals.append(result["proposal"])
                            self._registry.update_stats(self._audio_worker_idx, used=True, marginal_gain=None, conflicted=False)
                            executed.append({"idx": int(self._audio_worker_idx), "name": "audio_v1", "out": result.get("raw", {})})
                        except Exception:
                            self._registry.update_stats(self._audio_worker_idx, used=False, marginal_gain=None, conflicted=True)

                    # Aggregate final action with executed worker filtering (no OS dispatch; just stash for logging)
                    final_action_local = {"type": "noop"}
                    executed_local = []
                    try:
                        bus_result = self._action_bus.aggregate(
                            proposals, tick=int(self._tick_counter),
                            energy=float(self._energy), params=self.action_params,
                            executed_workers=executed
                        )
                        final_action_local = bus_result.get("final_action", {"type": "noop"})
                        executed_local = bus_result.get("executed_workers", [])
                    except Exception:
                        pass

                    # Per-tick energy/pressure accounting handled unconditionally below
                else:
                    pkt_now = torch.zeros(64, device=self.torch_device)

                # Per-tick energy and pressure accounting (support module) — unconditional
                try:
                    self._sup.apply_energy_drain(final_action_local if 'final_action_local' in locals() else {"type":"noop"})
                    self._sup.update_sleep_pressure(awake=True)
                except Exception:
                    pass

                # Build log record now and push to buffer (reward uses next-tick drivers)
                try:
                    rec = {
                        'time': datetime.utcnow().isoformat() + 'Z',
                        'tick': self._tick_counter,
                        'hash_bits': bits_str,
                        'hash_chain': chained,
                        'frozen_z': [float(x) for x in frozen_combined_z.tolist()],
                        'live_z': [float(x) for x in live_z.tolist()],
                        'live_loss': float(recon_loss),
                        'hash_bits_live': bits_live_str,
                        'pred_hash_live': self._last_pred_live_bits_str,
                        'pred_acc_exact': (float(self._acc_live_correct) / float(max(1, self._acc_live_total))) if self._acc_live_total else None,
                        'pred_acc_sim': (float(self._acc_live_sim_sum) / float(max(1, self._acc_live_total))) if self._acc_live_total else None,
                        'd_last_mse': self._last_d_loss,
                        'd_last_hash_sim': self._last_d_hash_sim,
                        'novelty': float(self._nov_ema),
                        'nov_latent_pred_mse': float(self._last_latent_pred_mse),
                        'nov_hash_delta_live': float(self._nov_last_hash_delta),
                        'nov_pred_hash_wrong': float(1.0 - (self._acc_live_sim_sum / max(1, self._acc_live_total))) if self._acc_live_total else 0.0,
                        'lsh_action': action,
                        'lsh_centroid': centroid,
                        'lsh_active_count': int(self.lsh.get_active_count()),
                        'lsh_archived_count': int(self.lsh.get_archived_count()),
                        'iilstm_k': int(self._iilstm_last_k),
                        'iilstm_selected': self._iilstm_last_selected,
                        'iilstm_workers': self._iilstm_last_workers,
                        'iilstm_routing_logits': self._iilstm_last_routing_logits,
                        'iilstm_k_logits': self._iilstm_last_k_logits,
                        'drivers': {
                            'novelty': float(self._nov_ema),
                            'energy': float(self._energy),
                            'sleep_pressure': float(self._sleep_pressure)
                        },
                        'executed_workers': executed_local,
                        'final_action': final_action_local,
                    }
                    if self._log_context:
                        rec.update(self._log_context)
                    self._log_buffer.append(rec)
                    # Periodic flush to ensure log file is created and updated during runtime
                    try:
                        if (self._tick_counter % 50) == 0 or len(self._log_buffer) >= 200:
                            self._flush_log_buffer()
                    except Exception:
                        pass
                except Exception:
                    rec = None

                # ---- World predictor transitions (state -> next-state) ----
                try:
                    if rec is not None:
                        if self._last_rec is not None:
                            self._last_rec["live_z_next"] = rec["live_z"]
                            self._exp.append(self._last_rec)
                        self._last_rec = {
                            "time": rec.get("time"),
                            "tick": rec.get("tick"),
                            "live_z": rec.get("live_z"),
                            "frozen_z": rec.get("frozen_z"),
                            "drivers": rec.get("drivers"),
                            "hash_bits_live": rec.get("hash_bits_live"),
                            "hash_bits": rec.get("hash_bits"),
                            "final_action": rec.get("final_action"),
                            "iilstm_selected": rec.get("iilstm_selected"),
                            "iilstm_k": rec.get("iilstm_k"),
                        }
                except Exception:
                    pass

                # ---- Experience buffer push (computes reward for previous tick) ----
                try:
                    if rec is not None:
                        self._exp.push(rec, self._reward_from_next)
                except Exception:
                    pass

                # Sleep gate: trigger when energy/pressure thresholds hit (single-shot)
                try:
                    should_sleep = self._sup.should_enter_sleep(self._tick_counter)
                except Exception as e:
                    should_sleep = False
                    try:
                        self.diag_info.emit(f"ERROR in should_enter_sleep: {e}")
                    except Exception:
                        pass
                
                # IMMEDIATE DEBUG: Log should_sleep result every tick when energy is very low
                if self._energy <= 0.05:
                    try:
                        self.diag_info.emit(f"CRITICAL DEBUG: tick={self._tick_counter}, energy={self._energy:.3f}, pressure={self._sleep_pressure:.3f}, should_sleep={should_sleep}, sleep_active={self._sleep_active}")
                    except Exception:
                        pass
                
                # TEMPORARY: Force sleep for testing if energy is 0 and we haven't slept recently
                if self._energy <= 0.0 and not self._sleep_active:
                    last_sleep = getattr(self, "_last_sleep_tick", -1)
                    if self._tick_counter - last_sleep > 50:  # At least 50 ticks since last sleep
                        should_sleep = True
                        try:
                            self.diag_info.emit(f"FORCE SLEEP: tick={self._tick_counter}, energy={self._energy:.3f}")
                        except Exception:
                            pass
                # Debug: log sleep gate evaluation every 50 ticks
                if self._tick_counter % 50 == 0:
                    try:
                        self.diag_info.emit(f"Sleep gate: tick={self._tick_counter}, energy={self._energy:.3f}, pressure={self._sleep_pressure:.3f}, should_sleep={should_sleep}, active={self._sleep_active}")
                    except Exception:
                        pass
                
                # Additional debug: log when sleep conditions are met
                if should_sleep and not self._sleep_active:
                    try:
                        self.diag_info.emit(f"DEBUG: Sleep conditions met! tick={self._tick_counter}, energy={self._energy:.3f}, pressure={self._sleep_pressure:.3f}")
                    except Exception:
                        pass
                
                # Debug cooldown logic
                if self._tick_counter % 100 == 0:  # Every 100 ticks
                    try:
                        last_sleep = getattr(self, "_last_sleep_tick", -1)
                        last_wake = getattr(self, "_last_wake_tick", 0)
                        cooldown_ticks = int(self.energy_params.get("sleep_cooldown_ticks", 500))
                        min_awake_ticks = int(self.energy_params.get("min_awake_ticks", 200))
                        ticks_since_sleep = self._tick_counter - last_sleep
                        ticks_since_wake = self._tick_counter - last_wake
                        self.diag_info.emit(f"DEBUG: Cooldown check - tick={self._tick_counter}, last_sleep={last_sleep}, last_wake={last_wake}, since_sleep={ticks_since_sleep}, since_wake={ticks_since_wake}, cooldown={cooldown_ticks}, min_awake={min_awake_ticks}")
                    except Exception:
                        pass
                if (not self._sleep_active) and should_sleep:
                    self._sleep_active = True
                    try:
                        # capture previous sleep tick BEFORE we mark a new one
                        _prev_last_sleep_tick = int(getattr(self, "_last_sleep_tick", -1))
                        
                        # Choose replay source
                        mode = str(self.sleep_params.get("replay_mode", "drain")).lower()
                        if mode not in ("drain", "snapshot"):
                            mode = "drain"
                        replay = self._exp.drain() if mode == "drain" else self._exp.snapshot()

                        # Minimum replay gate
                        min_replay = int(self.sleep_params.get("min_replay", 50))
                        if len(replay) < min_replay:
                            try:
                                self.diag_warning.emit(f"Sleep skipped: insufficient replay (have={len(replay)} < min={min_replay})")
                            except Exception:
                                pass
                            self._log_sleep_metric(event="skip", reason="insufficient_replay", have=int(len(replay)), min=int(min_replay))
                            try:
                                self._sup.note_wake(self._tick_counter)
                            except Exception:
                                pass
                            self._sleep_active = False
                            self._exp.start_episode()
                        else:
                            try:
                                self.diag_warning.emit(
                                    f"Entering sleep: mode={mode}, tick={self._tick_counter}, "
                                    f"energy={self._energy:.3f}, pressure={self._sleep_pressure:.3f}, "
                                    f"replay_size={len(replay)}"
                                )
                            except Exception:
                                pass

                            # log entry using the previous tick
                            self._sleep_log_ctx = {
                                "start_ts": time.perf_counter(),
                                "entry_energy": float(self._energy),
                                "entry_pressure": float(self._sleep_pressure),
                                "mode": mode,
                                "replay_size": int(len(replay)),
                                "last_sleep_tick": _prev_last_sleep_tick,
                            }
                            self._log_sleep_metric(
                                event="enter",
                                mode=mode,
                                energy=float(self._energy),
                                pressure=float(self._sleep_pressure),
                                replay_size=int(len(replay)),
                                last_sleep_tick=_prev_last_sleep_tick,
                                thr_energy=float(getattr(self, 'energy_params', {}).get("sleep_entry_energy", 0.1)),
                                thr_pressure=float(getattr(self, 'energy_params', {}).get("sleep_entry_pressure", 0.8))
                            )
                            
                            # NOW mark the new sleep start (don't do this before the log)
                            try:
                                self._sup.note_sleep_start(self._tick_counter)
                            except Exception:
                                pass

                            # Surface state changes to GUI and also log phases via on_state
                            try:
                                self.sleep_state_text.emit("nrem")
                            except Exception:
                                pass
                            self._sleep.run(
                                get_replay=lambda: replay,
                                nrem_hook=self._nrem_hook,
                                rem_hook=self._rem_hook,
                                recharge=self._recharge,
                                on_state=lambda phase: self._log_sleep_metric(event="phase", phase=str(phase))
                            )
                            try:
                                self.sleep_state_text.emit("awake")
                            except Exception:
                                pass

                            # completion metrics
                            try:
                                dur_ms = int((time.perf_counter() - self._sleep_log_ctx.get("start_ts", time.perf_counter())) * 1000)
                            except Exception:
                                dur_ms = 0
                            
                            # pull any summary the hooks built, append defaults if missing
                            summ = getattr(self, "_sleep_diag", {}) or {}
                            # ensure we also keep existing replay/REM info your SleepManager put there
                            # nothing else to do if SleepManager already added those
                            self._sleep_diag = summ
                            
                            self._log_sleep_metric(event="complete", duration_ms=dur_ms, summary=summ)

                            try:
                                self._sup.update_sleep_pressure(awake=False)
                                self._sup.note_wake(self._tick_counter)
                            except Exception:
                                pass

                            try:
                                sd = getattr(self, "_sleep_diag", {})
                                self.diag_info.emit(f"Sleep summary: {sd}")
                            except Exception:
                                pass

                            # Flush logs after sleep
                            try:
                                if len(self._log_buffer) > 0:
                                    self._flush_log_buffer()
                                if len(self._sleep_log_buffer) > 0:
                                    self._flush_sleep_log_buffer()
                            except Exception:
                                pass

                            self._exp.start_episode()
                            self._mode = "awake"
                            continue  # next tick after sleep
                    finally:
                        # ensure guard always clears
                        self._sleep_active = False

                # Frozen predicted hash bits for next step (from current bits_str)
                frozen_pred_str = self._hash_predictor.predict(bits_str)
                if frozen_pred_str is None:
                    frozen_pred_str = bits_str
                fb = np.frombuffer(frozen_pred_str.encode('ascii'), dtype=np.uint8)
                frozen_bits_vec = (fb == ord('1')).astype(np.float32)
                frozen_bits_t = torch.tensor(frozen_bits_vec, device=self.torch_device)

                # Live latent prediction input from simple predictor - extend with zeros for audio
                z_next_pred_combined = np.concatenate([z_next_pred, np.zeros(self.audio_latent_dim)])
                live_pred_now = torch.tensor(z_next_pred_combined.astype(np.float32), device=self.torch_device)

                # Build D rolling input and append
                step_vec = torch.cat([pkt_now, live_pred_now, frozen_bits_t], dim=0)
                self._d_seq.append(step_vec.detach())

                # Make and emit D-pipe prediction frame using rolling sequence
                if len(self._d_seq) > 0:
                    d_seq_np = torch.stack(list(self._d_seq), dim=0).to(self.torch_device).view(-1, 1, 64 + self.combined_latent_dim + self.lsh_bits)
                    self.d_pipe.reset_state()
                    d_pred_lat = self.d_pipe(d_seq_np)
                else:
                    d_pred_lat = torch.zeros(self.combined_latent_dim, device=self.torch_device)
                d_pred_np = d_pred_lat.detach().cpu().numpy()
                # For decoding, only use the vision component
                d_pred_vision_np = d_pred_np[:self.latent_dim]  
                d_pred_frame_flat = self.live_vae.decode(d_pred_vision_np)
                w, h = self.downsample_width, self.downsample_height
                img_d = (np.clip(d_pred_frame_flat, 0.0, 1.0) * 255.0).astype(np.uint8).reshape(h, w, 3)
                img_d_large = cv2.resize(img_d, (640, 480), interpolation=cv2.INTER_NEAREST)
                self.dpipe_prediction_frame.emit(img_d_large[:, :, ::-1])
                # Save for next-tick MSE vs actual live_z
                self._last_pred_live_z = z_next_pred.copy()

                # Queue current sample as pending for next tick training
                self._pending_train_sample = {
                    'pc_seq_tensor': seq_t.detach() if len(self._pc_seq) > 0 else torch.zeros(1, 1, 3 * self.combined_latent_dim, device=self.torch_device),
                    'd_seq_tensor': d_seq_np.detach() if len(self._d_seq) > 0 else torch.zeros(1, 1, 64 + self.combined_latent_dim + self.lsh_bits, device=self.torch_device),
                    'frozen_lat': torch.tensor(frozen_combined_z.astype(np.float32), device=self.torch_device),
                }
                # Update last live pred for PC window
                self._last_live_pred_for_pc = z_next_pred_combined.astype(np.float32)
            except Exception:
                pass

    def __del__(self):
        self.stop()
        # best-effort flush on destruction
        try:
            if len(self._log_buffer) > 0:
                self._flush_log_buffer()
            if len(self._sleep_log_buffer) > 0:
                self._flush_sleep_log_buffer()
        except Exception:
            pass

    # ---------------------- Hash map utils ----------------------
    def _update_hash_collection(self, bits: np.ndarray):
        code_str = self.lsh.bits_to_str(bits)
        self._current_bits = bits.copy()
        if code_str not in self._codes_seen:
            self._codes_seen.add(code_str)
            # store copy to avoid mutation
            self._codes_bits.append(bits.copy())
            self._code_to_index[code_str] = len(self._codes_bits) - 1
        # Track current index
        self._current_index = self._code_to_index.get(code_str, -1)

    def _recompute_embedding_if_needed(self):
        n = len(self._codes_bits)
        if n == 0:
            return
        # Rebuild index mapping (no truncation; keep all hashes)
        self._code_to_index = {}
        for i, b in enumerate(self._codes_bits):
            try:
                self._code_to_index[self.lsh.bits_to_str(b)] = i
            except Exception:
                pass
        # Refresh current index if possible
        if self._current_bits is not None:
            cur_str = self.lsh.bits_to_str(self._current_bits)
            self._current_index = self._code_to_index.get(cur_str, -1)
        # Compute normalized Hamming distances
        B = np.stack(self._codes_bits, axis=0).astype(np.uint8)
        # distance matrix
        D = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            # vectorized xor
            xor_rows = np.bitwise_xor(B[i], B)
            D[i] = np.sum(xor_rows, axis=1, dtype=np.int32) / float(B.shape[1])
        self._D = D
        # Classical MDS to 3D (guard for small n)
        J = np.eye(n, dtype=np.float32) - np.ones((n, n), dtype=np.float32) / n
        D2 = D ** 2
        Bmat = -0.5 * (J @ D2 @ J)
        # eigh for symmetric
        vals, vecs = np.linalg.eigh(Bmat)
        order = np.argsort(vals)
        take = min(3, n)
        idx = order[-take:]
        vals_sel = np.clip(vals[idx], a_min=1e-9, a_max=None)
        vecs_sel = vecs[:, idx]
        coords = vecs_sel * np.sqrt(vals_sel)
        # If less than 3 dims, pad with zeros to (n,3)
        if coords.shape[1] < 3:
            pad = np.zeros((n, 3 - coords.shape[1]), dtype=coords.dtype)
            coords = np.concatenate([coords, pad], axis=1)
        self._coords_3d = coords.astype(np.float32)

        # Compute clusters by union-find in Hamming space using threshold
        parents = list(range(n))

        def find(a: int) -> int:
            while parents[a] != a:
                parents[a] = parents[parents[a]]
                a = parents[a]
            return a

        def union(a: int, b: int):
            ra, rb = find(a), find(b)
            if ra != rb:
                parents[rb] = ra

        thr = self.cluster_threshold
        if n > 1:
            for i in range(n):
                for j in range(i + 1, n):
                    if D[i, j] <= thr:
                        union(i, j)
        clusters_map: dict[int, list[int]] = {}
        for i in range(n):
            r = find(i)
            clusters_map.setdefault(r, []).append(i)
        self._clusters = list(clusters_map.values())
        # Generate colors per cluster (BGR)
        self._cluster_colors = self._generate_palette(len(self._clusters))

    def _render_hashmap_image(self, width: int, height: int) -> np.ndarray | None:
        if self._coords_3d is None or self._coords_3d.shape[0] == 0:
            return None
        coords = self._coords_3d
        # Normalize to unit cube
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        span = np.maximum(maxs - mins, 1e-6)
        norm = (coords - mins) / span
        # Simple perspective projection with fixed camera
        # Map x->width, y->height, use z for point size
        xs = (norm[:, 0] * (width - 40) + 20).astype(np.int32)
        ys = (norm[:, 1] * (height - 40) + 20).astype(np.int32)
        zs = norm[:, 2]
        # Start BGR canvas for cv2 drawing
        img = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Grid and axes
        grid_color = (220, 220, 220)
        axis_color = (160, 160, 160)
        for gx in range(40, width, 40):
            cv2.line(img, (gx, 20), (gx, height - 20), grid_color, 1, cv2.LINE_AA)
        for gy in range(40, height, 40):
            cv2.line(img, (20, gy), (width - 20, gy), grid_color, 1, cv2.LINE_AA)
        cv2.rectangle(img, (20, 20), (width - 20, height - 20), axis_color, 1, cv2.LINE_AA)

        # Draw cluster translucent bubbles first (behind points)
        for idx_cluster, members in enumerate(self._clusters):
            if len(members) == 0:
                continue
            cx = int(np.mean(xs[members]))
            cy = int(np.mean(ys[members]))
            if len(members) == 1:
                radius = 18
            else:
                dx = xs[members] - cx
                dy = ys[members] - cy
                dist = np.sqrt(dx * dx + dy * dy)
                radius = int(min(max(np.max(dist) * 1.4 + 16, 18), min(width, height) // 2))
            color = self._cluster_colors[idx_cluster]
            self._draw_filled_circle_alpha(img, (cx, cy), radius, color, alpha=0.18)
            cv2.circle(img, (cx, cy), radius, tuple(int(c * 0.8) for c in color), 1, cv2.LINE_AA)

        # Determine current index and neighbors
        current_idx = self._current_index
        n = xs.shape[0]
        neighbors_idx: list[int] = []
        if current_idx is not None and 0 <= current_idx < n and n > 1:
            try:
                if self._D is not None and self._D.shape[0] == n:
                    dvec = self._D[current_idx]
                else:
                    BB = np.stack(self._codes_bits, axis=0).astype(np.uint8)
                    dvec = np.sum(np.bitwise_xor(BB[current_idx], BB), axis=1).astype(np.float32) / float(BB.shape[1])
                order = np.argsort(dvec)
                # skip self at order[0]
                k = min(5, n - 1)
                neighbors_idx = order[1:1 + k].tolist()
            except Exception:
                neighbors_idx = []

        # Draw neighbor lines
        if current_idx is not None and 0 <= current_idx < n:
            for j in neighbors_idx:
                if 0 <= j < n:
                    thickness = 1 + int(2 * (1.0 - float(abs(zs[current_idx] - zs[j]))))
                    cv2.line(img, (int(xs[current_idx]), int(ys[current_idx])), (int(xs[j]), int(ys[j])), (120, 200, 120), thickness, cv2.LINE_AA)

        # Draw points with depth sorting and shading
        current_color = (0, 0, 255)
        sizes = (2 + (1.0 - zs) * 5).astype(np.int32)
        order = np.argsort(zs)  # far to near
        for i in order:
            x, y, r = int(xs[i]), int(ys[i]), int(max(sizes[i], 2))
            # color by cluster
            color = (180, 160, 120)
            for idx_cluster, members in enumerate(self._clusters):
                if i in members:
                    color = self._cluster_colors[idx_cluster]
                    break
            # neighbor override tint
            if i in neighbors_idx:
                color = (int(0.6 * color[0] + 0.4 * 255), int(0.6 * color[1] + 0.4 * 180), int(0.6 * color[2] + 0.4 * 80))
            # shading by depth
            shade = 0.55 + 0.45 * (1.0 - float(zs[i]))
            color_shaded = (int(min(255, color[0] * shade)), int(min(255, color[1] * shade)), int(min(255, color[2] * shade)))
            cv2.circle(img, (x, y), r, color_shaded, -1, cv2.LINE_AA)

        # Highlight current point on top
        if current_idx is not None and 0 <= current_idx < n:
            # subtle shadow
            cv2.circle(img, (int(xs[current_idx]) + 1, int(ys[current_idx]) + 1), int(max(sizes[current_idx] + 2, 4)), (80, 80, 80), -1, cv2.LINE_AA)
            cv2.circle(img, (int(xs[current_idx]), int(ys[current_idx])), int(max(sizes[current_idx] + 1, 3)), current_color, -1, cv2.LINE_AA)

        # Legend
        legend_x, legend_y = 28, 28
        cv2.rectangle(img, (legend_x - 8, legend_y - 8), (legend_x + 180, legend_y + 64), (245, 245, 245), -1, cv2.LINE_AA)
        cv2.rectangle(img, (legend_x - 8, legend_y - 8), (legend_x + 180, legend_y + 64), (210, 210, 210), 1, cv2.LINE_AA)
        cv2.circle(img, (legend_x, legend_y), 5, current_color, -1, cv2.LINE_AA)
        cv2.putText(img, 'current', (legend_x + 14, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
        # cluster sample swatches
        swy = legend_y + 20
        for cidx in range(min(3, len(self._cluster_colors))):
            cv2.circle(img, (legend_x, swy + 20 * cidx), 5, self._cluster_colors[cidx], -1, cv2.LINE_AA)
            cv2.putText(img, f'cluster {cidx+1}', (legend_x + 14, swy + 20 * cidx + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(img, f'N={n}', (legend_x + 100, legend_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)

        # Convert BGR -> RGB for GUI consumption
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb

    def _emit_heatmap_hit(self):
        # Emit normalized (x,y) in [0,1] for current index using current coords_3d
        if self._coords_3d is None or self._coords_3d.shape[0] == 0:
            return
        idx = self._current_index
        if idx is None or idx < 0 or idx >= self._coords_3d.shape[0]:
            return
        coords = self._coords_3d
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        span = np.maximum(maxs - mins, 1e-6)
        norm = (coords - mins) / span
        x = float(np.clip(norm[idx, 0], 0.0, 1.0))
        y = float(np.clip(norm[idx, 1], 0.0, 1.0))
        self.heatmap_hit.emit(x, y)

    # ---------------------- Logging helpers ----------------------
    def add_log_field(self, key: str, value):
        """Add or update a persistent field to be included in every log record."""
        self._log_context[key] = value

    def _flush_log_buffer(self):
        if len(self._log_buffer) == 0:
            return
        # Append JSON lines
        with open(self._log_file_path, 'a', encoding='utf-8') as f:
            for rec in self._log_buffer:
                f.write(json.dumps(rec, separators=(',', ':'), ensure_ascii=False))
                f.write('\n')
        self._log_buffer.clear()

    def _flush_sleep_log_buffer(self):
        if len(self._sleep_log_buffer) == 0:
            return
        # Append JSON lines to sleep log file
        try:
            with open(self._sleep_log_file_path, 'a', encoding='utf-8') as f:
                for rec in self._sleep_log_buffer:
                    f.write(json.dumps(rec, separators=(',', ':'), ensure_ascii=False))
                    f.write('\n')
            self._sleep_log_buffer.clear()
        except Exception:
            # If sleep log fails, don't crash the whole system
            self._sleep_log_buffer.clear()

    def _format_hash_stats(self) -> str:
        n = len(self._codes_bits)
        k = self.lsh.num_bits
        num_clusters = len(self._clusters)
        cluster_sizes = [len(c) for c in self._clusters]
        largest = max(cluster_sizes) if cluster_sizes else 0
        smallest = min(cluster_sizes) if cluster_sizes else 0
        avg = (float(sum(cluster_sizes)) / max(1, len(cluster_sizes))) if cluster_sizes else 0.0
        # Top cluster sizes
        top_sizes = sorted(cluster_sizes, reverse=True)[:5]
        top_sizes_str = ", ".join(str(s) for s in top_sizes) if top_sizes else ""

        # Current context distances
        cur_line = "Current: —"
        if self._current_index is not None and 0 <= self._current_index < n and self._D is not None and self._D.shape[0] == n:
            dvec = self._D[self._current_index]
            # exclude self for stats
            if n > 1:
                d_ex = np.delete(dvec, self._current_index)
                cur_line = f"Current distances → min: {np.min(d_ex):.3f}, median: {np.median(d_ex):.3f}, mean: {np.mean(d_ex):.3f}"
            else:
                cur_line = "Current distances → min: 0.000, median: 0.000, mean: 0.000"

        # Intra-/inter-cluster distances (approximate)
        intra_avg = 0.0
        if self._D is not None and self._D.shape[0] == n and num_clusters > 0:
            intra_vals = []
            for members in self._clusters:
                m = len(members)
                if m >= 2:
                    sub = self._D[np.ix_(members, members)]
                    iu = np.triu_indices(m, k=1)
                    if iu[0].size > 0:
                        intra_vals.append(float(np.mean(sub[iu])))
            intra_avg = float(np.mean(intra_vals)) if intra_vals else 0.0

        inter_avg = 0.0
        if self._D is not None and self._D.shape[0] == n and num_clusters > 1:
            # Use cluster medoids for efficiency
            medoids = []
            for members in self._clusters:
                if len(members) == 0:
                    continue
                if len(members) == 1:
                    medoids.append(members[0])
                else:
                    sub = self._D[np.ix_(members, members)]
                    sums = np.sum(sub, axis=1)
                    medoids.append(members[int(np.argmin(sums))])
            inter_vals = []
            M = len(medoids)
            for i in range(M):
                for j in range(i + 1, M):
                    inter_vals.append(float(self._D[medoids[i], medoids[j]]))
            inter_avg = float(np.mean(inter_vals)) if inter_vals else 0.0

        # Bit density across codes
        ones_ratio_mean = 0.0
        ones_ratio_std = 0.0
        if n > 0:
            B = np.stack(self._codes_bits, axis=0).astype(np.uint8)
            ones_per_bit = np.mean(B, axis=0)
            ones_ratio_mean = float(np.mean(ones_per_bit))
            ones_ratio_std = float(np.std(ones_per_bit))

        text = [
            f"Total unique hashes: {n}",
            f"Code length (k): {k}",
            f"Clusters: {num_clusters}",
            f"Cluster sizes → min: {smallest}, avg: {avg:.2f}, max: {largest}",
            f"Top clusters (sizes): {top_sizes_str}",
            cur_line,
            f"Intra-cluster avg distance: {intra_avg:.3f}",
            f"Inter-cluster avg distance: {inter_avg:.3f}",
            f"Bit density (mean±std of ones): {ones_ratio_mean:.3f} ± {ones_ratio_std:.3f}",
        ]
        return "\n".join(text)

    def _generate_palette(self, n: int) -> list[tuple[int, int, int]]:
        if n <= 0:
            return []
        colors = []
        for i in range(n):
            hue = int(180.0 * i / max(1, n))  # OpenCV HSV hue range [0,180)
            hsv = np.uint8([[[hue, 160, 230]]])  # (1,1,3)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
        return colors

    def _draw_filled_circle_alpha(self, img: np.ndarray, center: tuple[int, int], radius: int, color_bgr: tuple[int, int, int], alpha: float):
        overlay = img.copy()
        cv2.circle(overlay, center, int(radius), color_bgr, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def _normalized_hamming_between_bitstrings(self, a: str, b: str) -> float:
        # assumes same length strings of '0'/'1'
        if not a or not b or len(a) != len(b):
            return 1.0
        diffs = sum(1 for ca, cb in zip(a, b) if ca != cb)
        return diffs / float(len(a))


# ---------------------- Prediction Models ----------------------
class _HashPredictor:
    def __init__(self):
        self.transitions: dict[str, dict[str, int]] = {}

    def update(self, current_bits_str: str, next_bits_str: str):
        bucket = self.transitions.setdefault(current_bits_str, {})
        bucket[next_bits_str] = bucket.get(next_bits_str, 0) + 1

    def predict(self, current_bits_str: str) -> str | None:
        bucket = self.transitions.get(current_bits_str)
        if not bucket:
            return None
        # return most frequent next
        return max(bucket.items(), key=lambda kv: kv[1])[0]

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.transitions, f, separators=(',', ':'))

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            self.transitions = json.load(f)


class _LiveLatentPredictor:
    def __init__(self, latent_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.latent_dim = latent_dim
        # Start near identity for stable one-step prediction baseline
        self.A = np.eye(latent_dim, dtype=np.float32) + (rng.standard_normal((latent_dim, latent_dim)) * 0.005).astype(np.float32)
        self.b = np.zeros((latent_dim,), dtype=np.float32)
        self.lr = 5e-3

    def predict(self, z_t: np.ndarray) -> np.ndarray:
        return (self.A @ z_t) + self.b

    def update(self, z_t: np.ndarray, z_t1: np.ndarray):
        # One-step SGD on MSE for next-step prediction
        pred = self.predict(z_t)
        err = pred - z_t1
        N = z_t.shape[0]
        grad_b = (2.0 / N) * err
        grad_A = np.outer((2.0 / N) * err, z_t)
        self.A -= self.lr * grad_A
        self.b -= self.lr * grad_b

    def save(self, path: str):
        np.savez(path, A=self.A, b=self.b)

    def load(self, path: str):
        data = np.load(path)
        self.A = data['A']
        self.b = data['b']



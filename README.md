# OLM MoE - Modular Sleep-Driven Control Architecture

A modular control architecture that learns from consequence rather than live backpropagation, featuring sleep-driven consolidation, locality-sensitive hashing for novelty detection, and comprehensive safety mechanisms.

## Overview

This system implements a bio-inspired learning architecture that separates wake and sleep phases:
- **Wake Phase**: Compresses sensory inputs to latent states, predicts transitions, routes control proposals, and logs all experiences
- **Sleep Phase**: Consolidates learning through deterministic replay (NREM) and sandboxed imagination (REM)
- **Safety First**: Emergency kill-switch, strict action gating, and intentionally disabled live OS dispatch during development

## Key Features

### 🧠 Sleep-Driven Learning
- **NREM Phase**: Deterministic replay for world model and worker consolidation
- **REM Phase**: Latent rollouts for policy exploration (diagnostic only)
- **Energy Homeostasis**: Action costs drain energy, sleep restores it

### 🔒 Safety Architecture
- Emergency kill-switch (END key) instantly disables all actions
- All actions are logged and clamped by policy
- Live OS dispatch intentionally withheld during skeleton phase
- Energy-based action gating prevents excessive activity

### 📊 Novelty Detection
- **LSH-based clustering** of latent states for memory organization
- **Multi-component novelty**: prediction error, hash dynamics, latent changes
- Novelty drives exploration pressure and sleep scheduling

### 🎯 Modular Design
- **Dual VAE**: Encodes sensory inputs to 32D latent space with frozen diagnostics
- **World Predictor**: Learns latent dynamics f(z_t) → z_{t+1}
- **IILSTM Router**: Selects k workers from available pool
- **Action Workers**: Mouse and keyboard controllers with discrete LSTM architecture
- **Experience Buffer**: Logs every tick for auditable training

## Architecture Components

### Core Modules
- `dual_vae.py` - Sensory encoding and latent compression
- `predictor_worker.py` - World model learning latent transitions
- `iilstm.py` - Internal-Integrator LSTM for worker routing
- `action_bus.py` - Action aggregation with safety clamps
- `lsh_system.py` - Locality-sensitive hashing for novelty
- `experience_buffer.py` - Experience logging and replay management
- `sleep_manager.py` - Sleep phase orchestration

### Worker System
- `mouse_worker.py` - Bounded mouse movement and click control
- `key_worker.py` - Small-vocabulary keyboard control with cooldowns
- `audio_worker.py` - Audio processing and speech recognition
- Worker registry for modular action proposal system

### Interface & Visualization
- GUI with real-time energy, pressure, and phase monitoring
- 3D heatmap visualization for attention systems
- Audio processing interface with speech-to-text
- Multiple display windows for system diagnostics

## Learning Process

### Wake Cycle
1. **Sensory Input** → VAE encoder → latent state `z`
2. **Prediction** → World model estimates `z_next`
3. **Novelty Computation** → LSH hashing + prediction error
4. **Routing** → IILSTM selects k workers based on context
5. **Action Execution** → ActionBus aggregates and safety-clamps
6. **Experience Logging** → All transitions recorded for replay

### Sleep Cycle
1. **NREM Phase**: 
   - Deterministic replay of recent experiences
   - World model consolidation with validation-based promotion
   - Worker imitation learning with acceptance criteria
2. **REM Phase**:
   - Latent-space rollouts for policy exploration
   - Diagnostic statistics (no promotion by default)
3. **Recharge**: Energy → 1.0, pressure → 0.0

## Safety & Testing

### Safety Mechanisms
- **Kill Switch**: END key immediately disables all worker execution
- **Energy Gating**: Actions require minimum energy threshold
- **Magnitude Clamping**: Mouse movements bounded, key events rate-limited
- **Execution Flags**: Global and per-worker enable/disable controls
- **Audit Trail**: Every action logged with full context

### Acceptance Criteria
- World predictor: Promote only if validation MSE improves by ε_abs=1e-6 or ε_rel=1e-3
- Workers: Promote only if movement MAE decreases and click/key accuracies increase
- LSH: Collision rate must remain stable post-maintenance
- Emergency hotkey forces noop within same tick

## Configuration

### Energy Parameters
```python
energy_params = {
    "drain_base": 0.001,
    "drain_per_mouse_px": 1e-4,
    "drain_per_click": 0.002,
    "drain_per_key": 0.0005,
    "sleep_entry_energy": 0.05,
    "sleep_entry_pressure": 0.995,
    "energy_min": 0.02
}
```

### Sleep Parameters
- `rem_steps`: 6 (REM rollout length)
- `rem_batch`: 4 (batch size for REM)
- `sleep_cooldown_ticks`: 500
- `min_awake_ticks`: 200

## Performance Monitoring

### Key Performance Indicators
- **world_mse_val**: Validation MSE of latent predictor (target: downward)
- **imitation_mae**: Worker movement accuracy (target: downward) 
- **click/key_acc**: Event classification accuracy (target: upward)
- **routing_entropy**: Worker selection diversity (target: >0.5)
- **collision_rate**: LSH bucket collision stability
- **sleep_latency_ms**: Sleep episode duration consistency

## Development Status

### Implemented ✅
- Core architecture with modular components
- Sleep-driven consolidation (NREM/REM phases)
- LSH-based novelty detection
- Comprehensive safety mechanisms
- Experience logging and deterministic replay
- GUI with real-time monitoring

### Known Limitations
- IILSTM routing promotion during REM not implemented
- Expectation buffer for novelty modulation pending
- OS dispatcher intentionally disabled (safety by design)
- VAE decoder reconsolidation deferred
- Minimal sleep visualization in GUI

### Roadmap
- **Milestone B**: Robust NREM predictor consolidation with persistence
- **Milestone C**: Conservative REM advantage-weighted imitation
- **Milestone D**: Expectation buffer and reality-aligned novelty
- **Milestone E**: Guarded OS dispatch behind master safety switch

## Usage

The system runs in skeleton mode by default - all learning and routing occurs normally, but no actual OS commands are dispatched. This enables safe development and testing of the learning architecture.

Emergency shutdown: Press END key to immediately disable all worker execution.

## Technical Details

- **Latent Dimension**: 32D compressed representation
- **Architecture**: Python with PyTorch/NumPy backend
- **Worker Pattern**: Discrete LSTM controllers
- **Memory System**: LSH with binary hash clustering
- **Replay System**: Deterministic train/val splits (90/10)
- **Promotion Protocol**: Clone-train-validate-promote with rollback protection

---

*This implementation prioritizes safety, modularity, and auditability over immediate functionality. The sleep-driven architecture enables consequence-based learning while maintaining strict safety boundaries during development.*
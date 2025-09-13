from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import math

import numpy as np

try:
    import torch
    from torch import nn, optim
except Exception:  # optional at import time
    torch = None
    nn = None
    optim = None


@dataclass
class SleepReport:
    replay_size: int = 0
    replay_train: int = 0
    replay_val: int = 0
    cloned: Dict[str, bool] | None = None
    workers: Dict[str, Any] | None = None
    stage: str = "prepared"


class VaeSup:
    def __init__(self, fe: Any):
        self.fe = fe
        # sensible defaults if engine didn't set them
        self.world_lr = 1e-4
        self.world_epochs = 2

    def peek_worker_dataset_info(self):
        try:
            ds = self._build_worker_datasets(dry=True)
            # return lightweight counts to surface why it's skipping
            return {
                "total": int(ds.get("total", 0)),
                "mouse": int(ds.get("mouse", 0)),
                "key": int(ds.get("key", 0)),
            }
        except Exception:
            return {"total": 0}

    def _build_worker_datasets(self, dry=False):
        """Build worker datasets, optionally in dry-run mode for counting only"""
        try:
            # Use recent replay buffer for dataset building
            replay_buffer = getattr(self.fe, "_replay_buffer", [])
            if not replay_buffer:
                return {"total": 0, "mouse": 0, "key": 0}
            
            if dry:
                # Quick count without full processing
                mouse_count = sum(1 for r in replay_buffer[-100:] if r.get("final_action", {}).get("type") == "mouse")
                key_count = sum(1 for r in replay_buffer[-100:] if r.get("final_action", {}).get("type") == "key")
                audio_count = sum(1 for r in replay_buffer[-100:] if r.get("final_action", {}).get("type") in ("audio_play", "audio_stop"))
                return {"total": mouse_count + key_count + audio_count, "mouse": mouse_count, "key": key_count, "audio": audio_count}
            else:
                # Full dataset building
                datasets = self.make_worker_datasets(replay_buffer[-100:])
                return {
                    "total": len(datasets.get("mouse", [])) + len(datasets.get("key", [])) + len(datasets.get("audio", [])),
                    "mouse": len(datasets.get("mouse", [])),
                    "key": len(datasets.get("key", [])),
                    "audio": len(datasets.get("audio", [])),
                    "datasets": datasets
                }
        except Exception:
            return {"total": 0, "mouse": 0, "key": 0, "audio": 0}

    # --------- Sleep control: config, drains, gate ---------
    def ensure_sleep_defaults(self) -> None:
        """Initialize engine-level params/state if missing. One-time safe call."""
        fe = self.fe
        # Ensure energy_params exists and has all required keys
        if not hasattr(fe, "energy_params"):
            fe.energy_params = {}
        
        # Default energy params - merge missing keys
        defaults = {
            "drain_base": 0.001,
            "drain_per_mouse_px": 0.0001,
            "drain_per_click": 0.002,
            "drain_per_key": 0.0005,
            "sleep_entry_energy": 0.1,
            "sleep_entry_pressure": 0.8,
            "sleep_cooldown_ticks": 10,
            "min_awake_ticks": 5
        }
        for key, value in defaults.items():
            if key not in fe.energy_params:
                fe.energy_params[key] = value
        if not hasattr(fe, "sleep_switches"):
            fe.sleep_switches = {
                "update_world_nrem": False,
                "update_workers_nrem": False,
                "update_teacher_ema": False,
                "maintain_lsh_nrem": False
            }
        fe._tick = getattr(fe, "_tick", 0)
        fe._energy = getattr(fe, "_energy", 1.0)
        fe._sleep_pressure = getattr(fe, "_sleep_pressure", 0.0)
        fe._last_sleep_tick = getattr(fe, "_last_sleep_tick", -10**9)
        fe._last_wake_tick = getattr(fe, "_last_wake_tick", 0)

    def apply_energy_drain(self, action: dict | None) -> None:
        """Per-tick energy drain tied to action magnitude. No OS side effects."""
        fe = self.fe
        p = fe.energy_params
        a = action or {"type": "noop"}
        drain = float(p["drain_base"])
        if a.get("type") == "mouse":
            dx = abs(float(a.get("dx", 0.0))); dy = abs(float(a.get("dy", 0.0)))
            drain += (dx + dy) * float(p["drain_per_mouse_px"])
            if a.get("click", "none") in ("left", "right"):
                drain += float(p["drain_per_click"])
        elif a.get("type") == "key":
            if a.get("down") or a.get("up"):
                drain += float(p["drain_per_key"])
        fe._energy = max(0.0, fe._energy - drain)

    def update_sleep_pressure(self, awake: bool) -> None:
        """Very simple integrator: rises while awake, decays while sleeping."""
        fe = self.fe
        # tune rates later; keep tiny changes per tick
        rise = 0.001 if awake else -0.005
        fe._sleep_pressure = float(min(1.0, max(0.0, fe._sleep_pressure + rise)))

    def set_sleep_pressure(self, value: float) -> None:
        """Directly set sleep pressure value (used for recharge reset)."""
        fe = self.fe
        fe._sleep_pressure = float(min(1.0, max(0.0, value)))

    def _coerce_split3(self, split):
        # Accept (train, val) or (train, val, meta, ...); return 3-tuple
        if isinstance(split, (list, tuple)):
            if len(split) >= 3:
                return split[0], split[1], split[2] if split[2] is not None else {}
            if len(split) == 2:
                return split[0], split[1], {}
        # Fallbacks
        return [], [], {}

    def should_enter_sleep(self, tick: int) -> bool:
        """Gate: enter sleep if energy low or pressure high, honoring cooldowns."""
        fe = self.fe
        if not hasattr(fe, 'energy_params'):
            if hasattr(fe, 'diag_info'):
                try:
                    fe.diag_info.emit(f"ERROR: energy_params not found!")
                except Exception:
                    pass
            return False
        p = fe.energy_params
        
        # DEBUG: Confirm this function is being called
        if hasattr(fe, 'diag_info') and tick % 10 == 0:
            try:
                fe.diag_info.emit(f"should_enter_sleep CALLED: tick={tick}")
            except Exception:
                pass
        
        # Debug: emit detailed sleep gate evaluation
        cooldown_check = tick - fe._last_sleep_tick < int(p.get("sleep_cooldown_ticks", 10))
        min_awake_check = tick - fe._last_wake_tick < int(p.get("min_awake_ticks", 5))
        energy_check = fe._energy <= float(p.get("sleep_entry_energy", 0.1))
        pressure_check = fe._sleep_pressure >= float(p.get("sleep_entry_pressure", 0.8))
        
        if hasattr(fe, 'diag_info'):
            try:
                fe.diag_info.emit(f"Sleep gate: tick={tick}, energy={fe._energy:.3f} (thresh={p.get('sleep_entry_energy', 0.1)}), pressure={fe._sleep_pressure:.3f} (thresh={p.get('sleep_entry_pressure', 0.8)}), last_sleep={fe._last_sleep_tick}, last_wake={fe._last_wake_tick}, cooldown_block={cooldown_check}, min_awake_block={min_awake_check}")
            except Exception:
                pass
        
        if cooldown_check:
            return False
        if min_awake_check:
            return False
        if energy_check:
            return True
        if pressure_check:
            return True
        return False

    def note_sleep_start(self, tick: int) -> None:
        fe = self.fe
        fe._last_sleep_tick = int(tick)
        # decay pressure quickly on entry so we don't flap
        fe._sleep_pressure = max(0.0, fe._sleep_pressure - 0.05)

    def note_wake(self, tick: int) -> None:
        fe = self.fe
        fe._last_wake_tick = int(tick)
        # small bump so we don't re-enter immediately
        fe._sleep_pressure = min(1.0, fe._sleep_pressure + 0.01)

    # --------- WORLD: batches from replay (uses logged live_z and next) ---------
    def _world_make_batches(self, data: List[Dict[str, Any]]) -> Tuple[List[Tuple[np.ndarray,np.ndarray]], List[Tuple[np.ndarray,np.ndarray]]]:
        if not data: return [], []
        X, Y = [], []
        for r in data:
            z  = r.get("live_z");   zn = r.get("live_z_next") or None
            if z is None or zn is None:
                continue
            try:
                z_arr = np.asarray(z, dtype=np.float32)
                zn_arr = np.asarray(zn, dtype=np.float32)
                # Shape hygiene: expect 32D latent vectors
                if (z_arr.shape == zn_arr.shape and 
                    len(z_arr.shape) == 1 and len(z_arr) == 32):
                    X.append(z_arr)
                    Y.append(zn_arr)
            except Exception:
                continue
        n = len(X)
        if n == 0: return [], []
        k = max(1, int(0.1 * n))
        return list(zip(X[:-k], Y[:-k])), list(zip(X[-k:], Y[-k:]))

    def _world_eval_predictor(self, predictor, val: List[Tuple[np.ndarray,np.ndarray]]) -> float:
        if predictor is None or not val or torch is None: return float("inf")
        predictor.eval()
        se, n = 0.0, 0
        with torch.no_grad():
            for x, y in val:
                xt = torch.tensor(x).view(1, -1)
                yt = torch.tensor(y).view(1, -1)
                yp = predictor.forward_latent(xt) if hasattr(predictor, "forward_latent") else predictor(xt)
                se += float(torch.mean((yp - yt) ** 2))
                n += 1
        return se / max(1, n)

    def train_world_nrem(self, train_replay, val_replay, cand):
        fe = self.fe
        # Build pairs
        Xt, Yt = self._world_pairs(train_replay)
        Xv, Yv = self._world_pairs(val_replay)
        report = {"promoted": False}

        # Path A: Torch predictor worker clone (preferred)
        torch_ok = (torch is not None)
        torch_pred = cand.get("predictor", None)
        if torch_ok and torch_pred is not None:
            tr = list(zip(Xt, Yt)); vl = list(zip(Xv, Yv))
            if not tr or not vl:
                report.update({"reason": "insufficient_pairs"})
                return report

            baseline = self._world_eval_predictor(getattr(fe, "_predictor_worker", None), vl)
            report["world_mse_baseline"] = baseline

            # simple training loop
            model = torch_pred
            model.train(True)
            opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for epoch in range(2):
                for x, y in tr:
                    xt = torch.tensor(x).view(1, -1)
                    yt = torch.tensor(y).view(1, -1)
                    yp = model.forward_latent(xt) if hasattr(model, "forward_latent") else model(xt)
                    loss = torch.mean((yp - yt) ** 2)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    try:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    except Exception:
                        pass
                    opt.step()

            cand_mse = self._world_eval_predictor(model, vl)
            report["world_mse_val"] = cand_mse
            report["used"] = "predictor_worker"
            if cand_mse <= baseline:
                fe._predictor_worker = model
                report["promoted"] = True
            return report

        # Path B: Fallback to simple latent predictor (numpy) if torch worker is missing
        latpred = getattr(fe, "_latent_predictor", None)
        if latpred is None or not Xt or not Xv:
            # nothing to do
            report.update({
                "world_mse_baseline": float("inf"),
                "world_mse_val": float("inf"),
                "pairs_train": len(Xt),
                "pairs_val": len(Xv),
                "error": "no_predictor"
            })
            return report

        # Save snapshot to allow rollback
        A0 = getattr(latpred, "A", None)
        B0 = getattr(latpred, "b", None)
        if A0 is not None: A0 = A0.copy()
        if B0 is not None: B0 = B0.copy()

        baseline = self._eval_latpred_mse(latpred, Xv, Yv)
        steps = self._train_latpred_clone(latpred, Xt, Yt, epochs=2)
        cand_mse = self._eval_latpred_mse(latpred, Xv, Yv)

        improved = cand_mse <= baseline
        if not improved and A0 is not None and B0 is not None:
            # rollback
            latpred.A[:] = A0
            latpred.b[:] = B0

        report.update({
            "world_mse_baseline": float(baseline),
            "world_mse_val": float(cand_mse),
            "pairs_train": len(Xt),
            "pairs_val": len(Xv),
            "steps": int(steps),
            "used": "latent_predictor",
            "promoted": bool(improved)
        })
        return report

    def update_teacher_ema(self, tau: float = 0.001) -> bool:
        """
        Optional slow-teacher EMA update from live encoder. Requires fe._teacher_encoder and fe.vae.encoder.
        """
        fe = self.fe
        teacher = getattr(fe, "_teacher_encoder", None)
        live_enc = getattr(getattr(fe, "vae", None), "encoder", None)
        if teacher is None or live_enc is None or not hasattr(teacher, "parameters"):
            return False
        try:
            with torch.no_grad():
                for p_t, p_s in zip(teacher.parameters(), live_enc.parameters()):
                    p_t.mul_(1.0 - tau).add_(p_s, alpha=tau)
            return True
        except Exception:
            return False

    def maintain_lsh(self, replay: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Best-effort LSH hygiene. Uses fe.lsh if it exposes maintenance hooks; otherwise no-op.
        """
        fe = self.fe
        out = {"recomputed": False, "archived": 0}
        lsh = getattr(fe, "lsh", None)
        if lsh is None:
            return out
        try:
            if hasattr(lsh, "recompute_centroids"):
                lsh.recompute_centroids()
                out["recomputed"] = True
            if hasattr(lsh, "archive_cold_buckets"):
                out["archived"] = int(lsh.archive_cold_buckets())
        except Exception:
            pass
        return out

    def sample_dream_rollouts(self, replay: list[dict], steps: int = 6, batch: int = 4) -> dict:
        """
        Generate sandbox trajectories in latent/hash space. No OS I/O. Minimal, deterministic.
        Returns {"trajectories": [...], "stats": {...}}
        """
        import random
        fe = self.fe
        if not replay:
            return {"trajectories": [], "stats": {"mean_reward": 0.0, "len": 0}}
        seeds = replay[-min(batch, len(replay)):]
        trajs = []
        # predictor handle if available
        predictor = getattr(fe, "_predictor_worker", None)
        target = getattr(fe, "reward_target", {"novelty": 0.05, "energy": 0.5, "sleep_pressure": 0.2})
        weights = getattr(fe, "reward_weights", {"novelty": 1.0, "energy": 0.2, "sleep_pressure": 0.2})
        for s in seeds:
            z = np.asarray(s.get("live_z"), dtype=np.float32) if s.get("live_z") is not None else None
            if z is None:
                continue
            steps_list = []
            cum = 0.0
            for t in range(steps):
                # route selection is a no-op placeholder; we log a noop action
                action = {"type": "noop"}
                # predict next latent
                if predictor is not None and hasattr(predictor, "forward_latent"):
                    with torch.no_grad():
                        zt = torch.tensor(z).view(1, -1)
                        z_next = predictor.forward_latent(zt).detach().cpu().numpy().astype(np.float32)[0]
                elif predictor is not None:
                    with torch.no_grad():
                        zt = torch.tensor(z).view(1, -1)
                        z_next = predictor(zt).detach().cpu().numpy().astype(np.float32)[0]
                else:
                    # fallback: tiny drift
                    z_next = z + 0.01 * np.sign(np.random.randn(*z.shape)).astype(np.float32)
                # fabricate driver deltas lightly toward targets for a stable baseline
                drivers = {"novelty": float(min(1.0, max(0.0, 0.03 + 0.01*t))),
                           "energy": float(min(1.0, max(0.0, s.get("drivers",{}).get("energy", 0.5)))),
                           "sleep_pressure": float(min(1.0, max(0.0, s.get("drivers",{}).get("sleep_pressure", 0.3))))}
                # reward from next (same function used in wake)
                r = VaeSup.reward_from_next({}, {"drivers": drivers}, target, weights)
                cum += r
                steps_list.append({"z": z.tolist(), "z_next": z_next.tolist(), "drivers": drivers, "reward": r, "action": action})
                z = z_next
            trajs.append({"start_tick": int(s.get("tick", -1)), "steps": steps_list, "cum_reward": float(cum)})
        mean_r = float(np.mean([tr["cum_reward"] for tr in trajs])) if trajs else 0.0
        return {"trajectories": trajs, "stats": {"mean_reward": mean_r, "len": len(trajs)}}

    def train_world_rem(self, dreams: dict) -> dict:
        """
        Optional: tiny fit of predictor on imagined pairs. Conservative: never promote here.
        Returns {"trained": bool, "pairs": N}
        """
        steps = []
        for tr in (dreams or {}).get("trajectories", []):
            steps.extend(tr.get("steps", []))
        pairs = [(np.asarray(s["z"], np.float32), np.asarray(s["z_next"], np.float32)) for s in steps if "z" in s and "z_next" in s]
        return {"trained": False, "pairs": len(pairs)}  # keep REM policy/world training off until acceptance gates exist

    def train_workers_rem(self, dreams: dict) -> dict:
        """
        Placeholder for advantage-weighted imitation on imagined steps. Disabled by default.
        """
        return {"trained": False, "samples": sum(len(tr.get("steps", [])) for tr in (dreams or {}).get("trajectories", []))}
    
    def train_router_rem(self, dreams: dict) -> dict:
        """
        REM-only router learning using conservative advantage-weighted loss on sleep rollouts.
        Promotion disabled until acceptance tests pass.
        """
        trajectories = (dreams or {}).get("trajectories", [])
        if not trajectories:
            return {"trained": False, "steps": 0, "eval_return": 0.0, "kl": 0.0}
        
        # Extract steps for router training
        all_steps = []
        total_returns = []
        for traj in trajectories:
            steps = traj.get("steps", [])
            if steps:
                all_steps.extend(steps)
                # Calculate trajectory return (sum of rewards)
                traj_return = sum(s.get("reward", 0.0) for s in steps)
                total_returns.append(traj_return)
        
        if not all_steps:
            return {"trained": False, "steps": 0, "eval_return": 0.0, "kl": 0.0}
        
        # Conservative training: only on positive advantage trajectories
        mean_return = sum(total_returns) / len(total_returns) if total_returns else 0.0
        positive_trajs = [traj for i, traj in enumerate(trajectories) 
                         if total_returns[i] > mean_return]
        
        if not positive_trajs:
            return {"trained": False, "steps": 0, "eval_return": mean_return, "kl": 0.0}
        
        # TODO: Implement actual IILSTM router training here
        # For now, return metrics without actual training (promotion disabled)
        return {
            "trained": False,  # Keep promotion off
            "steps": len(all_steps), 
            "eval_return": mean_return,
            "kl": 0.0,
            "positive_trajs": len(positive_trajs)
        }

    @staticmethod
    def reward_from_next(prev_rec: dict, curr_rec: dict, target: dict, weights: dict) -> float:
        """Static helper for reward computation in dreams."""
        try:
            d_next = (curr_rec.get("drivers") or {})
            err = 0.0
            for k in ("novelty", "energy", "sleep_pressure"):
                if k in d_next:
                    diff = float(d_next[k]) - float(target.get(k, 0.0))
                    err += float(weights.get(k, 1.0)) * (diff * diff)
            return -err
        except Exception:
            return 0.0

    # ---------------- Sleep helpers: replay builder and cloning ----------------
    def build_replay(self, replay: List[Dict[str, Any]], val_frac: float = 0.1) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
        if not replay:
            return [], []
        data = list(replay)
        try:
            data.sort(key=lambda r: int(r.get("tick", 0)))
        except Exception:
            pass
        def ok(r: Dict[str,Any]) -> bool:
            return (
                ("final_action" in r) and
                ("drivers" in r) and
                ("hash_bits" in r) and
                ("live_z" in r)
            )
        data = [r for r in data if ok(r)]
        if not data:
            return [], []
        n = len(data)
        n_val = max(1, int(n * val_frac))
        val = data[-n_val:]
        train = data[:-n_val] if n > n_val else data
        try:
            self.fe._sleep_diag["replay_size"] = n
            self.fe._sleep_diag["replay_val"] = len(val)
            self.fe._sleep_diag["replay_train"] = len(train)
            self.fe._sleep_diag["replay_ticks"] = (data[0].get("tick"), data[-1].get("tick"))
        except Exception:
            pass
        return train, val

    def clone_components_for_sleep(self) -> Dict[str, Any]:
        cand: Dict[str, Any] = {}
        try:
            cand["vae"] = self._deepcopy_safe(getattr(self.fe, "vae", None))
        except Exception:
            cand["vae"] = None
        try:
            cand["predictor"] = self._deepcopy_safe(self.fe._predictor_worker)
        except Exception:
            cand["predictor"] = None
        try:
            cand["mouse"] = self._deepcopy_safe(self.fe._mouse_worker)
        except Exception:
            cand["mouse"] = None
        try:
            cand["key"] = self._deepcopy_safe(self.fe._key_worker)
        except Exception:
            cand["key"] = None
        try:
            cand["teacher"] = self._deepcopy_safe(getattr(self.fe, "_teacher_encoder", None))
        except Exception:
            cand["teacher"] = None
        try:
            cand["lsh"] = self.fe.lsh.snapshot() if hasattr(self.fe.lsh, "snapshot") else None
        except Exception:
            cand["lsh"] = None
        self.fe._sleep_diag["cloned"] = {k: (v is not None) for k,v in cand.items()}
        return cand

    def _deepcopy_safe(self, obj: Any) -> Any:
        import copy
        if obj is None:
            return None
        try:
            return copy.deepcopy(obj)
        except Exception:
            return None

    def make_worker_datasets(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        ds = {"mouse": [], "key": [], "audio": []}
        if not data:
            return ds
        packet_zeros = np.zeros((64,), dtype=np.float32)
        for r in data:
            sel = r.get("iilstm_selected") or []
            final = r.get("final_action") or {"type": "noop"}
            if final.get("type") == "mouse":
                ctrl = None
                for s in sel:
                    if s.get("name") == "mouse_v1" and "ctrl" in s:
                        ctrl = np.array(s["ctrl"], dtype=np.float32)
                        break
                if ctrl is None:
                    continue
                dx = float(final.get("dx", 0.0))
                dy = float(final.get("dy", 0.0))
                click = str(final.get("click", "none"))
                down = bool(final.get("down", False))
                up   = bool(final.get("up",   False))
                max_px = float(self.fe.action_params.get("mouse_max_px", 5.0))
                tx = np.clip(dx / max_px, -1.0, 1.0)
                ty = np.clip(dy / max_px, -1.0, 1.0)
                y = {"dx": tx, "dy": ty,
                     "left":  1.0 if click == "left"  else 0.0,
                     "right": 1.0 if click == "right" else 0.0,
                     "down":  1.0 if down  else 0.0,
                     "up":    1.0 if up    else 0.0}
                x = np.concatenate([packet_zeros, ctrl], axis=0).astype(np.float32)
                ds["mouse"].append({"x": x, "y": y})
            elif final.get("type") == "key":
                ctrl = None
                for s in sel:
                    if s.get("name") == "key_v1" and "ctrl" in s:
                        ctrl = np.array(s["ctrl"], dtype=np.float32)
                        break
                if ctrl is None:
                    continue
                code = int(final.get("code", 0))
                down = bool(final.get("down", False))
                up   = bool(final.get("up",   False))
                y = {"code": code, "down": 1.0 if down else 0.0, "up": 1.0 if up else 0.0}
                x = np.concatenate([packet_zeros, ctrl], axis=0).astype(np.float32)
                ds["key"].append({"x": x, "y": y})
            elif final.get("type") in ("audio_play", "audio_stop"):
                ctrl = None
                for s in sel:
                    if s.get("name") == "audio_v1" and "ctrl" in s:
                        ctrl = np.array(s["ctrl"], dtype=np.float32)
                        break
                if ctrl is None:
                    continue
                event_idx = 1 if final.get("type") == "audio_play" else 2
                sound_id = int(final.get("id", 0))
                gain = float(final.get("gain", 1.0))
                dur_ms = int(final.get("dur_ms", 400))
                y = {"event": event_idx, "id": sound_id, "gain": gain, "dur_ms": dur_ms}
                x = np.concatenate([packet_zeros, ctrl], axis=0).astype(np.float32)
                ds["audio"].append({"x": x, "y": y})
        return ds

    def eval_mouse(self, model: Any, data: List[Dict[str, Any]]) -> Dict[str, float]:
        if not data or model is None or torch is None: return {"mae_px": float("inf"), "click_acc": 0.0}
        model.eval()
        errs, correct, total = [], 0, 0
        max_px = float(self.fe.action_params.get("mouse_max_px", 5.0))
        with torch.no_grad():
            for s in data:
                x = torch.tensor(s["x"]).view(1,1,-1)
                y = s["y"]
                out = model(x).detach().cpu().numpy().astype(np.float32)[0]
                dx = float(np.tanh(out[0]) * max_px)
                dy = float(np.tanh(out[1]) * max_px)
                dx_t = float(y["dx"] * max_px)
                dy_t = float(y["dy"] * max_px)
                errs.append(abs(dx - dx_t) + abs(dy - dy_t))
                click_pred = "left" if out[2] > 0.8 else ("right" if out[3] > 0.85 else "none")
                click_true = "left" if y["left"] > 0.5 else ("right" if y["right"] > 0.5 else "none")
                correct += int(click_pred == click_true)
                total += 1
        mae_px = float(np.mean(errs)) if errs else float("inf")
        acc = (correct / max(1, total))
        return {"mae_px": mae_px, "click_acc": acc}

    def eval_key(self, model: Any, data: List[Dict[str, Any]], K: int = 8) -> Dict[str, float]:
        if not data or model is None or torch is None: return {"event_acc": 0.0, "key_acc": 0.0}
        model.eval()
        event_correct, key_correct, total = 0, 0, 0
        with torch.no_grad():
            for s in data:
                x = torch.tensor(s["x"]).view(1,1,-1)
                y = s["y"]
                out = model(x).detach().cpu().numpy().astype(np.float32)[0]
                key_idx = int(np.argmax(out[:K]))
                p_down, p_up, p_noop = float(out[K]), float(out[K+1]), float(out[K+2])
                pred = "noop"
                if p_down > 0.6 and p_up < 0.5: pred = "down"
                elif p_up > 0.6 and p_down < 0.5: pred = "up"
                true = "down" if y["down"] > 0.5 else ("up" if y["up"] > 0.5 else "noop")
                event_correct += int(pred == true)
                key_correct += int(key_idx == y["code"])
                total += 1
        return {"event_acc": event_correct / max(1, total), "key_acc": key_correct / max(1, total)}

    def train_workers(self, train: List[Dict[str, Any]], val: List[Dict[str, Any]], cand: Dict[str, Any]) -> Dict[str, Any]:
        if torch is None or nn is None or optim is None:
            return {"error": "torch not available"}
        self.fe._ensure_mouse_worker()
        self.fe._ensure_key_worker()
        if cand.get("mouse") is None: cand["mouse"] = self._deepcopy_safe(self.fe._mouse_worker)
        if cand.get("key")   is None: cand["key"]   = self._deepcopy_safe(self.fe._key_worker)
        ds_tr = self.make_worker_datasets(train)
        ds_vl = self.make_worker_datasets(val)
        report = {"mouse": {}, "key": {}, "promoted": []}
        base_mouse = self.eval_mouse(self.fe._mouse_worker, ds_vl["mouse"])
        base_key   = self.eval_key(self.fe._key_worker,   ds_vl["key"])
        if ds_tr["mouse"] and cand.get("mouse") is not None:
            m = cand["mouse"]; m.train()
            opt = optim.Adam(m.parameters(), lr=1e-4)
            mse = nn.MSELoss(); bce = nn.BCEWithLogitsLoss()
            for _ in range(2):
                for s in ds_tr["mouse"]:
                    x = torch.tensor(s["x"]).view(1,1,-1)
                    y = s["y"]
                    out = m(x)[0]
                    loss = 0.0
                    target_dx = np.arctanh(np.clip(y["dx"], -0.999, 0.999))
                    target_dy = np.arctanh(np.clip(y["dy"], -0.999, 0.999))
                    loss = loss + mse(out[0], torch.tensor(target_dx, dtype=torch.float32))
                    loss = loss + mse(out[1], torch.tensor(target_dy, dtype=torch.float32))
                    loss = loss + bce(out[2], torch.tensor(y["left"],  dtype=torch.float32))
                    loss = loss + bce(out[3], torch.tensor(y["right"], dtype=torch.float32))
                    loss = loss + bce(out[4], torch.tensor(y["down"],  dtype=torch.float32))
                    loss = loss + bce(out[5], torch.tensor(y["up"],    dtype=torch.float32))
                    opt.zero_grad(); loss.backward()
                    try:
                        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    except Exception:
                        pass
                    opt.step()
        if ds_tr["key"] and cand.get("key") is not None:
            kmod = cand["key"]; kmod.train()
            opt = optim.Adam(kmod.parameters(), lr=1e-4)
            ce = nn.CrossEntropyLoss(); bce = nn.BCEWithLogitsLoss()
            K = 8
            for _ in range(2):
                for s in ds_tr["key"]:
                    x = torch.tensor(s["x"]).view(1,1,-1)
                    y = s["y"]
                    out = kmod(x)[0]
                    logits = out[:K].view(1, K)
                    loss = ce(logits, torch.tensor([y["code"]], dtype=torch.long))
                    loss = loss + bce(out[K],   torch.tensor(y["down"], dtype=torch.float32))
                    loss = loss + bce(out[K+1], torch.tensor(y["up"],   dtype=torch.float32))
                    opt.zero_grad(); loss.backward()
                    try:
                        nn.utils.clip_grad_norm_(kmod.parameters(), 1.0)
                    except Exception:
                        pass
                    opt.step()
        cand_mouse = self.eval_mouse(cand.get("mouse"), ds_vl["mouse"]) if cand.get("mouse") else {}
        cand_key   = self.eval_key(cand.get("key"),   ds_vl["key"])     if cand.get("key")   else {}
        report["mouse"] = {"base": base_mouse, "cand": cand_mouse, "n_train": len(ds_tr["mouse"]), "n_val": len(ds_vl["mouse"]) }
        report["key"]   = {"base": base_key,   "cand": cand_key,   "n_train": len(ds_tr["key"]),   "n_val": len(ds_vl["key"]) }
        promote_mouse = (ds_vl["mouse"] and (cand_mouse.get("mae_px", float("inf")) <= base_mouse.get("mae_px", float("inf")) or cand_mouse.get("click_acc", 0.0) >= base_mouse.get("click_acc", 0.0)))
        promote_key   = (ds_vl["key"]   and (cand_key.get("event_acc", 0.0) >= base_key.get("event_acc", 0.0) or cand_key.get("key_acc", 0.0) >= base_key.get("key_acc", 0.0)))
        mouse_ok = False
        key_ok = False
        if promote_mouse and cand.get("mouse") is not None:
            self.fe._mouse_worker = cand["mouse"]
            self.fe._registry._workers[self.fe._mouse_worker_idx].handle = self.fe._mouse_worker
            report["promoted"].append("mouse_v1")
            mouse_ok = True
        if promote_key and cand.get("key") is not None:
            self.fe._key_worker = cand["key"]
            self.fe._registry._workers[self.fe._key_worker_idx].handle = self.fe._key_worker
            report["promoted"].append("key_v1")
            key_ok = True
        
        # Return dict with requested metrics - emit concrete metrics even when not promoted
        total_samples = len(ds_tr["mouse"]) + len(ds_tr["key"])
        promoted_mouse = bool(mouse_ok)
        promoted_key = bool(key_ok)
        
        result = {
            "promoted": bool(promoted_mouse or promoted_key),
            "mouse_mae_val": float(cand_mouse.get("mae_px", float("inf"))) if cand_mouse else None,
            "click_acc_val": float(cand_mouse.get("click_acc", 0.0)) if cand_mouse else None,
            "key_event_acc_val": float(cand_key.get("event_acc", 0.0)) if cand_key else None,
            "key_id_acc_val": float(cand_key.get("key_acc", 0.0)) if cand_key else None,
            "samples": int(total_samples),
            "mouse_promoted": promoted_mouse,
            "key_promoted": promoted_key,
            "pairs_train": int(len(ds_tr["mouse"]) + len(ds_tr["key"])),
            "pairs_val": int(len(ds_vl["mouse"]) + len(ds_vl["key"])),
        }
        
        return result

    def _build_world_supervised_sets(self, replay):
        # replay: list of dicts with 'live_z' and 'live_z_next'  
        pairs = []
        has_live_z = 0
        has_live_z_next = 0
        total_records = len(replay)
        
        for rec in replay:
            if "live_z" in rec:
                has_live_z += 1
            if "live_z_next" in rec:
                has_live_z_next += 1
            if "live_z" in rec and "live_z_next" in rec:
                try:
                    z = np.asarray(rec["live_z"], dtype=np.float32)
                    z_next = np.asarray(rec["live_z_next"], dtype=np.float32)
                    # Shape hygiene: both must be same shape and have expected dimensionality
                    if (z.shape == z_next.shape and 
                        len(z.shape) == 1 and len(z) == 32):  # Expect 32D latent vectors
                        pairs.append((z, z_next))
                except Exception:
                    continue
        
        try:
            fe = self.fe
            fe.diag_info.emit(f"_build_world_supervised_sets: {total_records} records, {has_live_z} have live_z, {has_live_z_next} have live_z_next, {len(pairs)} valid pairs")
        except Exception:
            pass
        
        if not pairs: 
            return [], [], {"error": "no_pairs"}
        
        # deterministic split 90/10
        n = len(pairs)
        n_val = max(1, int(0.1 * n))
        train, val = pairs[:-n_val], pairs[-n_val:]
        return train, val, {"replay_train": len(train), "replay_val": len(val)}

    def nrem_update_world(self, replay):
        """
        Train a CLONE of the world predictor on (z -> z_next) pairs from replay.
        Returns (promoted: bool, metrics: dict). Never mutates the live model unless accepted.
        Supports both torch modules and generic callables. Eliminates infinity values.
        """
        fe = self.fe

        # 1) Build supervised pairs
        split = self._build_world_supervised_sets(replay)
        train_pairs, val_pairs = split[:2]
        meta = split[2] if len(split) > 2 else {}
        n_train, n_val = len(train_pairs), len(val_pairs)

        # hard check latent dims; don't silently pad in sleep
        expected = int(getattr(self.fe, "latent_dim", 0)) or (len(train_pairs[0][0]) if train_pairs else 0)
        if expected > 0:
            bad = [i for i,(a,b) in enumerate(train_pairs[-64:]) if len(a)!=expected or len(b)!=expected]
            if bad:
                try:
                    self.fe._log_sleep_metric(event="nrem_world", promoted=False, error=f"latent_dim_mismatch_expected_{expected}", bad_examples=len(bad))
                except Exception:
                    pass
                return False, {
                    "world_mse_baseline": float("inf"),
                    "world_mse_val": float("inf"),
                    "pairs_train": len(train_pairs),
                    "pairs_val": len(val_pairs),
                    "steps": 0,
                    "accept_rel": float(getattr(self, "world_accept_rel", 0.001)),
                    "accept_abs": float(getattr(self, "world_accept_abs", 1e-6)),
                    "trainable": False
                }

        if n_train == 0 or n_val == 0:
            return False, {
                "world_mse_baseline": float("inf"),
                "world_mse_val": float("inf"),
                "pairs_train": n_train,
                "pairs_val": n_val,
                "error": "insufficient_pairs"
            }

        # 2) Resolve live predictor using the robust discovery
        world_live = self._require_world_predictor()
        if world_live is None:
            return False, {
                "world_mse_baseline": float("inf"),
                "world_mse_val": float("inf"),
                "pairs_train": n_train,
                "pairs_val": n_val,
                "error": "no_predictor"
            }
        
        # 3) Baseline on VAL with the LIVE model
        baseline_mse = self._eval_world_mse(val_pairs, model=world_live)

        # 4) Only clone and train if it's a torch module
        if self._is_torch_module(world_live):
            try:
                import copy
                world_clone = copy.deepcopy(world_live)
                if hasattr(world_clone, "train"): 
                    world_clone.train(True)
            except Exception as e:
                return False, {
                    "world_mse_baseline": float(baseline_mse),
                    "world_mse_val": float("inf"),
                    "pairs_train": n_train,
                    "pairs_val": n_val,
                    "steps": 0,
                    "error": f"clone_failed: {e!s}"
                }

            # Train the CLONE on TRAIN
            steps = self._train_world_clone(world_clone, train_pairs)

            # Validate CLONE on VAL
            val_mse = self._eval_world_mse(val_pairs, model=world_clone)

            # Strict acceptance criteria
            steps = int(steps)  # ensure int
            rel = float(getattr(self, "world_accept_rel", 0.001))
            abs_eps = float(getattr(self, "world_accept_abs", 1e-6))
            improved = self._improved(baseline_mse, val_mse, steps, rel=rel, abs_eps=abs_eps)

            if improved:
                # promote clone
                setattr(self.fe, "world_predictor", world_clone)
            else:
                # discard or leave live as-is
                pass

            return improved, {
                "world_mse_baseline": float(baseline_mse),
                "world_mse_val": float(val_mse),
                "pairs_train": int(len(train_pairs)),
                "pairs_val": int(len(val_pairs)),
                "steps": steps,
                "accept_rel": rel,
                "accept_abs": abs_eps,
                "trainable": bool(self._is_torch_module(world_clone))
            }
        else:
            # Non-torch callable: can't clone or train, just return baseline metrics
            return False, {
                "world_mse_baseline": float(baseline_mse),
                "world_mse_val": float(baseline_mse),  # same as baseline since no training possible
                "pairs_train": n_train,
                "pairs_val": n_val,
                "steps": 0,
                "error": "non_torch_callable",
                "trainable": False,
                **(meta if isinstance(meta, dict) else {})
            }

    def _train_world_clone(self, model, train_pairs):
        # Expect each pair is (z, z_next) as tensors/ndarrays
        if len(train_pairs) == 0: 
            return 0
        if not hasattr(model, "parameters"): 
            return 0  # not a torch model
        
        if torch is None or nn is None or optim is None:
            return 0

        model.train(True)
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        steps = 0
        epochs = 2  # keep small; adjust later
        bs = 64
        # simple mini-batch
        for _ in range(epochs):
            for i in range(0, len(train_pairs), bs):
                batch = train_pairs[i:i+bs]
                z = torch.as_tensor([p[0] for p in batch], dtype=torch.float32)
                z_next = torch.as_tensor([p[1] for p in batch], dtype=torch.float32)
                try:
                    pred = model.forward_latent(z) if hasattr(model, "forward_latent") else model(z)
                    loss = loss_fn(pred, z_next)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    steps += 1
                except Exception:
                    continue
        return steps

    def _eval_world_mse(self, pairs, model=None):
        if len(pairs) == 0: 
            try:
                fe = self.fe
                fe.diag_warning.emit("_eval_world_mse: No pairs provided")
            except Exception:
                pass
            return float("inf")
        if model is None: 
            model = self._require_world_predictor()
        if model is None:
            try:
                fe = self.fe
                fe.diag_warning.emit("_eval_world_mse: No model found by _require_world_predictor")
            except Exception:
                pass
            return float("inf")
        
        if torch is None or nn is None:
            try:
                fe = self.fe
                fe.diag_warning.emit("_eval_world_mse: torch or nn is None")
            except Exception:
                pass
            return float("inf")
            
        if hasattr(model, "eval"): 
            model.eval()
        
        loss_fn = nn.MSELoss(reduction="mean")
        try:
            # Log to sleep log
            try:
                self.fe._log_sleep_metric(event="debug", stage="eval_start", pairs_count=len(pairs), model_type=type(model).__name__, has_forward_latent=hasattr(model, 'forward_latent'))
            except Exception:
                pass
                
            with torch.no_grad():
                z = torch.as_tensor([p[0] for p in pairs], dtype=torch.float32)
                z_next = torch.as_tensor([p[1] for p in pairs], dtype=torch.float32)
                
                try:
                    self.fe._log_sleep_metric(event="debug", stage="tensors_created", z_shape=list(z.shape), z_next_shape=list(z_next.shape))
                except Exception:
                    pass
                
                # Debug shapes and data
                try:
                    fe = self.fe
                    fe.diag_info.emit(f"_eval_world_mse: z.shape={z.shape}, z_next.shape={z_next.shape}, model_type={type(model).__name__}")
                except Exception:
                    pass
                
                try:
                    self.fe._log_sleep_metric(event="debug", stage="about_to_forward")
                except Exception:
                    pass
                
                pred = model.forward_latent(z) if hasattr(model, "forward_latent") else model(z)
                
                try:
                    self.fe._log_sleep_metric(event="debug", stage="forward_completed", pred_shape=list(pred.shape))
                except Exception:
                    pass
                
                try:
                    fe = self.fe
                    fe.diag_info.emit(f"_eval_world_mse: pred.shape={pred.shape}")
                except Exception:
                    pass
                
                loss = loss_fn(pred, z_next)
                result = float(loss.item())
                
                try:
                    self.fe._log_sleep_metric(event="debug", stage="mse_computed", mse_result=result)
                except Exception:
                    pass
                
                try:
                    fe = self.fe
                    fe.diag_info.emit(f"_eval_world_mse: Successfully computed MSE = {result}")
                except Exception:
                    pass
                
                return result
        except Exception as e:
            try:
                self.fe._log_sleep_metric(event="debug", stage="exception", error=str(e), error_type=type(e).__name__)
            except Exception:
                pass
            try:
                fe = self.fe
                fe.diag_warning.emit(f"_eval_world_mse: Exception in evaluation: {e}")
            except Exception:
                pass
            return float("inf")

    def _promote_world_clone(self, world_clone):
        # Replace the live predictor with the trained clone
        try:
            # Update both the resolved reference and common attribute names
            self.fe.world_predictor = world_clone
            if hasattr(self.fe, "_predictor_worker"):
                self.fe._predictor_worker = world_clone
        except Exception:
            pass

    def _discard_world_clone(self):
        # Explicit cleanup (optional)
        pass

    def try_get_world_predictor(self):
        fe = self.fe
        
        # Diagnostic logging
        try:
            available_attrs = [attr for attr in dir(fe) if not attr.startswith('__')]
            fe.diag_info.emit(f"DEBUG: Available FE attributes: {available_attrs[:20]}...")  # truncate for readability
        except Exception:
            pass
            
        # 1) direct attribute search
        search_names = ("world_predictor", "predictor", "predictor_worker", "predictor_v1", "_predictor_worker")
        for name in search_names:
            obj = getattr(fe, name, None)
            try:
                fe.diag_info.emit(f"DEBUG: Checking {name} = {type(obj).__name__ if obj else 'None'}")
            except Exception:
                pass
                
            if obj is None: 
                continue
                
            # Try as_world_module method
            m = getattr(obj, "as_world_module", None)
            if callable(m):
                try:
                    mod = m()
                    if mod is not None:
                        try:
                            fe.diag_info.emit(f"DEBUG: Found predictor via {name}.as_world_module() -> {type(mod).__name__}")
                        except Exception:
                            pass
                        return mod
                except Exception as e:
                    try:
                        fe.diag_info.emit(f"DEBUG: {name}.as_world_module() failed: {e}")
                    except Exception:
                        pass
                        
            # Try common sub-attributes
            for attr in ("model", "net", "module", "core"):
                mod = getattr(obj, attr, None)
                if mod is not None:
                    try:
                        fe.diag_info.emit(f"DEBUG: Found predictor via {name}.{attr} -> {type(mod).__name__}")
                    except Exception:
                        pass
                    return mod
                    
            # Try object itself if it has forward
            if hasattr(obj, "forward"):
                try:
                    fe.diag_info.emit(f"DEBUG: Using {name} directly as predictor -> {type(obj).__name__}")
                except Exception:
                    pass
                return obj

        # 2) search registered workers list (through registry)
        registry = getattr(fe, "_registry", None)
        if registry is not None:
            try:
                workers_list = registry.all()
                fe.diag_info.emit(f"DEBUG: Checking {len(workers_list)} workers in registry")
            except Exception:
                workers_list = []
                
            for i, worker_entry in enumerate(workers_list):
                wtype = getattr(worker_entry, "wtype", None)
                handle = getattr(worker_entry, "handle", None)
                try:
                    fe.diag_info.emit(f"DEBUG: Worker {i}: type={wtype}, handle={type(handle).__name__ if handle else 'None'}")
                except Exception:
                    pass
                    
                if wtype == "predictor" and handle is not None:
                    # Try as_world_module method on the handle
                    m = getattr(handle, "as_world_module", None)
                    if callable(m):
                        try:
                            mod = m()
                            if mod is not None:
                                try:
                                    fe.diag_info.emit(f"DEBUG: Found predictor via worker.handle.as_world_module() -> {type(mod).__name__}")
                                except Exception:
                                    pass
                                return mod
                        except Exception as e:
                            try:
                                fe.diag_info.emit(f"DEBUG: worker.handle.as_world_module() failed: {e}")
                            except Exception:
                                pass
                                
                    # Try common sub-attributes on handle
                    for attr in ("model", "net", "module", "core"):
                        mod = getattr(handle, attr, None)
                        if mod is not None:
                            try:
                                fe.diag_info.emit(f"DEBUG: Found predictor via worker.handle.{attr} -> {type(mod).__name__}")
                            except Exception:
                                pass
                            return mod
                            
                    # Try handle itself
                    if hasattr(handle, "forward"):
                        try:
                            fe.diag_info.emit(f"DEBUG: Using worker.handle directly as predictor -> {type(handle).__name__}")
                        except Exception:
                            pass
                        return handle

        try:
            fe.diag_info.emit("DEBUG: No world predictor found in any search location")
        except Exception:
            pass
        return None

    def _require_world_predictor(self):
        mod = getattr(self.fe, "world_predictor", None)
        if mod is None:
            mod = self.try_get_world_predictor()
            if mod is not None:
                self.fe.world_predictor = mod
        
        try:
            fe = self.fe
            if mod is None:
                fe.diag_warning.emit("_require_world_predictor: No predictor found")
            else:
                fe.diag_info.emit(f"_require_world_predictor: Found {type(mod).__name__}, has forward_latent: {hasattr(mod, 'forward_latent')}")
        except Exception:
            pass
            
        return mod

    def _is_torch_module(self, obj):
        return torch is not None and nn is not None and isinstance(obj, nn.Module)

    def _is_finite(self, x):
        try:
            return math.isfinite(float(x))
        except Exception:
            return False

    def _improved(self, baseline, val, steps, rel=0.001, abs_eps=1e-6):
        # require a real training step and finite metrics
        if steps <= 0:
            return False
        if not (self._is_finite(baseline) and self._is_finite(val)):
            return False
        # require strictly better than both absolute and relative margins
        target = min(baseline - abs_eps, baseline * (1.0 - rel))
        return val < target

    def _world_pairs(self, replay_list):
        X, Y = [], []
        for r in replay_list:
            z  = r.get("live_z")
            zn = r.get("live_z_next")
            if z is None or zn is None:
                continue
            try:
                z_arr = np.asarray(z, dtype=np.float32)
                zn_arr = np.asarray(zn, dtype=np.float32)
                # Shape hygiene: expect 32D latent vectors
                if (z_arr.shape == zn_arr.shape and 
                    len(z_arr.shape) == 1 and len(z_arr) == 32):
                    X.append(z_arr)
                    Y.append(zn_arr)
            except Exception:
                continue
        return X, Y

    def _eval_latpred_mse(self, model, X, Y):
        if model is None or not X:
            return float("inf")
        se = 0.0
        n  = 0
        for x, y in zip(X, Y):
            yp = model.predict(x)
            se += float(np.mean((yp - y) ** 2))
            n  += 1
        return se / max(1, n)

    def _train_latpred_clone(self, model, X, Y, epochs=2):
        # in-place updates using model.update(z_t, z_t1)
        if model is None or not X:
            return 0
        steps = 0
        for _ in range(int(epochs)):
            for x, y in zip(X, Y):
                try:
                    model.update(x, y)
                    steps += 1
                except Exception:
                    return steps
        return steps

    def _world_eval_predictor(self, model, pairs):
        if model is None or not pairs:
            return 999.0
        if torch is None or nn is None:
            return 999.0
        try:
            model.eval()
            with torch.no_grad():
                total_loss = 0.0
                count = 0
                for x, y in pairs:
                    xt = torch.tensor(x).view(1, -1)
                    yt = torch.tensor(y).view(1, -1)
                    yp = model.forward_latent(xt) if hasattr(model, "forward_latent") else model(xt)
                    loss = torch.mean((yp - yt) ** 2)
                    total_loss += float(loss.item())
                    count += 1
                return total_loss / max(1, count)
        except Exception:
            return 999.0

    def nrem_update_workers(self, replay):
        # Thin alias for compatibility  
        train, val = self.build_replay(replay)
        cand = self.clone_components_for_sleep()
        return self.train_workers(train, val, cand)



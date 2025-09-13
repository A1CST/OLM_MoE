from typing import List, Dict, Any, Optional


class ActionBus:
    """
    Aggregates worker outputs into a single action set with safety:
      - One mouse + one key per tick max
      - Pixel clamp, click/key cooldowns, energy gate
      - Button-state tracking for down/up sanity
    """
    def __init__(self):
        self._last_left = False
        self._last_right = False
        self._key_cooldowns: Dict[int, int] = {}   # key_idx -> ticks remaining
        self._click_cooldown_ticks = 0
        self._audio_cooldown_ticks = 0             # audio cooldown
        self._tick = 0

    def _cooldown_step(self):
        self._tick += 1
        self._click_cooldown_ticks = max(0, self._click_cooldown_ticks - 1)
        self._audio_cooldown_ticks = max(0, self._audio_cooldown_ticks - 1)
        for k in list(self._key_cooldowns.keys()):
            self._key_cooldowns[k] = max(0, self._key_cooldowns[k] - 1)
            if self._key_cooldowns[k] == 0:
                del self._key_cooldowns[k]

    def aggregate(self, proposals: List[Dict[str, Any]], *, tick: int, energy: float, params: Dict[str, Any], executed_workers: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._cooldown_step()

        # Energy gate
        if energy <= params.get("energy_min", 0.02):
            final_action = {"type": "noop"}
        else:
            # Select at most one mouse, key, and audio proposal
            mouse = next((p for p in proposals if p.get("type") == "mouse"), None)
            key   = next((p for p in proposals if p.get("type") == "key"), None)
            audio = next((p for p in proposals if p.get("type") == "audio"), None)

            out_mouse = None
            if mouse is not None and params.get("mouse_enabled", True):
                s_px = float(params.get("mouse_max_px", 5.0))
                dx = float(max(-s_px, min(s_px, mouse.get("dx", 0.0))))
                dy = float(max(-s_px, min(s_px, mouse.get("dy", 0.0))))
                click = "none"
                # Debounce clicks
                if params.get("clicks_enabled", False) and self._click_cooldown_ticks == 0:
                    req = mouse.get("click", "none")
                    if req in ("left", "right"):
                        click = req
                        self._click_cooldown_ticks = int(params.get("click_cooldown_ticks", 20))
                out_mouse = {"type": "mouse", "dx": dx, "dy": dy, "click": click}

            out_key = None
            if key is not None and params.get("keys_enabled", True):
                key_idx = int(key.get("code", -1))
                if key_idx >= 0 and self._key_cooldowns.get(key_idx, 0) == 0:
                    down = bool(key.get("down", False))
                    up   = bool(key.get("up", False))
                    # Enforce at most one transition; noop if both false
                    if down or up:
                        out_key = {"type": "key", "code": key_idx, "down": down, "up": up}
                        self._key_cooldowns[key_idx] = int(params.get("key_cooldown_ticks", 10))

            out_audio = None
            if audio is not None and params.get("audio_enabled", True):
                if self._audio_cooldown_ticks == 0:
                    event = audio.get("event", "noop")
                    if event in ("play", "stop"):
                        sound_id = int(max(0, min(63, audio.get("id", 0))))  # Clamp to vocab_size
                        gain = float(max(0.0, min(1.0, audio.get("gain", 1.0))))  # Clamp 0-1
                        dur_ms = int(max(1, min(400, audio.get("dur_ms", 400))))  # Clamp duration
                        
                        if event == "play":
                            out_audio = {"type": "audio_play", "id": sound_id, "gain": gain, "dur_ms": dur_ms}
                        elif event == "stop":
                            out_audio = {"type": "audio_stop", "id": sound_id}
                        
                        self._audio_cooldown_ticks = int(params.get("audio_cooldown_ticks", 4))

            # Priority: audio > mouse > key (audio is least frequent, most impactful)
            if out_audio:
                final_action = out_audio
            elif out_mouse and out_key:
                final_action = {"type": "combo", "mouse": out_mouse, "key": out_key}
            elif out_mouse:
                final_action = out_mouse
            elif out_key:
                final_action = out_key
            else:
                final_action = {"type": "noop"}

        # If the bus decided to noop, do not mark any worker as executed
        if final_action.get("type") == "noop":
            executed_workers = []  # ensure no training samples get harvested
        else:
            executed_workers = executed_workers or []  # only bus-approved executions
            
        result = {
            "final_action": final_action,
            "executed_workers": executed_workers,
        }
        return result



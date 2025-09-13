import hashlib
import json
from typing import Optional, Tuple
import numpy as np


class LSHSystem:
    """Random-hyperplane LSH over vision latents with chained SHA-256.

    - Hyperplanes H shape (latent_dim, num_bits), columns unit-normalized
    - Code: bits_j = 1 if v·h_j >= 0 else 0
    - Chain: sha256(hex(prev_hash) || bitstring)
    - Persistence: save/load hyperplanes and last hash
    """

    def __init__(self, latent_dim: int, num_bits: int = 64, seed: int = 0):
        self.latent_dim = latent_dim
        self.num_bits = num_bits
        self._seed = seed
        self._init_hyperplanes()
        self.last_hash_hex: Optional[str] = None
        # Active/archived code management
        self._active_codes: set[str] = set()
        # centroid -> [codes]
        self._archive: dict[str, list[str]] = {}
        # code -> centroid (direct lookup for restore)
        self._archived_codes: dict[str, str] = {}

    def compute_code(self, v: np.ndarray) -> np.ndarray:
        assert v.shape[0] == self.latent_dim, "latent dim mismatch"
        projections = v @ self.hyperplanes  # (num_bits,)
        bits = (projections >= 0.0).astype(np.uint8)
        return bits

    @staticmethod
    def bits_to_str(bits: np.ndarray) -> str:
        return ''.join('1' if b else '0' for b in bits.tolist())

    @staticmethod
    def hamming_distance(bits_a: np.ndarray, bits_b: np.ndarray) -> int:
        assert bits_a.shape == bits_b.shape
        return int(np.sum(bits_a != bits_b))

    def get_vision_similarity(self, bits_a: np.ndarray, bits_b: np.ndarray) -> float:
        return self.hamming_distance(bits_a, bits_b) / float(self.num_bits)

    def compute_chained_hash(self, bits: np.ndarray, previous_hash_hex: Optional[str] = None) -> str:
        prev = previous_hash_hex or self.last_hash_hex or 'GENESIS'
        payload = json.dumps({
            'prev': prev,
            'bits': self.bits_to_str(bits),
        }, separators=(',', ':'), sort_keys=True)
        digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()
        self.last_hash_hex = digest
        return digest

    def hash_latent(self, v: np.ndarray, previous_hash_hex: Optional[str] = None) -> Tuple[np.ndarray, str]:
        bits = self.compute_code(v)
        h = self.compute_chained_hash(bits, previous_hash_hex)
        return bits, h

    def save_state(self, path: str):
        last = self.last_hash_hex if self.last_hash_hex is not None else ''
        active_json = json.dumps(sorted(list(self._active_codes)))
        archive_json = json.dumps(self._archive)
        archived_codes_json = json.dumps(self._archived_codes)
        np.savez(path,
                 hyperplanes=self.hyperplanes,
                 last_hash_hex=np.array(last),
                 active_codes_json=np.array(active_json),
                 archive_json=np.array(archive_json),
                 archived_codes_json=np.array(archived_codes_json))

    def load_state(self, path: str):
        data = np.load(path, allow_pickle=True)
        hp = data['hyperplanes']
        try:
            hp = hp.astype(np.float32)
        except Exception:
            hp = None
        if hp is None or hp.ndim != 2:
            # Reinitialize on corruption
            self._init_hyperplanes()
        else:
            self.hyperplanes = hp
            self.latent_dim = self.hyperplanes.shape[0]
            self.num_bits = self.hyperplanes.shape[1]
        last = data['last_hash_hex'] if 'last_hash_hex' in data else None
        if last is None:
            self.last_hash_hex = None
        else:
            try:
                self.last_hash_hex = str(last.item()) if hasattr(last, 'item') else str(last)
                if self.last_hash_hex == '':
                    self.last_hash_hex = None
            except Exception:
                self.last_hash_hex = None
        # Load active/archive if present
        try:
            act = data['active_codes_json'] if 'active_codes_json' in data else None
            arc = data['archive_json'] if 'archive_json' in data else None
            arc_map = data['archived_codes_json'] if 'archived_codes_json' in data else None
            if act is not None:
                s = str(act.item()) if hasattr(act, 'item') else str(act)
                self._active_codes = set(json.loads(s))
            if arc is not None:
                s = str(arc.item()) if hasattr(arc, 'item') else str(arc)
                self._archive = json.loads(s)
            if arc_map is not None:
                s = str(arc_map.item()) if hasattr(arc_map, 'item') else str(arc_map)
                self._archived_codes = json.loads(s)
            # Backfill code->centroid if missing
            if not getattr(self, '_archived_codes', None):
                self._archived_codes = {}
                for cent, lst in self._archive.items():
                    for c in lst:
                        self._archived_codes[c] = cent
        except Exception:
            self._active_codes = set()
            self._archive = {}
            self._archived_codes = {}

    def _init_hyperplanes(self):
        rng = np.random.default_rng(self._seed)
        H = rng.standard_normal((self.latent_dim, self.num_bits)).astype(np.float32)
        norms = np.linalg.norm(H, axis=0) + 1e-12
        self.hyperplanes = (H / norms)

    # ---------------- Active/Archive Management ----------------
    def prune_or_bucket(self, v: np.ndarray, threshold_bits: int = 3) -> tuple[str, str, Optional[str]]:
        """Given latent v, compute its code and decide:
        - If exact code exists in active: return (code, 'existing', code)
        - Else find nearest active by Hamming; if distance <= threshold_bits, archive under centroid and return (code, 'bucketed', centroid)
        - Else add code to active: return (code, 'new', code)
        """
        bits = self.compute_code(v)
        code = self.bits_to_str(bits)
        if code in self._active_codes:
            return code, 'existing', code
        # Find nearest active
        nearest = None
        best_d = self.num_bits + 1
        for cent in self._active_codes:
            d = self._hamming_str(code, cent)
            if d < best_d:
                best_d = d
                nearest = cent
                if best_d == 0:
                    break
        if nearest is not None and best_d <= threshold_bits:
            # Archive under nearest centroid
            bucket = self._archive.setdefault(nearest, [])
            if code not in bucket:
                bucket.append(code)
                self._archived_codes[code] = nearest
            return code, 'bucketed', nearest
        # Otherwise add as new active centroid
        self._active_codes.add(code)
        return code, 'new', code

    def restore(self, code: str) -> tuple[bool, Optional[str]]:
        """Restore a previously archived code to active set.
        Returns (success, centroid_used).
        """
        cent = self._archived_codes.get(code)
        if cent is None:
            # fallback scan for legacy archives
            for k, lst in list(self._archive.items()):
                if code in lst:
                    cent = k
                    break
        if cent is None:
            return False, None
        lst = self._archive.get(cent, [])
        if code in lst:
            lst.remove(code)
            if len(lst) == 0:
                self._archive.pop(cent, None)
        self._archived_codes.pop(code, None)
        self._active_codes.add(code)
        return True, cent

    def get_active_count(self) -> int:
        return len(self._active_codes)

    def get_archived_count(self) -> int:
        return sum(len(v) for v in self._archive.values())

    def get_archive_counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self._archive.items()}

    def _hamming_str(self, a: str, b: str) -> int:
        la = len(a)
        lb = len(b)
        if la != lb:
            # compare up to min length plus penalty for remainder
            m = min(la, lb)
            return sum(1 for i in range(m) if a[i] != b[i]) + abs(la - lb)
        return sum(1 for i in range(la) if a[i] != b[i])



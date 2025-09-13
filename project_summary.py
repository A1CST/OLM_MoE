import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, List


class ProjectSummaryLogger:
    """Reusable project summary logger.

    Writes a single JSON file at state/project_summary.json with structure:
    {
      "schema_version": 1,
      "created_at": "...",
      "entries": [ {event}, ... ]
    }
    """

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        self.state_dir = os.path.join(base_dir, 'state')
        try:
            os.makedirs(self.state_dir, exist_ok=True)
        except Exception:
            pass
        self.summary_path = os.path.join(self.state_dir, 'project_summary.json')
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.summary_path):
            doc = {
                "schema_version": 1,
                "created_at": datetime.utcnow().isoformat() + 'Z',
                "entries": []
            }
            self._write_json(doc)

    def _read_json(self) -> Dict[str, Any]:
        try:
            with open(self.summary_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            # Reset file if corrupted
            doc = {
                "schema_version": 1,
                "created_at": datetime.utcnow().isoformat() + 'Z',
                "entries": []
            }
            self._write_json(doc)
            return doc

    def _write_json(self, doc: Dict[str, Any]):
        tmp_path = self.summary_path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(doc, f, ensure_ascii=False, separators=(',', ':'), indent=2)
        os.replace(tmp_path, self.summary_path)

    def add_event(
        self,
        category: str,
        title: str,
        description: str = '',
        outcome: Optional[str] = None,  # "good" | "bad" | None
        tags: Optional[List[str]] = None,
        files_changed: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        doc = self._read_json()
        event = {
            "time": datetime.utcnow().isoformat() + 'Z',
            "category": category,
            "title": title,
            "description": description,
        }
        if outcome is not None:
            event["outcome"] = outcome
        if tags:
            event["tags"] = list(tags)
        if files_changed:
            event["files_changed"] = list(files_changed)
        if meta:
            event["meta"] = meta
        doc.setdefault("entries", []).append(event)
        self._write_json(doc)

    def path(self) -> str:
        return self.summary_path


# Singleton-style convenience
_LOGGER = ProjectSummaryLogger()


def log_event(
    category: str,
    title: str,
    description: str = '',
    outcome: Optional[str] = None,
    tags: Optional[List[str]] = None,
    files_changed: Optional[List[str]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    _LOGGER.add_event(category, title, description, outcome, tags, files_changed, meta)


def get_summary_path() -> str:
    return _LOGGER.path()




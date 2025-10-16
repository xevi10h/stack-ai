import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class JSONPersister:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _json_serializer(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    def save_state(self, state: Dict[str, Any], filename: str) -> None:
        filepath = self.data_dir / filename

        with open(filepath, "w") as f:
            json.dump(state, f, default=self._json_serializer, indent=2)

    def load_state(self, filename: str) -> Dict[str, Any]:
        filepath = self.data_dir / filename

        if not filepath.exists():
            return {}

        with open(filepath, "r") as f:
            return json.load(f)

    def delete_state(self, filename: str) -> None:
        filepath = self.data_dir / filename

        if filepath.exists():
            os.remove(filepath)

    def state_exists(self, filename: str) -> bool:
        filepath = self.data_dir / filename
        return filepath.exists()

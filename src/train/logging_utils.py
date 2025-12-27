import json
import os
import time
from typing import Any, Dict


class ResultLogger:
    def __init__(self, log_dir: str, run_name: str):
        self.run_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.run_dir, "metrics.json")
        self._metrics_file = open(self.metrics_path, "a", encoding="utf-8")
        self.start_time = time.time()

    def log_config(self, config_payload: Dict[str, Any]) -> None:
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2, sort_keys=True)

    def log_episode(self, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload["wall_time_sec"] = round(time.time() - self.start_time, 3)
        self._metrics_file.write(json.dumps(payload) + "\n")
        self._metrics_file.flush()

    def close(self) -> None:
        if self._metrics_file:
            self._metrics_file.close()

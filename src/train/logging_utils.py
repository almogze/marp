import json
import os
import time
from typing import Any, Dict, Tuple

from torch.utils.tensorboard import SummaryWriter


class ResultLogger:
    def __init__(self, log_dir: str, run_name: str):
        self.run_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.run_dir, "metrics.json")
        self._metrics_file = open(self.metrics_path, "a", encoding="utf-8")
        self.start_time = time.time()
        self._writer = SummaryWriter(os.path.join(self.run_dir, "tensorboard"))

    def log_config(self, config_payload: Dict[str, Any]) -> None:
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2, sort_keys=True)

    def log_episode(self, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload["wall_time_sec"] = round(time.time() - self.start_time, 3)
        self._metrics_file.write(json.dumps(payload) + "\n")
        self._metrics_file.flush()
        self._log_tensorboard(payload)

    def _log_tensorboard(self, payload: Dict[str, Any]) -> None:
        step = int(payload.get("episode", 0))
        scalar_pairs: list[Tuple[str, float]] = []
        if "steps" in payload:
            scalar_pairs.append(("train/steps", float(payload["steps"])))
        if "reward_sum" in payload:
            scalar_pairs.append(("train/reward_sum", float(payload["reward_sum"])))
        if "reward_mean" in payload:
            scalar_pairs.append(("train/reward_mean", float(payload["reward_mean"])))
        for tag, value in scalar_pairs:
            self._writer.add_scalar(tag, value, step)

        reward_per_agent = payload.get("reward_per_agent")
        if isinstance(reward_per_agent, dict):
            for agent_id, reward in reward_per_agent.items():
                self._writer.add_scalar(f"reward/agent_{agent_id}", float(reward), step)

        social_metrics = payload.get("social_metrics")
        if isinstance(social_metrics, (list, tuple)) and len(social_metrics) == 4:
            names = ("efficiency", "equality", "sustainability", "peace")
            for name, value in zip(names, social_metrics):
                self._writer.add_scalar(f"social/{name}", float(value), step)
        elif isinstance(social_metrics, dict):
            for name, value in social_metrics.items():
                if isinstance(value, (int, float)):
                    self._writer.add_scalar(f"social/{name}", float(value), step)

        algo_metrics = payload.get("algo_metrics")
        if isinstance(algo_metrics, dict):
            if "avg_loss" in algo_metrics and isinstance(algo_metrics["avg_loss"], (int, float)):
                self._writer.add_scalar("train/loss", float(algo_metrics["avg_loss"]), step)
            if "loss" in algo_metrics and isinstance(algo_metrics["loss"], (int, float)):
                self._writer.add_scalar("train/loss", float(algo_metrics["loss"]), step)
            for name, value in algo_metrics.items():
                if isinstance(value, (int, float)):
                    self._writer.add_scalar(f"algo/{name}", float(value), step)

    def close(self) -> None:
        if self._metrics_file:
            self._metrics_file.close()
        if self._writer:
            self._writer.close()

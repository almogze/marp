import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EnvConfig:
    map_type: str = "small"
    num_agents: int = 1
    agent_view_range: int = 5
    ep_length: int = 600
    render: bool = False
    spawn_speed: str = "slow"
    metric: str = "Efficiency"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "EnvConfig":
        return EnvConfig(**data)


@dataclass
class DQNConfig:
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.995
    batch_size: int = 32
    replay_buffer_size: int = 5000
    train_after: int = 100
    train_every: int = 1
    target_update_freq: int = 200
    normalize_obs: bool = True
    device: str = "auto"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "DQNConfig":
        return DQNConfig(**data)


@dataclass
class PPOConfig:
    policy: str = "MultiInputPolicy"
    total_timesteps: int = 100_000
    per_agent_timesteps: int = 100_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    n_steps: int = 1024
    batch_size: int = 256
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)
    flatten_obs: bool = True
    multi_agent_mode: str = "independent"
    opponent_policy: str = "random"
    device: str = "auto"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PPOConfig":
        return PPOConfig(**data)


@dataclass
class AlgorithmConfig:
    name: str = "dqn"
    dqn: DQNConfig = field(default_factory=DQNConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AlgorithmConfig":
        dqn = DQNConfig.from_dict(data.get("dqn", {}))
        ppo = PPOConfig.from_dict(data.get("ppo", {}))
        return AlgorithmConfig(
            name=data.get("name", "dqn"),
            dqn=dqn,
            ppo=ppo,
        )


@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    run_name: Optional[str] = None
    log_interval: int = 1
    video_enabled: bool = True
    video_every_n_episodes: int = 100
    video_max_steps: int = 600
    video_fps: int = 10
    video_keep_frames: bool = False

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "LoggingConfig":
        return LoggingConfig(**data)


@dataclass
class TrainerConfig:
    episodes: int = 100
    steps_per_episode: int = 600
    seed: int = 0
    env: EnvConfig = field(default_factory=EnvConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TrainerConfig":
        return TrainerConfig(
            episodes=data.get("episodes", 100),
            steps_per_episode=data.get("steps_per_episode", 600),
            seed=data.get("seed", 0),
            env=EnvConfig.from_dict(data.get("env", {})),
            algorithm=AlgorithmConfig.from_dict(data.get("algorithm", {})),
            logging=LoggingConfig.from_dict(data.get("logging", {})),
        )


def load_config(path: str) -> TrainerConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return TrainerConfig.from_dict(data)


def save_config(path: str, config: TrainerConfig) -> None:
    payload = {
        "episodes": config.episodes,
        "steps_per_episode": config.steps_per_episode,
        "seed": config.seed,
        "env": config.env.__dict__,
        "algorithm": {
            "name": config.algorithm.name,
            "dqn": config.algorithm.dqn.__dict__,
            "ppo": config.algorithm.ppo.__dict__,
        },
        "logging": config.logging.__dict__,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

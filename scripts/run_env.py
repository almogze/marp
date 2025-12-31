import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import Trainer, load_config

DEFAULT_CONFIGS = {
    "dqn": "configs/train_dqn.json",
    "ppo": "configs/train_ppo.json",
    "mappo": "configs/train_mappo.json",
    "random": "configs/train_dqn.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MARP training with CLI overrides.")
    parser.add_argument(
        "--algo",
        default="dqn",
        choices=sorted(DEFAULT_CONFIGS.keys()),
        help="Algorithm to run.",
    )
    parser.add_argument("--episodes", type=int, help="Number of training episodes.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--map", dest="map_type", help="Map type (e.g., small).")
    parser.add_argument("--agents", type=int, help="Number of agents.")
    rm_group = parser.add_mutually_exclusive_group()
    rm_group.add_argument(
        "--reward-model",
        action="store_true",
        dest="reward_model",
        help="Enable reward modeling.",
    )
    rm_group.add_argument(
        "--no-reward-model",
        action="store_true",
        dest="no_reward_model",
        help="Disable reward modeling.",
    )
    parser.add_argument("--mode", help="Reward model mode (e.g., narrow_view).")
    parser.add_argument("--phi", help="Reward model objective (e.g., efficiency_x_peace).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = DEFAULT_CONFIGS[args.algo]
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    config = load_config(config_path)

    config.algorithm.name = args.algo
    if args.episodes is not None:
        config.episodes = int(args.episodes)
    if args.seed is not None:
        config.seed = int(args.seed)
    if args.map_type is not None:
        config.env.map_type = args.map_type
    if args.agents is not None:
        config.env.num_agents = int(args.agents)

    reward_model_enabled = None
    if args.reward_model:
        reward_model_enabled = True
    elif args.no_reward_model:
        reward_model_enabled = False
    if reward_model_enabled is not None:
        config.reward_model.enabled = reward_model_enabled

    if args.mode is not None:
        config.reward_model.mode = args.mode
    if args.phi is not None:
        config.reward_model.phi = args.phi

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

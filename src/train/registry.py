from .config import AlgorithmConfig


def build_algorithm(config: AlgorithmConfig):
    name = config.name
    if name == "dqn":
        from .algorithms.dqn import DQNAlgorithm

        return DQNAlgorithm(config.dqn)
    if name == "ppo":
        from .algorithms.ppo import PPOAlgorithm

        return PPOAlgorithm(config.ppo)
    if name == "mappo":
        from .algorithms.mappo import MAPPOAlgorithm

        return MAPPOAlgorithm(config.mappo)
    if name == "random":
        from .algorithms.random_policy import RandomAlgorithm

        return RandomAlgorithm(config)
    raise ValueError(f"Unknown algorithm '{name}'. Available: ['dqn', 'ppo', 'mappo', 'random']")

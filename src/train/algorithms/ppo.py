import os
from typing import Any, Dict, Optional

from .base import Algorithm
from ..env_wrappers import RewardModelSingleAgentGymWrapper, SingleAgentGymWrapper
from src.reward_model.preference_buffer import PreferenceBuffer
from src.reward_model.reward_model import RewardModel
from src.reward_model.reward_trainer import RewardModelTrainer


class PPOAlgorithm(Algorithm):
    def __init__(self, config: Any):
        super().__init__(config)
        self._env = None

    def uses_external_loop(self) -> bool:
        return False

    def on_env_ready(self, env) -> None:
        self._env = env

    def train(self, env, logger, config, video_recorder=None) -> None:
        import torch
        from gymnasium.wrappers import RecordEpisodeStatistics
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback

        class EpisodeLoggerCallback(BaseCallback):
            def __init__(self, result_logger, agent_id_getter, recorder, agent_label=None):
                super().__init__()
                self.result_logger = result_logger
                self.agent_id_getter = agent_id_getter
                self.recorder = recorder
                self.agent_label = agent_label
                self.episode_idx = 0
                self.episode_step = 0
                if self.recorder is not None:
                    self.recorder.start(self.episode_idx)

            def _on_step(self) -> bool:
                if self.recorder is not None:
                    env = self.training_env.envs[0].unwrapped._env
                    self.recorder.record(env, self.episode_step)
                    self.episode_step += 1
                infos = self.locals.get("infos", [])
                if not infos:
                    return True
                info = infos[0]
                if "episode" in info:
                    agent_id = self.agent_id_getter()
                    episode_info = info["episode"]
                    social_metrics = info.get("social_metrics")
                    pred_reward_sum = float(episode_info.get("r", 0.0))
                    env_reward_sum = info.get("env_reward_sum")
                    payload = {
                        "episode": self.episode_idx,
                        "steps": int(episode_info.get("l", 0)),
                        "reward_sum": pred_reward_sum,
                        "reward_mean": pred_reward_sum,
                        "reward_per_agent": {agent_id: pred_reward_sum},
                        "social_metrics": social_metrics,
                        "algo_metrics": {"policy": "ppo", "agent_id": self.agent_label or agent_id},
                    }
                    if env_reward_sum is not None:
                        payload["reward_env_sum"] = float(env_reward_sum)
                        payload["reward_env_mean"] = float(env_reward_sum)
                        payload["reward_env_per_agent"] = {agent_id: float(env_reward_sum)}
                    pred_reward_sum_info = info.get("pred_reward_sum")
                    if pred_reward_sum_info is not None:
                        payload["reward_pred_sum"] = float(pred_reward_sum_info)
                        payload["reward_pred_mean"] = float(pred_reward_sum_info)
                        payload["reward_pred_per_agent"] = {agent_id: float(pred_reward_sum_info)}
                    rm_metrics = info.get("reward_model")
                    if isinstance(rm_metrics, dict) and rm_metrics:
                        payload["algo_metrics"]["reward_model"] = rm_metrics
                    self.result_logger.log_episode(payload)
                    if self.recorder is not None:
                        self.recorder.finish()
                    self.episode_idx += 1
                    self.episode_step = 0
                    if self.recorder is not None:
                        self.recorder.start(self.episode_idx)
                return True

        policy_kwargs = dict(self.config.policy_kwargs or {})
        extractor_name = policy_kwargs.pop("features_extractor", None)
        if extractor_name:
            from stable_baselines3.common.torch_layers import FlattenExtractor, NatureCNN, CombinedExtractor

            mapping = {
                "FlattenExtractor": FlattenExtractor,
                "NatureCNN": NatureCNN,
                "CombinedExtractor": CombinedExtractor,
            }
            if extractor_name not in mapping:
                raise ValueError(f"Unknown features_extractor '{extractor_name}'.")
            policy_kwargs["features_extractor_class"] = mapping[extractor_name]

        max_steps = config.env.ep_length
        agent_ids = list(env.agents.keys())
        if env.num_agents == 1 or self.config.multi_agent_mode != "independent":
            agent_ids = [agent_ids[0]]

        for agent_id in agent_ids:
            device = self.config.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            rm_cfg = config.reward_model
            if rm_cfg.enabled:
                obs_shape = env.observation_space["curr_obs"].shape
                num_actions = int(env.action_space.n)
                reward_model = RewardModel(obs_shape=obs_shape, num_actions=num_actions)
                rm_trainer = RewardModelTrainer(reward_model, lr=rm_cfg.lr, device=rm_cfg.device)
                pref_buffer = PreferenceBuffer(rm_cfg.max_episodes_in_buffer)
                wrapped_env = RewardModelSingleAgentGymWrapper(
                    env,
                    reward_model=rm_trainer.reward_model,
                    rm_trainer=rm_trainer,
                    pref_buffer=pref_buffer,
                    rm_config=rm_cfg,
                    run_dir=logger.run_dir,
                    normalize_obs=config.algorithm.dqn.normalize_obs,
                    max_steps=max_steps,
                    transpose_obs=not self.config.flatten_obs,
                    flatten_obs=self.config.flatten_obs,
                    agent_id=agent_id,
                    opponent_policy=self.config.opponent_policy,
                )
            else:
                wrapped_env = SingleAgentGymWrapper(
                    env,
                    max_steps=max_steps,
                    transpose_obs=not self.config.flatten_obs,
                    flatten_obs=self.config.flatten_obs,
                    agent_id=agent_id,
                    opponent_policy=self.config.opponent_policy,
                )
            wrapped_env = RecordEpisodeStatistics(wrapped_env)

            model = PPO(
                policy=self.config.policy,
                env=wrapped_env,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                gae_lambda=self.config.gae_lambda,
                clip_range=self.config.clip_range,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                policy_kwargs=policy_kwargs or None,
                device=device,
                verbose=0,
            )

            callback = EpisodeLoggerCallback(
                logger,
                lambda: wrapped_env.unwrapped.agent_id,
                video_recorder,
                agent_label=agent_id,
            )
            timesteps = self.config.total_timesteps
            if env.num_agents > 1 and self.config.multi_agent_mode == "independent":
                timesteps = self.config.per_agent_timesteps
            model.learn(total_timesteps=timesteps, callback=callback)
            model_path = os.path.join(logger.run_dir, f"ppo_model_{agent_id}_last")
            model.save(model_path)
            if rm_cfg.enabled:
                reward_model_path = os.path.join(logger.run_dir, f"reward_model_{agent_id}_last.pt")
                reward_model.save(reward_model_path)
        if video_recorder is not None:
            video_recorder.finalize()

    def act(self, observations: Dict[str, Any], step: int) -> Dict[str, int]:
        raise RuntimeError("PPO does not use the external training loop.")

    def observe(
        self,
        observations: Dict[str, Any],
        actions: Dict[str, int],
        rewards: Dict[str, float],
        next_observations: Dict[str, Any],
        dones: Dict[str, bool],
        infos: Dict[str, Any],
        step: int,
    ) -> None:
        raise RuntimeError("PPO does not use the external training loop.")

    def on_episode_end(self, episode: int) -> Dict[str, Any]:
        raise RuntimeError("PPO does not use the external training loop.")

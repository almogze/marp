import os
import time
import numpy as np

from src.env.commons_env import HarvestCommonsEnv, MAP
from .config import TrainerConfig, save_config
from .logging_utils import ResultLogger
from .registry import build_algorithm
from .video_utils import VideoRecorder


class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.env = self._build_env()
        self.algorithm = build_algorithm(config.algorithm)
        self.algorithm.on_env_ready(self.env)
        self.logger = self._build_logger()

    def _build_env(self) -> HarvestCommonsEnv:
        env_cfg = self.config.env
        ascii_map = MAP[env_cfg.map_type]
        return HarvestCommonsEnv(
            ascii_map=ascii_map,
            num_agents=env_cfg.num_agents,
            render=env_cfg.render,
            agent_view_range=env_cfg.agent_view_range,
            ep_length=env_cfg.ep_length,
            spawn_speed=env_cfg.spawn_speed,
            metric=env_cfg.metric,
        )

    def _build_logger(self) -> ResultLogger:
        log_cfg = self.config.logging
        algo = self.config.algorithm.name
        run_name = log_cfg.run_name
        if not run_name:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            run_name = f"{timestamp}-{algo}-map={self.config.env.map_type}-agents={self.config.env.num_agents}"
        logger = ResultLogger(log_cfg.log_dir, run_name)
        config_path = os.path.join(logger.run_dir, "config.json")
        save_config(config_path, self.config)
        return logger

    def _build_video_recorder(self) -> VideoRecorder:
        log_cfg = self.config.logging
        videos_dir = os.path.join(self.logger.run_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        return VideoRecorder(
            base_dir=videos_dir,
            enabled=log_cfg.video_enabled,
            every_n_episodes=log_cfg.video_every_n_episodes,
            max_steps=log_cfg.video_max_steps,
            fps=log_cfg.video_fps,
            keep_frames=log_cfg.video_keep_frames,
        )

    def train(self) -> None:
        if not self.algorithm.uses_external_loop():
            video_recorder = self._build_video_recorder()
            self.algorithm.train(self.env, self.logger, self.config, video_recorder)
            video_recorder.finalize()
            self.logger.close()
            return

        video_recorder = self._build_video_recorder()
        for episode in range(self.config.episodes):
            obs, infos = self.env.reset(seed=self.config.seed)
            episode_rewards = {agent_id: 0.0 for agent_id in obs.keys()}
            step_count = 0
            video_recorder.start(episode)
            for step in range(self.config.steps_per_episode):
                actions = self.algorithm.act(obs, step)
                next_obs, rewards, dones, infos = self.env.step(actions)
                self.algorithm.observe(obs, actions, rewards, next_obs, dones, infos, step)
                video_recorder.record(self.env, step)
                for agent_id, reward in rewards.items():
                    episode_rewards[agent_id] += reward
                obs = next_obs
                step_count = step + 1
                if dones.get("__all__", False):
                    break

            self.env.compute_social_metrics()
            metrics = self.env.get_social_metrics()
            algo_metrics = self.algorithm.on_episode_end(episode)
            payload = {
                "episode": episode,
                "steps": step_count,
                "reward_sum": float(np.sum(list(episode_rewards.values()))),
                "reward_mean": float(np.mean(list(episode_rewards.values()))),
                "reward_per_agent": episode_rewards,
                "social_metrics": metrics,
                "algo_metrics": algo_metrics,
            }
            if episode % self.config.logging.log_interval == 0:
                self.logger.log_episode(payload)
            video_recorder.finish()

        video_recorder.finalize()
        self.logger.close()

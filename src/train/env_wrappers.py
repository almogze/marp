from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import os

from src.reward_model.preference_buffer import EpisodeRecord, PreferenceBuffer
from src.reward_model.reward_model import RewardModel
from src.reward_model.reward_trainer import RewardModelTrainer


class SingleAgentGymWrapper(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        env,
        max_steps: Optional[int] = None,
        transpose_obs: bool = False,
        flatten_obs: bool = False,
        agent_id: Optional[str] = None,
        opponent_policy: str = "random",
    ):
        super().__init__()
        self._env = env
        self._max_steps = max_steps
        self._step_count = 0
        self._agent_id = None
        self._transpose_obs = transpose_obs
        self._flatten_obs = flatten_obs
        self._fixed_agent_id = agent_id
        self._opponent_policy = opponent_policy
        self.action_space = env.action_space
        obs_space = env.observation_space
        if isinstance(obs_space, dict):
            img_space = obs_space["curr_obs"]
            if self._flatten_obs:
                flat_shape = (int(img_space.shape[0] * img_space.shape[1] * img_space.shape[2]),)
                self.observation_space = gym.spaces.Box(
                    low=img_space.low.min(),
                    high=img_space.high.max(),
                    shape=flat_shape,
                    dtype=img_space.dtype,
                )
            elif self._transpose_obs:
                transposed = gym.spaces.Box(
                    low=img_space.low.min(),
                    high=img_space.high.max(),
                    shape=(img_space.shape[2], img_space.shape[0], img_space.shape[1]),
                    dtype=img_space.dtype,
                )
                self.observation_space = gym.spaces.Dict({"curr_obs": transposed})
            else:
                self.observation_space = gym.spaces.Dict(obs_space)
        else:
            self.observation_space = obs_space

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs_dict, info_dict = self._env.reset(seed=seed)
        if not obs_dict:
            raise RuntimeError("Environment returned empty observations.")
        if self._fixed_agent_id is not None:
            if self._fixed_agent_id not in obs_dict:
                raise RuntimeError(f"Agent '{self._fixed_agent_id}' not found in observations.")
            self._agent_id = self._fixed_agent_id
        else:
            self._agent_id = next(iter(obs_dict.keys()))
        self._step_count = 0
        obs = obs_dict[self._agent_id]
        if self._flatten_obs:
            obs = obs["curr_obs"].reshape(-1)
        elif self._transpose_obs:
            obs = {"curr_obs": obs["curr_obs"].transpose(2, 0, 1)}
        return obs, info_dict.get(self._agent_id, {})

    def step(self, action: int):
        actions = {self._agent_id: action}
        for other_id in self._env.agents.keys():
            if other_id == self._agent_id:
                continue
            if self._opponent_policy == "zero":
                actions[other_id] = 0
            else:
                actions[other_id] = self._env.action_space.sample()
        obs_dict, reward_dict, done_dict, info_dict = self._env.step(actions)
        self._step_count += 1

        terminated = bool(done_dict.get(self._agent_id, False) or done_dict.get("__all__", False))
        truncated = False
        if self._max_steps is not None and self._step_count >= self._max_steps:
            truncated = True

        info = info_dict.get(self._agent_id, {})
        if terminated or truncated:
            self._env.compute_social_metrics()
            info = dict(info)
            info["social_metrics"] = self._env.get_social_metrics()

        obs = obs_dict[self._agent_id]
        if self._flatten_obs:
            obs = obs["curr_obs"].reshape(-1)
        elif self._transpose_obs:
            obs = {"curr_obs": obs["curr_obs"].transpose(2, 0, 1)}
        return (
            obs,
            float(reward_dict[self._agent_id]),
            terminated,
            truncated,
            info,
        )

    def render(self, mode: str = "human"):
        return self._env.render(mod=mode)

    def close(self):
        return self._env.close()


class RewardModelSingleAgentGymWrapper(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        env,
        reward_model: RewardModel,
        rm_trainer: Optional[RewardModelTrainer] = None,
        pref_buffer: Optional[PreferenceBuffer] = None,
        rm_config: Optional[Any] = None,
        run_dir: Optional[str] = None,
        normalize_obs: bool = False,
        max_steps: Optional[int] = None,
        transpose_obs: bool = False,
        flatten_obs: bool = False,
        agent_id: Optional[str] = None,
        opponent_policy: str = "random",
    ):
        super().__init__()
        self._env = env
        self._reward_model = reward_model
        self._rm_trainer = rm_trainer
        self._pref_buffer = pref_buffer
        self._rm_config = rm_config
        self._run_dir = run_dir
        self._normalize_obs = normalize_obs
        self._max_steps = max_steps
        self._step_count = 0
        self._global_step = 0
        self._last_rm_update_step = 0
        self._episode_idx = 0
        self._agent_id = None
        self._transpose_obs = transpose_obs
        self._flatten_obs = flatten_obs
        self._fixed_agent_id = agent_id
        self._opponent_policy = opponent_policy
        self._last_obs_dict = None
        self._episode_traj = None
        self._episode_env_reward_sum = 0.0
        self._episode_pred_reward_sum = 0.0
        self._last_rm_metrics = {}
        self.action_space = env.action_space
        obs_space = env.observation_space
        if isinstance(obs_space, dict):
            img_space = obs_space["curr_obs"]
            if self._flatten_obs:
                flat_shape = (int(img_space.shape[0] * img_space.shape[1] * img_space.shape[2]),)
                self.observation_space = gym.spaces.Box(
                    low=img_space.low.min(),
                    high=img_space.high.max(),
                    shape=flat_shape,
                    dtype=img_space.dtype,
                )
            elif self._transpose_obs:
                transposed = gym.spaces.Box(
                    low=img_space.low.min(),
                    high=img_space.high.max(),
                    shape=(img_space.shape[2], img_space.shape[0], img_space.shape[1]),
                    dtype=img_space.dtype,
                )
                self.observation_space = gym.spaces.Dict({"curr_obs": transposed})
            else:
                self.observation_space = gym.spaces.Dict(obs_space)
        else:
            self.observation_space = obs_space

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def _format_reward_obs(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        img = obs_dict[self._agent_id]["curr_obs"]
        if self._normalize_obs:
            return (img / 255.0).astype(np.float32)
        return img.astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs_dict, info_dict = self._env.reset(seed=seed)
        if not obs_dict:
            raise RuntimeError("Environment returned empty observations.")
        if self._fixed_agent_id is not None:
            if self._fixed_agent_id not in obs_dict:
                raise RuntimeError(f"Agent '{self._fixed_agent_id}' not found in observations.")
            self._agent_id = self._fixed_agent_id
        else:
            self._agent_id = next(iter(obs_dict.keys()))
        self._step_count = 0
        self._last_obs_dict = obs_dict
        self._episode_traj = []
        self._episode_env_reward_sum = 0.0
        self._episode_pred_reward_sum = 0.0
        self._last_rm_metrics = {}
        obs = obs_dict[self._agent_id]
        if self._flatten_obs:
            obs = obs["curr_obs"].reshape(-1)
        elif self._transpose_obs:
            obs = {"curr_obs": obs["curr_obs"].transpose(2, 0, 1)}
        return obs, info_dict.get(self._agent_id, {})

    def step(self, action: int):
        if self._last_obs_dict is None:
            raise RuntimeError("Must call reset() before step().")
        obs_img = self._format_reward_obs(self._last_obs_dict)
        if self._episode_traj is not None:
            self._episode_traj.append((obs_img, int(action)))
        pred_reward = float(self._reward_model.predict(obs_img, int(action)))

        actions = {self._agent_id: int(action)}
        for other_id in self._env.agents.keys():
            if other_id == self._agent_id:
                continue
            if self._opponent_policy == "zero":
                actions[other_id] = 0
            else:
                actions[other_id] = self._env.action_space.sample()
        obs_dict, reward_dict, done_dict, info_dict = self._env.step(actions)
        self._step_count += 1
        self._global_step += 1
        self._episode_env_reward_sum += float(reward_dict[self._agent_id])
        self._episode_pred_reward_sum += float(pred_reward)

        terminated = bool(done_dict.get(self._agent_id, False) or done_dict.get("__all__", False))
        truncated = False
        if self._max_steps is not None and self._step_count >= self._max_steps:
            truncated = True

        info = info_dict.get(self._agent_id, {})
        if terminated or truncated:
            self._env.compute_social_metrics()
            info = dict(info)
            info["social_metrics"] = self._env.get_social_metrics()
            info["env_reward_sum"] = float(self._episode_env_reward_sum)
            info["pred_reward_sum"] = float(self._episode_pred_reward_sum)
            if self._pref_buffer is not None and self._episode_traj is not None:
                record = EpisodeRecord(
                    agent_trajs={self._agent_id: self._episode_traj},
                    metrics=info["social_metrics"],
                )
                self._pref_buffer.add_episode(record)
            if (
                self._rm_trainer is not None
                and self._rm_config is not None
                and self._pref_buffer is not None
            ):
                if (self._episode_idx + 1) >= int(self._rm_config.warmup_episodes):
                    if (self._global_step - self._last_rm_update_step) >= int(
                        self._rm_config.update_every_env_steps
                    ):
                        self._last_rm_metrics = dict(
                            self._rm_trainer.train(
                                self._pref_buffer,
                                phi_key=self._rm_config.phi,
                                mode=self._rm_config.mode,
                                batch_pairs=int(self._rm_config.batch_pairs),
                                train_steps=int(self._rm_config.train_steps_per_update),
                            )
                        )
                        self._last_rm_update_step = self._global_step
                if self._run_dir and (self._episode_idx + 1) % int(
                    self._rm_config.save_every_episodes
                ) == 0:
                    save_path = os.path.join(self._run_dir, f"reward_model_{self._agent_id}.pt")
                    self._reward_model.save(save_path)
            info["reward_model"] = dict(self._last_rm_metrics) if self._last_rm_metrics else {}
            self._episode_idx += 1

        self._last_obs_dict = obs_dict
        obs = obs_dict[self._agent_id]
        if self._flatten_obs:
            obs = obs["curr_obs"].reshape(-1)
        elif self._transpose_obs:
            obs = {"curr_obs": obs["curr_obs"].transpose(2, 0, 1)}
        return (
            obs,
            float(pred_reward),
            terminated,
            truncated,
            info,
        )

    def render(self, mode: str = "human"):
        return self._env.render(mod=mode)

    def close(self):
        return self._env.close()

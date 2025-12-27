from typing import Any, Dict, Optional, Tuple

import gymnasium as gym


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

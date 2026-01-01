import os
import random
from collections import deque
from typing import Any, Dict, Tuple, List

import numpy as np
import torch
from torch import nn

from .base import Algorithm


class ReplayBuffer:
    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)

    def add(self, transition: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        if len(self.buffer) < batch_size:
            return []
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self) -> int:
        return len(self.buffer)


class DQNNetwork(nn.Module):
    def __init__(self, obs_shape: Tuple[int, int, int], num_actions: int) -> None:
        super().__init__()
        height, width, channels = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            conv_out = self.conv(dummy)
            conv_out_size = int(np.prod(conv_out.shape[1:]))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 4:
            raise ValueError(f"Expected obs with shape (B,H,W,C), got {obs.shape}")
        x = obs.permute(0, 3, 1, 2)
        x = self.conv(x)
        return self.head(x)


class DQNAlgorithm(Algorithm):
    def __init__(self, config: Any):
        super().__init__(config)
        self.agent_ids: List[str] = []
        self.obs_shape: Tuple[int, int, int] = (0, 0, 0)
        self.num_actions = 0
        self.device = torch.device("cpu")
        self.model: nn.Module
        self.target_model: nn.Module
        self.optimizer: torch.optim.Optimizer
        self.replay: ReplayBuffer
        self.epsilon: float = 1.0
        self.train_steps = 0
        self.last_loss: float = 0.0

    def on_env_ready(self, env) -> None:
        obs_space = env.observation_space["curr_obs"]
        self.obs_shape = obs_space.shape
        self.num_actions = int(env.action_space.n)
        self.agent_ids = list(env.agents.keys())

        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        print(f"DQN: Using device: {self.device}")
        if torch.cuda.is_available() and device == "cuda":
            print(f"DQN: GPU device: {torch.cuda.get_device_name(0)}")

        self.model = DQNNetwork(self.obs_shape, self.num_actions).to(self.device)
        self.target_model = DQNNetwork(self.obs_shape, self.num_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.replay = ReplayBuffer(self.config.replay_buffer_size)
        self.epsilon = self.config.epsilon_start

    def _format_obs(self, obs: Dict[str, Any], agent_id: str) -> np.ndarray:
        img = obs[agent_id]["curr_obs"]
        if self.config.normalize_obs:
            return (img / 255.0).astype(np.float32)
        return img.astype(np.float32)

    def act(self, observations: Dict[str, Any], step: int) -> Dict[str, int]:
        actions = {}
        for agent_id in self.agent_ids:
            obs = self._format_obs(observations, agent_id)
            if random.random() < self.epsilon:
                actions[agent_id] = random.randrange(self.num_actions)
            else:
                obs_tensor = torch.from_numpy(np.expand_dims(obs, axis=0)).float().to(self.device)
                with torch.no_grad():
                    q_values = self.model(obs_tensor)
                actions[agent_id] = int(torch.argmax(q_values[0]).item())
        return actions

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
        # Store all agents' transitions in shared replay buffer
        for agent_id in self.agent_ids:
            obs = self._format_obs(observations, agent_id)
            next_obs = self._format_obs(next_observations, agent_id)
            done = bool(dones.get(agent_id, False) or dones.get("__all__", False))
            self.replay.add((obs, actions[agent_id], rewards[agent_id], next_obs, done))

        # Train shared network periodically
        if len(self.replay) >= self.config.train_after and len(self.replay) >= self.config.batch_size:
            if self.train_steps % self.config.train_every == 0:
                self._train_step()
            self.train_steps += 1

    def _train_step(self) -> None:
        batch = self.replay.sample(self.config.batch_size)
        if not batch:
            return

        obs_batch = np.stack([b[0] for b in batch], axis=0)
        action_batch = np.array([b[1] for b in batch], dtype=np.int32)
        reward_batch = np.array([b[2] for b in batch], dtype=np.float32)
        next_obs_batch = np.stack([b[3] for b in batch], axis=0)
        done_batch = np.array([b[4] for b in batch], dtype=np.float32)

        obs_tensor = torch.from_numpy(obs_batch).float().to(self.device)
        next_obs_tensor = torch.from_numpy(next_obs_batch).float().to(self.device)
        action_tensor = torch.from_numpy(action_batch).long().to(self.device)
        reward_tensor = torch.from_numpy(reward_batch).float().to(self.device)
        done_tensor = torch.from_numpy(done_batch).float().to(self.device)

        with torch.no_grad():
            next_q = self.target_model(next_obs_tensor)
            max_next_q = next_q.max(dim=1).values
            target_q = reward_tensor + (1.0 - done_tensor) * self.config.gamma * max_next_q

        q_values = self.model(obs_tensor)
        action_q = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        loss_fn = nn.HuberLoss()
        loss = loss_fn(action_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.train_steps > 0 and self.train_steps % self.config.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        self.last_loss = float(loss.item())

    def on_episode_end(self, episode: int) -> Dict[str, Any]:
        # Decay epsilon
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
        metrics = {"epsilon": self.epsilon}
        if self.last_loss > 0:
            metrics["loss"] = self.last_loss
            self.last_loss = 0.0
        return metrics

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "model_state": self.model.state_dict(),
            "target_model_state": self.target_model.state_dict(),
            "epsilon": self.epsilon,
            "obs_shape": self.obs_shape,
            "num_actions": self.num_actions,
            "agent_ids": self.agent_ids,
            "config": self.config.__dict__,
        }
        torch.save(payload, path)

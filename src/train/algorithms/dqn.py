import random
from collections import deque
from typing import Any, Dict, Tuple, List

import numpy as np
import tensorflow as tf

from .base import Algorithm


class ReplayBuffer:
    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)

    def add(self, transition: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        num_actions: int,
        learning_rate: float,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        batch_size: int,
        replay_buffer_size: int,
        target_update_freq: int,
        train_after: int,
        train_every: int,
    ):
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_after = train_after
        self.train_every = train_every
        self.replay = ReplayBuffer(replay_buffer_size)
        self.train_steps = 0

        self.loss_fn = tf.keras.losses.Huber()
        self.model = self._build_model(learning_rate)
        self.target_model = self._build_model(learning_rate)
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self, learning_rate: float) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=self.obs_shape)
        x = tf.keras.layers.Conv2D(32, 3, activation="relu")(inputs)
        x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        outputs = tf.keras.layers.Dense(self.num_actions)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=self.loss_fn)
        return model

    def act(self, obs: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        q_values = self.model(np.expand_dims(obs, axis=0), training=False)
        return int(tf.argmax(q_values[0]).numpy())

    def remember(self, obs, action, reward, next_obs, done) -> None:
        self.replay.add((obs, action, reward, next_obs, done))

    def train_step(self) -> Dict[str, float]:
        if len(self.replay) < self.train_after or len(self.replay) < self.batch_size:
            return {}
        if self.train_steps % self.train_every != 0:
            self.train_steps += 1
            return {}

        batch = self.replay.sample(self.batch_size)
        obs_batch = np.stack([b[0] for b in batch], axis=0)
        action_batch = np.array([b[1] for b in batch], dtype=np.int32)
        reward_batch = np.array([b[2] for b in batch], dtype=np.float32)
        next_obs_batch = np.stack([b[3] for b in batch], axis=0)
        done_batch = np.array([b[4] for b in batch], dtype=np.float32)

        next_q = self.target_model(next_obs_batch, training=False)
        max_next_q = tf.reduce_max(next_q, axis=1).numpy()
        target_q = reward_batch + (1.0 - done_batch) * self.gamma * max_next_q

        with tf.GradientTape() as tape:
            q_values = self.model(obs_batch, training=True)
            action_q = tf.reduce_sum(
                q_values * tf.one_hot(action_batch, self.num_actions), axis=1
            )
            loss = self.loss_fn(target_q, action_q)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        if self.train_steps > 0 and self.train_steps % self.target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        self.train_steps += 1
        return {"loss": float(loss.numpy())}

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


class DQNAlgorithm(Algorithm):
    def __init__(self, config: Any):
        super().__init__(config)
        self.agents: Dict[str, DQNAgent] = {}
        self.last_losses: Dict[str, float] = {}

    def on_env_ready(self, env) -> None:
        obs_space = env.observation_space["curr_obs"]
        obs_shape = obs_space.shape
        num_actions = int(env.action_space.n)
        for agent_id in env.agents.keys():
            self.agents[agent_id] = DQNAgent(
                obs_shape=obs_shape,
                num_actions=num_actions,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                epsilon_start=self.config.epsilon_start,
                epsilon_end=self.config.epsilon_end,
                epsilon_decay=self.config.epsilon_decay,
                batch_size=self.config.batch_size,
                replay_buffer_size=self.config.replay_buffer_size,
                target_update_freq=self.config.target_update_freq,
                train_after=self.config.train_after,
                train_every=self.config.train_every,
            )

    def _format_obs(self, obs: Dict[str, Any], agent_id: str) -> np.ndarray:
        img = obs[agent_id]["curr_obs"]
        if self.config.normalize_obs:
            return (img / 255.0).astype(np.float32)
        return img.astype(np.float32)

    def act(self, observations: Dict[str, Any], step: int) -> Dict[str, int]:
        actions = {}
        for agent_id, agent in self.agents.items():
            obs = self._format_obs(observations, agent_id)
            actions[agent_id] = agent.act(obs, training=True)
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
        for agent_id, agent in self.agents.items():
            obs = self._format_obs(observations, agent_id)
            next_obs = self._format_obs(next_observations, agent_id)
            done = bool(dones.get(agent_id, False))
            agent.remember(obs, actions[agent_id], rewards[agent_id], next_obs, done)
            loss_info = agent.train_step()
            if loss_info.get("loss") is not None:
                self.last_losses[agent_id] = loss_info["loss"]

    def on_episode_end(self, episode: int) -> Dict[str, Any]:
        for agent in self.agents.values():
            agent.decay_epsilon()
        avg_loss = None
        if self.last_losses:
            avg_loss = float(np.mean(list(self.last_losses.values())))
        self.last_losses = {}
        return {"avg_loss": avg_loss, "epsilon": self._mean_epsilon()}

    def _mean_epsilon(self) -> float:
        if not self.agents:
            return 0.0
        eps = [agent.epsilon for agent in self.agents.values()]
        return float(np.mean(eps))

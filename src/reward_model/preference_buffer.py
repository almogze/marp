import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class EpisodeRecord:
    agent_trajs: Dict[str, List[Tuple[np.ndarray, int]]]
    metrics: Dict[str, float]


class PreferenceBuffer:
    def __init__(self, max_episodes: int):
        self._episodes: Deque[EpisodeRecord] = deque(maxlen=max_episodes)

    def add_episode(self, record: EpisodeRecord) -> None:
        self._episodes.append(record)

    def __len__(self) -> int:
        return len(self._episodes)

    def _as_list(self) -> List[EpisodeRecord]:
        return list(self._episodes)

    def sample_episode_pairs(self, batch_pairs: int) -> List[Tuple[EpisodeRecord, EpisodeRecord]]:
        episodes = self._as_list()
        n = len(episodes)
        if n < 2:
            return []
        pairs = []
        for _ in range(batch_pairs):
            i, j = random.sample(range(n), 2)
            pairs.append((episodes[i], episodes[j]))
        return pairs

    def aggregate_episode(self, record: EpisodeRecord) -> List[Tuple[np.ndarray, int]]:
        merged: List[Tuple[np.ndarray, int]] = []
        for agent_id in sorted(record.agent_trajs.keys()):
            merged.extend(record.agent_trajs.get(agent_id, []))
        return merged

    def sample_agent_trajectory(self, record: EpisodeRecord) -> List[Tuple[np.ndarray, int]]:
        if not record.agent_trajs:
            return []
        agent_ids = list(record.agent_trajs.keys())
        agent_id = random.choice(agent_ids)
        return record.agent_trajs.get(agent_id, [])

    def sample_narrow_view_pairs(
        self, batch_pairs: int
    ) -> List[Tuple[List[Tuple[np.ndarray, int]], List[Tuple[np.ndarray, int]], EpisodeRecord, EpisodeRecord]]:
        pairs = self.sample_episode_pairs(batch_pairs)
        output = []
        for ep_i, ep_j in pairs:
            traj_i = self.sample_agent_trajectory(ep_i)
            traj_j = self.sample_agent_trajectory(ep_j)
            output.append((traj_i, traj_j, ep_i, ep_j))
        return output

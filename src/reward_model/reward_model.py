from typing import List, Tuple, Sequence, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class RewardModel(nn.Module):
    def __init__(self, obs_shape: Tuple[int, int, int], num_actions: int = 8) -> None:
        super().__init__()
        self.obs_shape = obs_shape
        self.num_actions = num_actions
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
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + num_actions, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        if actions.ndim == 0:
            actions = actions.unsqueeze(0)
        if obs.ndim != 4:
            raise ValueError(f"Expected obs with shape (B,H,W,C), got {obs.shape}")
        x = obs.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = self.encoder(x)
        action_feat = F.one_hot(actions.long(), num_classes=self.num_actions).float()
        x = torch.cat([x, action_feat], dim=1)
        reward = self.head(x).squeeze(-1)
        return reward

    def predict(self, obs_img: np.ndarray, action: int) -> float:
        obs_tensor = torch.from_numpy(np.asarray(obs_img)).float().unsqueeze(0).to(self.device)
        action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)
        with torch.no_grad():
            reward = self.forward(obs_tensor, action_tensor)
        return float(reward.item())

    def sequence_score(
        self,
        traj: Sequence[Tuple[np.ndarray, int]],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if not traj:
            return torch.tensor(0.0, device=device or self.device)
        obs = np.stack([step[0] for step in traj], axis=0).astype(np.float32)
        actions = np.array([step[1] for step in traj], dtype=np.int64)
        obs_tensor = torch.from_numpy(obs).float().to(device or self.device)
        action_tensor = torch.from_numpy(actions).long().to(device or self.device)
        rewards = self.forward(obs_tensor, action_tensor)
        return rewards.sum()

    def save(self, path: str) -> None:
        payload = {
            "state_dict": self.state_dict(),
            "obs_shape": self.obs_shape,
            "num_actions": self.num_actions,
        }
        torch.save(payload, path)

    @staticmethod
    def load(path: str, device: str = "auto") -> "RewardModel":
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        payload = torch.load(path, map_location=device)
        model = RewardModel(tuple(payload["obs_shape"]), int(payload["num_actions"]))
        model.load_state_dict(payload["state_dict"])
        model.to(torch.device(device))
        return model

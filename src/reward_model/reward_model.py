from typing import List, Tuple, Sequence, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# Default chunk size for processing long sequences
DEFAULT_CHUNK_SIZE: int = 512


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


    def _forward_chunk(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        """Forward pass for a single chunk of observations and actions."""
        obs_tensor = torch.from_numpy(obs).float().to(device)
        action_tensor = torch.from_numpy(actions).long().to(device)
        return self.forward(obs_tensor, action_tensor)

    def sequence_score(
        self,
        traj: Sequence[Tuple[np.ndarray, int]],
        device: Optional[torch.device] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> torch.Tensor:
        """Score a single trajectory, processing in chunks to limit memory."""
        if not traj:
            return torch.tensor(0.0, device=device or self.device)
        
        dev = device or self.device
        obs = np.stack([step[0] for step in traj], axis=0).astype(np.float32)
        actions = np.array([step[1] for step in traj], dtype=np.int64)
        
        # Process in chunks if sequence is long
        if len(traj) <= chunk_size:
            rewards = self._forward_chunk(obs, actions, dev)
            return rewards.sum()
        
        # Chunked processing for memory efficiency
        total = torch.tensor(0.0, device=dev)
        for i in range(0, len(traj), chunk_size):
            end = min(i + chunk_size, len(traj))
            chunk_rewards = self._forward_chunk(obs[i:end], actions[i:end], dev)
            total = total + chunk_rewards.sum()
        return total

    def batch_sequence_scores(
        self,
        trajectories: List[Sequence[Tuple[np.ndarray, int]]],
        device: Optional[torch.device] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> torch.Tensor:
        """
        Score multiple trajectories efficiently using batched forward passes.
        
        Returns a tensor of shape (num_trajectories,) with the sum of rewards per trajectory.
        """
        dev = device or self.device
        
        if not trajectories:
            return torch.tensor([], device=dev)
        
        # Filter out empty trajectories and track their indices
        non_empty_indices = []
        non_empty_trajs = []
        for i, traj in enumerate(trajectories):
            if traj:
                non_empty_indices.append(i)
                non_empty_trajs.append(traj)
        
        # Initialize result tensor with zeros
        result = torch.zeros(len(trajectories), device=dev)
        
        if not non_empty_trajs:
            return result
        
        # Collect all observations and actions with trajectory indices
        all_obs = []
        all_actions = []
        traj_indices = []  # Which trajectory each step belongs to
        
        for local_idx, traj in enumerate(non_empty_trajs):
            for step in traj:
                all_obs.append(step[0])
                all_actions.append(step[1])
                traj_indices.append(local_idx)
        
        # Convert to arrays
        all_obs = np.stack(all_obs, axis=0).astype(np.float32)
        all_actions = np.array(all_actions, dtype=np.int64)
        traj_indices = np.array(traj_indices, dtype=np.int64)
        
        total_steps = len(all_obs)
        
        # Process in chunks and accumulate rewards per trajectory
        local_scores = torch.zeros(len(non_empty_trajs), device=dev, dtype=torch.float32)
        
        for i in range(0, total_steps, chunk_size):
            end = min(i + chunk_size, total_steps)
            chunk_obs = all_obs[i:end]
            chunk_actions = all_actions[i:end]
            chunk_indices = traj_indices[i:end]
            
            # Forward pass for this chunk
            chunk_rewards = self._forward_chunk(chunk_obs, chunk_actions, dev)
            
            # Ensure float32 for scatter_add_ compatibility (handles autocast FP16)
            chunk_rewards = chunk_rewards.float()
            
            # Scatter-add rewards to their respective trajectories
            chunk_indices_tensor = torch.from_numpy(chunk_indices).long().to(dev)
            local_scores.scatter_add_(0, chunk_indices_tensor, chunk_rewards)
        
        # Map back to original indices
        for local_idx, orig_idx in enumerate(non_empty_indices):
            result[orig_idx] = local_scores[local_idx]
        
        return result

    def batch_sequence_scores(
        self,
        trajectories: List[Sequence[Tuple[np.ndarray, int]]],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Compute sequence scores for multiple trajectories in a single batched forward pass.
        
        This is significantly faster than calling sequence_score() for each trajectory
        individually, as it minimizes GPU transfers and leverages batch parallelism.
        
        Args:
            trajectories: List of trajectories, each a sequence of (obs, action) tuples
            device: Device to use for computation
            
        Returns:
            Tensor of shape (num_trajectories,) with the sum of rewards for each trajectory
        """
        dev = device or self.device
        if not trajectories:
            return torch.tensor([], device=dev)
        
        # Handle empty trajectories and collect lengths
        lengths = []
        non_empty_indices = []
        for i, traj in enumerate(trajectories):
            if traj:
                lengths.append(len(traj))
                non_empty_indices.append(i)
        
        # If all trajectories are empty, return zeros
        if not lengths:
            return torch.zeros(len(trajectories), device=dev)
        
        max_len = max(lengths)
        num_non_empty = len(non_empty_indices)
        
        # Get obs shape from first non-empty trajectory
        first_traj = trajectories[non_empty_indices[0]]
        obs_shape = first_traj[0][0].shape
        
        # Pre-allocate padded arrays
        all_obs = np.zeros((num_non_empty, max_len, *obs_shape), dtype=np.float32)
        all_actions = np.zeros((num_non_empty, max_len), dtype=np.int64)
        mask = np.zeros((num_non_empty, max_len), dtype=np.float32)
        
        # Fill arrays
        for batch_idx, traj_idx in enumerate(non_empty_indices):
            traj = trajectories[traj_idx]
            traj_len = len(traj)
            for step_idx, (obs, action) in enumerate(traj):
                all_obs[batch_idx, step_idx] = obs
                all_actions[batch_idx, step_idx] = action
            mask[batch_idx, :traj_len] = 1.0
        
        # Reshape for batch processing: (num_seqs, max_len, H, W, C) -> (num_seqs * max_len, H, W, C)
        batch_obs = all_obs.reshape(-1, *obs_shape)
        batch_actions = all_actions.reshape(-1)
        
        # Single transfer to GPU and single forward pass
        obs_tensor = torch.from_numpy(batch_obs).float().to(dev)
        action_tensor = torch.from_numpy(batch_actions).long().to(dev)
        mask_tensor = torch.from_numpy(mask).to(dev)
        
        # Forward pass for all steps at once
        rewards = self.forward(obs_tensor, action_tensor)  # (num_seqs * max_len,)
        rewards = rewards.view(num_non_empty, max_len)  # (num_seqs, max_len)
        
        # Apply mask and sum per trajectory
        masked_rewards = rewards * mask_tensor
        non_empty_scores = masked_rewards.sum(dim=1)  # (num_non_empty,)
        
        # Build full output tensor with zeros for empty trajectories
        scores = torch.zeros(len(trajectories), device=dev)
        for batch_idx, traj_idx in enumerate(non_empty_indices):
            scores[traj_idx] = non_empty_scores[batch_idx]
        
        return scores

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

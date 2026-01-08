from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from .oracle import compute_phi, preference
from .preference_buffer import EpisodeRecord, PreferenceBuffer
from .reward_model import RewardModel


class RewardModelTrainer:
    def __init__(self, reward_model: RewardModel, lr: float = 1e-4, device: str = "auto") -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.reward_model = reward_model.to(self.device)
        self.optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=lr)

    def _pair_sequences(
        self, buffer: PreferenceBuffer, ep_i: EpisodeRecord, ep_j: EpisodeRecord, mode: str
    ) -> Tuple[List[Tuple[np.ndarray, int]], List[Tuple[np.ndarray, int]]]:
        if mode == "input_aggregation":
            return buffer.aggregate_episode(ep_i), buffer.aggregate_episode(ep_j)
        if mode == "narrow_view":
            return buffer.sample_agent_trajectory(ep_i), buffer.sample_agent_trajectory(ep_j)
        raise ValueError(f"Unsupported reward model mode: {mode}")

    @staticmethod
    def _weights_from_deltas(deltas: torch.Tensor) -> torch.Tensor:
        if deltas.numel() == 0:
            return deltas
        mean = deltas.mean()
        std = deltas.std(unbiased=False)
        z = (deltas - mean) / (std + 1e-6)
        weights = torch.softmax(z, dim=0)
        return weights

    def _episode_score_corr(
        self,
        buffer: PreferenceBuffer,
        batch_pairs: List[Tuple[EpisodeRecord, EpisodeRecord]],
        phi_key: str,
    ) -> Optional[float]:
        episodes = []
        seen = set()
        for ep_i, ep_j in batch_pairs:
            if id(ep_i) not in seen:
                episodes.append(ep_i)
                seen.add(id(ep_i))
            if id(ep_j) not in seen:
                episodes.append(ep_j)
                seen.add(id(ep_j))
        if len(episodes) < 2:
            return None
        
        # Batch compute all episode scores at once
        seqs = [buffer.aggregate_episode(ep) for ep in episodes]
        phis = [compute_phi(ep.metrics, phi_key) for ep in episodes]
        
        self.reward_model.eval()
        with torch.no_grad():
            scores_tensor = self.reward_model.batch_sequence_scores(seqs, device=self.device)
            scores = scores_tensor.cpu().numpy()
        
        scores_arr = np.array(scores, dtype=np.float32)
        phis_arr = np.array(phis, dtype=np.float32)
        if scores_arr.std() == 0 or phis_arr.std() == 0:
            return None
        corr = float(np.corrcoef(scores_arr, phis_arr)[0, 1])
        return corr

    def train(
        self,
        buffer: PreferenceBuffer,
        phi_key: str,
        mode: str,
        batch_pairs: int,
        train_steps: int,
    ) -> Dict[str, float]:
        if len(buffer) < 2:
            return {}
        losses: List[float] = []
        accuracies: List[float] = []
        last_pairs: List[Tuple[EpisodeRecord, EpisodeRecord]] = []

        self.reward_model.train()
        for _ in range(train_steps):
            pairs = buffer.sample_episode_pairs(batch_pairs)
            if not pairs:
                break
            last_pairs = pairs
            
            # Collect all sequences and metadata first (with caching for repeated episodes)
            seqs_i = []
            seqs_j = []
            mus = []
            deltas = []
            episode_cache: Dict[int, List[Tuple[np.ndarray, int]]] = {}
            
            for ep_i, ep_j in pairs:
                phi_i = compute_phi(ep_i.metrics, phi_key)
                phi_j = compute_phi(ep_j.metrics, phi_key)
                mu, delta = preference(phi_i, phi_j)
                
                # Cache aggregated episodes to avoid recomputation
                ep_i_id = id(ep_i)
                ep_j_id = id(ep_j)
                if ep_i_id not in episode_cache:
                    seq_i, _ = self._pair_sequences(buffer, ep_i, ep_j, mode)
                    episode_cache[ep_i_id] = seq_i
                if ep_j_id not in episode_cache:
                    _, seq_j = self._pair_sequences(buffer, ep_i, ep_j, mode)
                    episode_cache[ep_j_id] = seq_j
                
                seqs_i.append(episode_cache[ep_i_id])
                seqs_j.append(episode_cache[ep_j_id])
                mus.append(mu)
                deltas.append(delta)

            # Batch all sequences together for a single forward pass
            all_seqs = seqs_i + seqs_j
            all_scores = self.reward_model.batch_sequence_scores(all_seqs, device=self.device)
            scores_i_t = all_scores[:len(seqs_i)]
            scores_j_t = all_scores[len(seqs_i):]
            mu_t = torch.tensor(mus, dtype=torch.float32, device=self.device)
            delta_t = torch.tensor(deltas, dtype=torch.float32, device=self.device)
            weights = self._weights_from_deltas(delta_t)
            prob = torch.sigmoid(scores_i_t - scores_j_t)
            bce = F.binary_cross_entropy(prob, mu_t, reduction="none")
            loss = (bce * weights).sum() / (weights.sum() + 1e-8)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            acc = ((prob >= 0.5).float() == mu_t).float().mean().item()
            losses.append(float(loss.item()))
            accuracies.append(float(acc))

        corr = None
        if last_pairs:
            corr = self._episode_score_corr(buffer, last_pairs, phi_key)

        metrics: Dict[str, float] = {}
        if losses:
            metrics["loss"] = float(np.mean(losses))
        if accuracies:
            metrics["pref_accuracy"] = float(np.mean(accuracies))
        if corr is not None:
            metrics["score_phi_corr"] = float(corr)
        return metrics

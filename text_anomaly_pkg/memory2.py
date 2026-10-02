"""
Prioritized Experience Replay (PER) with Sliding-Window Pruning Constraints
===========================================================================
Implements a binary sum-tree structure with an automated rolling window pruning 
parameter to clear stale transitions seamlessly during live stream processing tasks.

Author: AI Collaborator
Date: October 2026
"""

import numpy as np
import random
import torch
from typing import Tuple, List, Any, Dict

class PruningSumTree:
    """A binary tree data structure that supports index-based stale leaf eviction."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        # Structural metadata tracking array to store system runtime execution steps
        self.timestamps = np.zeros(capacity, dtype=int)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    @property
    def total_priority(self) -> float:
        return self.tree[0]

    def add(self, p: float, data: Any, step_index: int):
        """Inserts a transition experience bound with a persistent step validation label."""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.timestamps[self.write] = step_index
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, p: float):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def prune_stale_leaves(self, current_step: int, max_age_window: int) -> int:
        """Identifies and zeroes out transition elements breaking lookback boundaries.
        
        Returns:
            The integer count of stale items safely evicted during the cleanup cycle.
        """
        pruned_count = 0
        # Sweep active memory spaces to find nodes that breach the sliding window age limit
        for i in range(self.n_entries):
            if self.tree[i + self.capacity - 1] > 0: # Ensure node is currently active
                age = current_step - self.timestamps[i]
                if age > max_age_window:
                    # Clear leaf node priority to isolate it from downstream sampling pools
                    leaf_idx = i + self.capacity - 1
                    self.update(leaf_idx, 0.0)
                    self.data[i] = None
                    pruned_count += 1
                    
        return pruned_count


class PrioritizedReplayBuffer:
    """Memory buffer managing priority weights and sliding-window expiry rules."""
    
    def __init__(self, capacity: int = 5000, alpha: float = 0.6, beta_start: float = 0.4, max_age_window: int = 1000):
        """
        Args:
            capacity: Maximum number of experiences stored in the buffer.
            alpha: Randomization exponent parameter (0 = uniform, 1 = strict priority).
            beta_start: Importance-sampling correction factor starting scale.
            max_age_window: Sliding-window constraint parameter (prunes items older than N steps).
        """
        self.tree = PruningSumTree(capacity)
        self.alpha = alpha
        self.beta = beta_start
        self.max_age_window = max_age_window
        self.epsilon = 1e-5
        self.global_step_counter = 0

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Saves a transition experience linked to the current global system time-step."""
        self.global_step_counter += 1
        
        # Periodically enforce sliding window constraints automatically every 100 insertions
        if self.global_step_counter % 100 == 0:
            evicted = self.tree.prune_stale_leaves(self.global_step_counter, self.max_age_window)
            if evicted > 0:
                print(f"[MEM CLEANUP] Evicted {evicted} stale transitions older than {self.max_age_window} frames.")

        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = 1.0
            
        transition_payload = (state, action, reward, next_state, done)
        self.tree.add(max_p, transition_payload, self.global_step_counter)

    def sample(self, batch_size: int) -> Tuple[List[int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idxs, priorities = [], []
        
        segment = self.tree.total_priority / batch_size
        self.beta = min(1.0, self.beta + 0.001)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            
            # Fallback guard routine to filter out raw dead branches during race conditions
            if data is None:
                continue
                
            priorities.append(p)
            idxs.append(idx)
            states.append(data)
            actions.append(data)
            rewards.append(data)
            next_states.append(data)
            dones.append(data)

        sampling_probabilities = np.array(priorities) / (self.tree.total_priority + 1e-10)
        weights = (self.tree.n_entries * sampling_probabilities) ** (-self.beta)
        weights /= (weights.max() + 1e-10)

        return (
            idxs,
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1),
            torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
        )

    def update_priorities(self, idxs: List[int], errors: np.ndarray):
        for idx, error in zip(idxs, errors):
            p = (np.abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)

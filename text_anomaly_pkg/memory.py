"""
Prioritized Experience Replay (PER) Module via Sum-Tree
======================================================
Implements a binary sum-tree structure to optimize memory management 
and experience tracking for reinforcement learning agents. High-error 
text blocks are prioritized to maximize sampling efficiency.

Author: AI Collaborator
Date: October 2026
"""

import numpy as np
import random
import torch
from typing import Tuple, List, Any

class SumTree:
    """A binary tree data structure where parent nodes store the sum of child priorities."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        # A tree array of size (2 * capacity - 1) holds all parent and leaf node priorities
        self.tree = np.zeros(2 * capacity - 1)
        # Flat object array matching raw transition parameters
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        """Recursively updates parent nodes up to the root element."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Traverses down the tree branches to find a leaf node matching priority value s."""
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
        """Returns the global cumulative priority sum stored at the tree root node."""
        return self.tree[0]

    def add(self, p: float, data: Any):
        """Inserts a new data sample with its corresponding priority scale into memory."""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0  # Wrap around buffer index
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, p: float):
        """Updates the priority value of a specific index and cascades the change upstream."""
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        """Retrieves a transition sample matching the scalar value scale profile."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Memory buffer managing priority tracking and importance-sampling adjustments."""
    
    def __init__(self, capacity: int = 5000, alpha: float = 0.6, beta_start: float = 0.4):
        """
        Args:
            capacity: Maximum number of experiences stored in the buffer queue.
            alpha: Randomization exponent parameter (0 = uniform, 1 = strict priority).
            beta_start: Importance-sampling correction factor starting scale boundaries.
        """
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta_start
        self.epsilon = 1e-5

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Saves a transition experience with maximum current priority initialization."""
        # Isolate leaf choices slicing window segment
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = 1.0
        self.tree.add(max_p, (state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[List[int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples a mini-batch of experiences based on tree priority scales."""
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idxs, priorities = [], []
        
        segment = self.tree.total_priority / batch_size
        self.beta = min(1.0, self.beta + 0.001)  # Anneal beta toward 1.0

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            
            priorities.append(p)
            idxs.append(idx)
            states.append(data[0])
            actions.append(data[1])
            rewards.append(data[2])
            next_states.append(data[3])
            dones.append(data[4])

        # Compute Importance-Sampling Weights (IS) to correct distribution shift bias
        sampling_probabilities = np.array(priorities) / (self.tree.total_priority + 1e-10)
        weights = (self.tree.n_entries * sampling_probabilities) ** (-self.beta)
        weights /= (weights.max() + 1e-10)  # Normalize weights scale baseline

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
        """Updates tree node weights with fresh absolute TD errors from model loss logs."""
        for idx, error in zip(idxs, errors):
            # Smooth error scaling via alpha and protection epsilon parameters
            p = (np.abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)

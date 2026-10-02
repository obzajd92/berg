import torch
import torch.nn as nn

class DuelingDQNNetwork(nn.Module):
    """Splits network layers into state-value V(s) and action-advantage A(s,a) components."""
    def __init__(self, state_dim: int = 2, action_dim: int = 3):
        super(DuelingDQNNetwork, self).__init__()
        self.backbone = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU())
        self.value_stream = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, action_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

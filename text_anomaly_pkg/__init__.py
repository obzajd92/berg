"""
Text Anomaly Package Namespace Interface
========================================
Exposes top-level class architectures for the Dueling Double-DQN pipeline, 
Prioritized Experience Replay (PER), and live streaming text evaluation.

Modules:
    - LiveTextInferenceRunner: Stream processor and weight reloader layer.
    - DuelingDQNNetwork: Value/Advantage splitting policy model.
    - PrioritizedReplayBuffer: Binary Sum-Tree experience tracking memory.
"""

from .inference import LiveTextInferenceRunner
from .models import DuelingDQNNetwork
from .memory import PrioritizedReplayBuffer, SumTree

# Establish public API exposure layout
__all__ = [
    "LiveTextInferenceRunner",
    "DuelingDQNNetwork",
    "PrioritizedReplayBuffer",
    "SumTree",
]

__version__ = "1.0.0"

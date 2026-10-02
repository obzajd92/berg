{
  "environment": {
    "name": "T.S. Eliot Multi-Movement Ingestion Engine",
    "debug_mode": true
  },
  "dqn_agent": {
    "state_dim": 2,
    "action_dim": 3,
    "learning_rate": 0.001
  },
  "memory_per": {
    "capacity": 5000,
    "alpha": 0.6,
    "beta": 0.4,
    "max_age_window": 1200,
    "min_age_window_bound": 300,
    "latency_spike_threshold_ms": 15.0
  }
}

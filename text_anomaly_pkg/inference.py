import os
import re
import numpy as np
import torch
from typing import Dict, Any
from .models import DuelingDQNNetwork

class LiveTextInferenceRunner:
    """Loads saved models to evaluate incoming live text streams frame-by-frame."""
    def __init__(self, model_path: str, state_dim: int = 2, action_dim: int = 3):
        self.model = DuelingDQNNetwork(state_dim, action_dim)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print(f"-> Production weights reloaded from: {model_path}")
        else:
            print("-> Warning: Weight file not found. Running initialized defaults.")
        self.model.eval()

    def process_stream_frame(self, raw_text_chunk: str) -> Dict[str, Any]:
        tokens = re.findall(r'\b\w+\b', raw_text_chunk.lower())
        if not tokens:
            return {"selected_action": -1, "words_mean_len": 0.0, "token_density_ratio": 0.0, "raw_token_count": 0}
        
        word_freq = len(tokens)
        words_mean = np.mean([len(w) for w in tokens])
        token_density = len(set(tokens)) / word_freq
        
        state_vector = np.array([words_mean, token_density], dtype=np.float32)
        state_tensor = torch.tensor(state_vector).unsqueeze(0)
        
        with torch.no_grad():
            action = int(self.model(state_tensor).argmax(dim=1).item())
            
        return {
            "selected_action": action, 
            "words_mean_len": float(words_mean), 
            "token_density_ratio": float(token_density), 
            "raw_token_count": word_freq
        }

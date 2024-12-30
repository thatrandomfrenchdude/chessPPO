import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn

# policy network to choose moves
class ChessPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.LayerNorm(4096)
        )

    def forward(self, x):
        return self.net(x)

# critic network to evaluate positions
class ChessValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Input processing
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Hidden layers
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Value head
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def choose_move(moves, policy_net, state, temperature=1.0):
    """Choose a move using the policy network with safety checks."""
    with torch.no_grad():
        # Ensure state has batch dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        # Get and validate logits
        logits = policy_net(state)
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        
        # Temperature scaling
        logits = logits / temperature
        
        # Safe softmax
        probs = nn.Softmax(dim=-1)(logits)
        probs = probs.squeeze(0)
        
        # Create move lookup with validation
        move_probs = []
        for move in moves:
            move_idx = move.from_square * 64 + move.to_square
            if move_idx < 4096:  # Valid index check
                prob = probs[move_idx].item()
                if prob > 0 and not np.isnan(prob) and not np.isinf(prob):
                    move_probs.append((move, prob))
        
        if not move_probs:
            return random.choice(moves)
            
        # Safe sampling
        moves, probs = zip(*move_probs)
        probs = torch.tensor(probs)
        probs = torch.clamp(probs, min=1e-10)
        probs = probs / probs.sum()
        
        try:
            move_idx = torch.multinomial(probs, 1).item()
            return moves[move_idx]
        except RuntimeError:
            return random.choice(moves)

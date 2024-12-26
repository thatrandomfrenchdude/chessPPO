import numpy as np
import os
import random
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

def encode_fen(fen):
    '''
    Create a binary vector representation of the chess position.
    '''
    
    piece_map = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                 'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    
    board_tensor = torch.zeros(12, 8, 8)  # 12 piece types, 8x8 board
    
    # Parse the FEN string
    board_str = fen.split(' ')[0]
    row, col = 0, 0
    
    for char in board_str:
        if char == '/':
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        else:
            board_tensor[piece_map[char]][row][col] = 1
            col += 1
            
    return board_tensor.flatten()

def choose_move(env, policy_net):
    '''
    Choose a move using the policy network with temperature scaling.

    Args:
        env: Chess environment object
        policy_net: Policy network model
    '''

    moves = env.get_legal_moves()
    if not moves:
        return None
    
    # Get state representation
    state = encode_fen(env.board.fen())
    state = state.unsqueeze(0)
    
    # Get action probabilities with numerical stability
    with torch.no_grad():
        logits = policy_net(state)
        # Add temperature scaling for numerical stability
        logits = logits / 10.0  # Reduce extreme values
        probs = nn.Softmax(dim=1)(logits)
    
    # Create move lookup with validation
    move_probs = []
    for move in moves:
        move_idx = (move.from_square * 64 + move.to_square)
        if move_idx < 4096:  # Ensure index is within bounds
            prob = probs[0][move_idx].item()
            if prob > 0 and not np.isnan(prob) and not np.isinf(prob):
                move_probs.append((move, prob))
    
    # Fallback to random move if no valid probabilities
    if not move_probs:
        return random.choice(moves)
    
    # Safe probability normalization
    moves, probs = zip(*move_probs)
    probs = torch.tensor(probs)
    probs = torch.clamp(probs, min=1e-10)  # Prevent zero probabilities
    probs = probs / probs.sum()  # Normalize
    
    try:
        move_idx = torch.multinomial(probs, 1).item()
        return moves[move_idx]
    except RuntimeError:
        # Fallback to random choice if sampling fails
        return random.choice(moves)
    
def load_or_create_model(model_class, model_path, device):
    '''
    Load existing model or create new one
    
    Args:
        model_class: Model class to instantiate
        model_path: File path to load model weights
        device: Device to load model on
        
    Returns:
        model: Model object    
    '''
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = model_class().to(device)
        model.load_state_dict(torch.load(model_path))
        return model
    else:
        print(f"No existing model found at {model_path}, creating new model")
        model = model_class().to(device)
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # Save initial model
        torch.save(model.state_dict(), model_path)
        return model
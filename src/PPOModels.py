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

def choose_move(
    legal_moves: list,
    policy_net: ChessPolicy,
    board_fen: str,
    temperature: float = 10.0
) -> str:
    '''
    Choose a move using the policy network with temperature scaling.
    '''
    
    # Get state representation
    state = encode_fen(board_fen)
    state = state.unsqueeze(0)
    
    # Get action probabilities
    with torch.no_grad():
        logits = policy_net(state)
        logits = logits / temperature  # Temperature scaling to match chessPPO.py
        probs = nn.Softmax(dim=1)(logits)
    
    # Create move lookup with all the moves and their probabilities
    move_probs = []
    for move in legal_moves:
        move_idx = move.from_square * 64 + move.to_square
        if move_idx < 4096:
            prob = probs[0][move_idx].item()
            if prob > 0 and not np.isnan(prob) and not np.isinf(prob):
                move_probs.append((move, prob))
    
    # system fault if there are no valid moves
    # there is a chack for legal moves prior to the call 
    if not move_probs:
        sys.exit("Error: no valid moves.")
        # return random.choice(moves)
        
    # Sample move based on probabilities
    moves, probs = zip(*move_probs) # unzip the tuples of moves and probabilities
    probs = torch.tensor(probs) # convert to tensor
    probs = torch.clamp(probs, min=1e-10)  # avoid log(0)
    probs = probs / probs.sum() # normalize probabilities
    
    try:
        move_idx = torch.multinomial(probs, 1).item()
        return moves[move_idx]
    except RuntimeError:
        sys.exit("Error: move_idx is out of bounds.")
        # return random.choice(moves)

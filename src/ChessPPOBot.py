import numpy as np
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim

"""
The chess PPO bot is focused on winning the
game by making the best moves possible.

I haven't yet understood the best way to
maintain evaluations for the individual
board states and the game as a whole.

Do I need 2 critic models? One for game,
one for the individual move chosen? Or
does one need an evaluation function?

When I am playing a game, I am using the
policy model to guess a move at each step.
The policy model, after selecting a move,
is corrected by ...? PPO would tend to
suggest the value model, but this needs
to be saved to assess the bots overall
performance. Or does it?

The value model needs to be trained though.
I need to provide rewards. Can rewards
be provided by stockfish?

Suggestion from ChatGPT about Hybrid
training approach (moves + game):

After each move:
- Compute an intermediate reward(e.g.,
from Stockfish or material advantage).
- Store the trajectory for training.
- Optionally, update the models
incrementally using the intermediate
reward.

After the game:
- Assign a cumulative reward based on
the outcome (win, loss, draw).
- Use the full game trajectory to perform
updates on the models.
"""
class ChessPPOBot:
    """
    Module containing the bot.

    Loads or creates a model.

    Saves the model after training.

    Contains functions to:
        - choose a chess move
        - evaluate the move choice using the before and after
        - update the model based on the move
    """
    def __init__(
        self,
        bot_name,
        training,
        policy_model_path,
        policy_learning_rate,
        value_model_path,
        value_learning_rate
    ):
        self.bot_name = bot_name
        # initialize bot models
        """
        Actor
        
        Takes observations from the environment
        and returns a probability distribution
        over the available actions.

        Take in a board position. Calculate all
        legal positions from the current board
        and the probability of each move being
        chosen for a given position.

        I belive that the actor model needs to be
        modified for chess such that, given a chess
        position, it returns the best move to make.

        The reason it needs to be modified is that
        the number of legal chess moves in any given
        position is not fixed and often quite large.
        As such, I think it would make sense to use
        Monte Carlo Tree Search or some other method
        to choose the best move.
        """
        self.actor = self.load_or_create_model(
            ChessPolicy,
            policy_model_path
        )

        """
        Critic
        
        Estimates the total possible rewards for
        the chess game as a whole.

        Estimates the probability of winning from
        a given board position.
        """
        self.critic = self.load_or_create_model(
            ChessEval,
            value_model_path,
        )

        if training:
            # initialize optimizers
            self.actor_optimizer = optim.Adam(
                self.policy_net.parameters(),
                lr=policy_learning_rate
            )
            self.critic_optimizer = optim.Adam(
                self.value_net.parameters(),
                lr=value_learning_rate
            )

    def load_or_create_model(
        self,
        model_class: nn.Module,
        model_path: str,
    ) -> nn.Module:
        '''
        Load existing model from file or create a new one.

        Args:
            model_class: Model class to instantiate
            model_path: Path to model file

        Returns:
            Model object
        '''

        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path}")
            model = model_class().to(self.config['device'])
            model.load_state_dict(torch.load(model_path))
            return model
        else:
            print(f"No existing model found at {model_path}, creating new model")
            model = model_class().to(self.config['device'])
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            # Save initial model
            torch.save(model.state_dict(), model_path)
            return model

    def choose_move(self, board):
        """
        Choose a move based on the current board state.

        Uses the policy model to choose a move.

        Policy model returning the probability of
        each legal move being chosen for the given
        board state.
        """
        # get the move probabilities
        move_probs = self.actor.forward(board)

        # return the move with the highest probability
        return max(move_probs, key=lambda x: x[1])

    def evaluate_move(self, board, move):
        """
        Evaluate the move based on the current board state and the move.
        """
        raise NotImplementedError

    def update(self, board, move):
        """
        Update the actor model based on the best move choice.
        """
        raise NotImplementedError

    def save_bot(self):
        """
        Save the bot models to disk.
        """
        raise NotImplementedError

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
class ChessEval(nn.Module):
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

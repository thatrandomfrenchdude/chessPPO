from datetime import datetime
import logging
import sys

import torch

from src.ChessPPOBot import ChessPPOBot
from src.TrainingSession import TrainingSession
import traceback

# references
# ppo implementation from paper
# https://www.youtube.com/watch?v=hlv79rcHws0
# analyze chess positions
# https://blog.propelauth.com/chess-analysis-in-python/

logging.basicConfig(
    filename=f'logs/chess_ppo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
#####################
MODE = "train" # or "play" to play against the bot

#####################
CHESS_PPO_BOT_NAME = "uno"
CHESS_PPO_BOT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# policy model
CHESS_PPO_BOT_POLICY_MODEL_PATH = f"models/{CHESS_PPO_BOT_NAME}_policy.pth"
CHESS_PPO_BOT_POLICY_LEARNING_RATE = 0.0005
# value model
CHESS_PPO_BOT_VALUE_MODEL_PATH = f"models/{CHESS_PPO_BOT_NAME}_value.pth"
CHESS_PPO_BOT_VALUE_LEARNING_RATE = 0.0005

#####################
TRAINING_SESSION_SAVE_SESSION = False
TRAINING_SESSION_SAVE_MODELS = False
TRAINING_SESSION_TYPE = "build"
TRAINING_SESSION_LENGTH = 1 # 100 usually, starting with 1 for testing
TRAINING_SESSION_DIR = "training_sessions"
TRAINING_SESSIONS_GAMES_DIR = f"{TRAINING_SESSION_DIR}/games"
TRAINING_SESSIONS_METRICS_DIR = f"{TRAINING_SESSION_DIR}/metrics"

#####################
CHESS_GAME_MAX_MOVES = 100

#####################
# main
class ChessPPOBotMain:
    """
    Module to train a PPO bot.

    Loads a PPO bot.

    Creates a Training Session.

    Playes Chess Games in the Training Session.
    """
    def __init__(self, mode):
        # initialize the bot
        self.bot = ChessPPOBot(
            device=CHESS_PPO_BOT_DEVICE,
            bot_name=CHESS_PPO_BOT_NAME,
            training=True, # True for self-play training, False for playing against a human
            policy_model_path=CHESS_PPO_BOT_POLICY_MODEL_PATH,
            policy_learning_rate=CHESS_PPO_BOT_POLICY_LEARNING_RATE,
            value_model_path=CHESS_PPO_BOT_VALUE_MODEL_PATH,
            value_learning_rate=CHESS_PPO_BOT_VALUE_LEARNING_RATE,
        )

        # run the bot in the specified mode
        if mode == "train":
            self.train()
        elif mode == "play":
            self.play()
        else:
            raise ValueError("Mode must be either 'train' or 'play'.")
        
    def train(self):
        # initialize the training session
        self.training_session = TrainingSession(
            self.bot,
            TRAINING_SESSION_LENGTH,
            TRAINING_SESSION_SAVE_SESSION,
            TRAINING_SESSION_SAVE_MODELS,
            TRAINING_SESSION_DIR,
            TRAINING_SESSIONS_GAMES_DIR,
            TRAINING_SESSIONS_METRICS_DIR,
            CHESS_GAME_MAX_MOVES
        )
        
        # launch the training session
        self.training_session.run()

    def play(self):
        raise NotImplementedError("Play mode not implemented yet.")

if __name__ == "__main__":
    ChessPPOBotMain(MODE)

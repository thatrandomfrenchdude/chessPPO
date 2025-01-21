# Chess PPO System

## System Overview
A reinforcement learning system that uses Proximal Policy Optimization (PPO) to train a chess-playing agent.

## Architecture
### File Structure
```
ChessPPO/
├── README.md
├── requirements.txt        # Python dependencies
├── chessPPO.py             # Training script
├── main.py                 # Training script
├── config.yaml             # Hyperparameters
├── src/
│   ├── ChessEnv.py         # Chess environment
│   ├── ChessPPOBot.py      # Neural network models
│   ├── TrainingSession.py  # Training Session
│   └── GameMetrics.py      # Game Metrics
├── models/                 # Model checkpoints
├── self-play-games/        # Saved PGN games
├── training-metrics/       # JSON metrics
└── logs/                   # Training logs
```

### Core Components
- `ChessGame`: Chess environment with reward calculation and game state management
    - Values
        - `board`: the game environment
        - `moves`: list of moves that have been played
    - Functions
        - `save`: Save the game to a PGN file
        - `get_postion`: Get the current board as fen string
        - `get_rewards`: Get the rewards for a move
        - `get_done`: Check if the game is over
        - `get_results`: Get the game result
        - `step`: Play a move, update the game state, and return the new state, reward, and done flag
        - `calc_position_value`: Calculate the value of the current position
- `ChessPPOBot`: PPO agent with policy and value networks
    - Values
        - `bot_name`
        - `Actor`: 768 -> 1024 -> 2048 -> 4096
        - `Critic`: 768 -> 1024 -> 512 -> 256 -> 64 -> 1
        - `actor_optimizer`
        - `critic_optimizer`
    - Functions
        - `load_or_create_model`: Load an existing model from file or create one if not found
        - `choose_move`: Choose a move based on the given state using the actor network
        - `evaluate_move`: Evaluate the chosen move using the critic network
        - `update`: Update the actor and critic networks
        - `save_bot`: Save the bot models to files
- `GameMetrics`: Game statistics tracking
    - Values
        - `start`: game start time
        - `timestamps`: list of timestamps for each step
        - `start_positions`: starting positions for each step
        - `actions`: list of moves played at each step
        - `rewards`: list of rewards for each step
        - `end_positions`: ending positions for each step
        - `dones`: list of done flags for each step
        - `result`: game result
    - Functions
        - `save_step`: Save the current game step
        - `save_result`: Save the game result
        - `save`: Save the game metrics to a JSON file
- `TrainingSession`: Training loop with self-play
    - Values
        - `bot`: The ChessPPOBot agent
        - `session_length`: Number of games to play
        - `save_session`: Save the session to files
        - `save_models`: Save the bot models to files
        - `session_dir`: Directory to save the session files
        - `games_dir`: Directory to save the pgn game files
        - `metrics_dir`: Directory to save the game metric files
    - Functions
        - `run`: Run the training session
        - `save`: Save the session to files
        - `plot_session`: Plot the session

## Usage
### v1
TBD

### v0
1. Configure hyperparameters in config dictionary
2. Activate or create the virtual environment:
    ```
    source chess-ppo-venv/bin/activate
    ```
    or
    ```
    python3.12 -m venv chess-ppo-venv
    pip install -r requirements.txt
    ```
3. Run training:
    ```
    python chessPPO.py
    ```

3. Monitor progress:
    - real-time metrics every 10 games
    - training plots saved as `training_progress.png`
    - detailed metrics in training-metrics/
    - games saved in PGN format in self-play-games/

#### Output Files
- `self-play-games/games_YYYYMMDD_batchX.pgn`: Saved games
- `training-metrics/metrics_DDMMYY_HHMMSS.json`: Training statistics
- `models/chess_ppo_checkpoint.pth`: Model weights
- `training_progress.png`: Training visualization
- `logs/chess_ppo_YYYYMMDD_HHMMSS.log`: Training logs

## Notes
### Neural Network Details
Network details are in progress.

v0 of the project uses these network details:
- Policy Network: Outputs move probabilities across all possible moves
- Value Network: Estimates state value with dropout layers for regularization
- Input encoding: FEN string converted to 768-dim binary vector (12 piece types × 8 × 8 board)

### Reward Structure
Reward structure is in progress.

v0 of the project uses this reward structure:
- Material evaluation using standard piece values
- Positional scoring via piece-square tables
- Game outcome rewards: Win (+10), Loss (-10), Draw (0)
- Capture rewards with configurable aggression modifier

### Configuration
v1 configuration is hardcoded in the main file.

v0 configuration:
```python
config = {
    'lr': 0.0005,                    # Learning rate
    'gamma': 0.99,                   # Discount factor
    'epsilon_clip': 0.1,             # PPO clipping parameter
    'k_epochs': 3,                   # PPO epochs per update
    'self_play_games': 1000,         # Number of training games
    'max_moves': 100,                # Max moves per game
    'device': 'cpu',                 # Training device
    'aggression': 0.0,               # Aggression factor [-1,1]
    'eval_games': 20,                # Evaluation games count
    'print_self_play': False         # Debug printing
}
```

### Strategies
Some general strategies I intend to impart to the agent:
- win > draw > loss
- resign as quickly as possible if loss is inevitable
    - recognize defeat on either side as quickly as possible
    - scaled reward/penalty based on the positional balance score, exponential to punish more heavily for larger imbalances
- generate observations about the state context for the language model to reason with
- context
    - general understanding of the game rules
    - general understanding of the game pieces and their general strategic usage

<!-- ### Training Process
1. Self-play episodes generate training data
2. PPO updates policy and value networks
3. Metrics tracked:
   - Win/loss/draw ratios
   - Episode rewards
   - Game lengths
   - Material advantage -->
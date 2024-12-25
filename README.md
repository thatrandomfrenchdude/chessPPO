# Chess PPO System

## System Overview
A reinforcement learning system that uses Proximal Policy Optimization (PPO) to train a chess-playing agent. The system features:

- Self-play training with PPO algorithm
- Configurable neural network architecture
- Comprehensive reward system including material and positional evaluation
- Metrics tracking and visualization
- PGN game storage

## Architecture

### Core Components
- `ChessEnv`: Chess environment with reward calculation and game state management
- `ChessPolicy`: Policy network (768 -> 1024 -> 2048 -> 4096)
- `ChessValue`: Value network (768 -> 1024 -> 512 -> 256 -> 64 -> 1)
- `TrainingMetrics`: Training metrics collection and storage

### Neural Network Details
- Policy Network: Outputs move probabilities across all possible moves
- Value Network: Estimates state value with dropout layers for regularization
- Input encoding: FEN string converted to 768-dim binary vector (12 piece types × 8 × 8 board)

### Reward Structure
- Material evaluation using standard piece values
- Positional scoring via piece-square tables
- Game outcome rewards: Win (+10), Loss (-10), Draw (0)
- Capture rewards with configurable aggression modifier

## Configuration

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
- win > draw > loss
- resign as quickly as possible if loss is inevitable
    - recognize defeat on either side as quickly as possible
    - scaled reward/penalty based on the positional balance score, exponential to punish more heavily for larger imbalances
- generate observations about the state context for the language model to reason with

### Context
- general understanding of the game rules
- general understanding of the game pieces and their general strategic usage

### Training Process
1. Self-play episodes generate training data
2. PPO updates policy and value networks
3. Metrics tracked:
   - Win/loss/draw ratios
   - Episode rewards
   - Game lengths
   - Material advantage

### File Structure
```
ChessPPO/
├── README.md
├── requirements.txt       # Python dependencies
├── chessPPO.py            # Training script
├── src/
│   ├── ChessEnv.py        # Chess environment
│   ├── PPOModels.py       # Neural network models
│   └── TrainingMetrics.py # Metrics tracking
├── models/                # Model checkpoints
├── self-play-games/       # Saved PGN games
├── training-metrics/      # JSON metrics
└── logs/                  # Training logs
```

### Usage
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

### Output Files
- `self-play-games/games_YYYYMMDD_batchX.pgn`: Saved games
- `training-metrics/metrics_DDMMYY_HHMMSS.json`: Training statistics
- `models/chess_ppo_checkpoint.pth`: Model weights
- `training_progress.png`: Training visualization
- `logs/chess_ppo_YYYYMMDD_HHMMSS.log`: Training logs
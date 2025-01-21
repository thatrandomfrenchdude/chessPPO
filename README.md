# Chess PPO System

## Table of Contents
1. [First, a Note on the Versions](#first-a-note-on-the-versions)
2. [Version Status](#version-status)
3. [System Overview](#system-overview)
4. [Architecture](#architecture)
    - [Core Components](#core-components)
    - [Project Structure](#project-structure)
5. [Usage](#usage)
    - [v1](#v1)
    - [v0](#v0)
6. [Working Notes](#working-notes)


## First, a Note on the Versions
This is a living project. v0 was an initial exploration of many different ideas and concepts. v1 is a more focused and refined version of the project.

I will continue to update it as I have time with the best intentions to maintain documentation and resources as I go.

That being said, some of this may require you to explore the wilderness of my thought process. Everything is here, but the map may not be immediately clear. Have fun!

## Version Status
- **[Active Development]** v1: Not fully functional, still in progress. Reworked the project architecture to more closely follow the PPO algorithm.
- **[Archived]** v0: Working prototype, doesn't train well. Mostly project layout exploration. This is "archived," all the code is still there. Follow the v0 usage instructions to run it. This will eventually be deprecated.

## System Overview
A reinforcement learning system that uses Proximal Policy Optimization (PPO) to train a chess-playing agent. The agent uses self-play to learn.

## Architecture
### Core Components
- [`ChessGame`](/docs/ChessGame/README.md): Chess environment with reward calculation and game state management
- [`ChessPPOBot`](/docs/ChessPPOBot/README.md): PPO agent with policy and value networks
- [`GameMetrics`](/docs/GameMetrics/README.md): Game statistics tracking
- [`TrainingSession`](/docs/TrainingSession/README.md): Training loop with self-play

Click each component for v1 developer documentation.

### Project Structure
```
ChessPPO/
├── README.md
├── requirements.txt        # Python dependencies
├── chessPPO.py             # Training script
├── main.py                 # Training script
├── config.yaml             # Hyperparameters
├── src/
│   ├── ChessGame.py        # Chess environment
│   ├── ChessPPOBot.py      # Neural network models
│   ├── TrainingSession.py  # Training Session
│   └── GameMetrics.py      # Game Metrics
├── models/                 # Model checkpoints
├── self-play-games/        # Saved PGN games
├── training-metrics/       # JSON metrics
└── logs/                   # Training logs
```

## Usage
### v1
1. Activate or create the virtual environment:
    ```
    source chess-ppo-venv/bin/activate
    ```
    or
    ```
    python3.12 -m venv chess-ppo-venv
    pip install -r requirements.txt
    ```
2. Run training:
    ```
    python main.py
    ```

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

## Working Notes
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
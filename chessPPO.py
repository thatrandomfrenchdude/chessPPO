import torch
import torch.nn as nn
import torch.optim as optim
# import chess
import chess.pgn
import random
import os
# import io
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
import logging
import traceback

from src.ChessEnv import ChessEnv, save_game, setup_games_directory
from src.PPOModels import ChessPolicy, ChessValue, encode_fen, choose_move
from src.TrainingMetrics import TrainingMetrics, plot_training_progress

##### CONFIGURATION #####
# Configuration section - modify these hyperparameters as needed
config = {}

ppo_config = {
    # PPO hyperparameters
    'lr': 0.0005,
    'gamma': 0.99,
    'epsilon_clip': 0.1,
    'k_epochs': 3,
    'self_play_games': 100,
    'max_moves': 100,
    'device': 'cpu',

    # manual configuration
    'print_self_play': False # Set to True to print self-play game details
}

path_config = {
    # default save/load paths for model checkpoints
    'save_model_path': 'models/chess_ppo_checkpoint.pth',
    'load_model_path': 'models/chess_ppo_checkpoint.pth', # None  # Set this to a file path to load existing weights

    # self-play game save file configuration
    'games_dir': 'self-play-games', # Directory to save self-play games
    'games_per_file': 25, # Number of games to save in each PGN file

    # training metrics directory
    'metrics_dir': 'training-metrics',  # Directory to save metrics
    'metrics_format': 'json'  # Format to save metrics
}

eval_config = {
    'eval_games': 20,  # Number of games to evaluate performance
    'plot_metrics': True,  # Whether to plot training metrics
    'metrics_smoothing': 10,  # Window for smoothing metrics
}

agent_behavior_config = {
    'aggression': 0.0,  # Range [-1, 1] for defensive to aggressive play
}

# unify configurations
config.update(ppo_config)
config.update(path_config)
config.update(eval_config)
config.update(agent_behavior_config)
##### END CONFIGURATION #####

##### LOGGING #####
logging.basicConfig(
    filename=f'logs/chess_ppo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
##### END LOGGING #####

# def encode_fen(fen):
#     '''
#     Create a binary vector representation of the chess position.
#     '''
    
#     piece_map = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
#                  'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    
#     board_tensor = torch.zeros(12, 8, 8)  # 12 piece types, 8x8 board
    
#     # Parse the FEN string
#     board_str = fen.split(' ')[0]
#     row, col = 0, 0
    
#     for char in board_str:
#         if char == '/':
#             row += 1
#             col = 0
#         elif char.isdigit():
#             col += int(char)
#         else:
#             board_tensor[piece_map[char]][row][col] = 1
#             col += 1
            
#     return board_tensor.flatten()

# def choose_move(env, policy_net):
#     '''
#     Choose a move using the policy network with temperature scaling.

#     Args:
#         env: Chess environment object
#         policy_net: Policy network model
#     '''

#     moves = env.get_legal_moves()
#     if not moves:
#         return None
    
#     # Get state representation
#     state = encode_fen(env.board.fen())
#     state = state.unsqueeze(0)
    
#     # Get action probabilities with numerical stability
#     with torch.no_grad():
#         logits = policy_net(state)
#         # Add temperature scaling for numerical stability
#         logits = logits / 10.0  # Reduce extreme values
#         probs = nn.Softmax(dim=1)(logits)
    
#     # Create move lookup with validation
#     move_probs = []
#     for move in moves:
#         move_idx = (move.from_square * 64 + move.to_square)
#         if move_idx < 4096:  # Ensure index is within bounds
#             prob = probs[0][move_idx].item()
#             if prob > 0 and not np.isnan(prob) and not np.isinf(prob):
#                 move_probs.append((move, prob))
    
#     # Fallback to random move if no valid probabilities
#     if not move_probs:
#         return random.choice(moves)
    
#     # Safe probability normalization
#     moves, probs = zip(*move_probs)
#     probs = torch.tensor(probs)
#     probs = torch.clamp(probs, min=1e-10)  # Prevent zero probabilities
#     probs = probs / probs.sum()  # Normalize
    
#     try:
#         move_idx = torch.multinomial(probs, 1).item()
#         return moves[move_idx]
#     except RuntimeError:
#         # Fallback to random choice if sampling fails
#         return random.choice(moves)

def train_ppo(trajectory, policy_net, value_net, optimizer_p, optimizer_v):
    '''
    Train the policy and value networks using PPO.
    
    Args:
        trajectory: Dictionary containing trajectory data
        policy_net: Policy network model
        value_net: Value network model
        optimizer_p: Policy network optimizer
        optimizer_v: Value network optimizer
    '''

    try:
        states = trajectory['states']
        actions = trajectory['actions']
        rewards = trajectory['rewards']
        old_values = trajectory['values']
        old_log_probs = trajectory['log_probs']
        
        # Calculate returns and advantages with numerical stability
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + config['gamma'] * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        for _ in range(config['k_epochs']):
            try:
                # Get current policy predictions with stability measures
                logits = policy_net(states)
                logits = logits / 10.0  # Temperature scaling
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
                
                # Ensure no NaN values
                if torch.isnan(logits).any():
                    logging.error("NaN values detected in logits")
                    continue
                
                # Calculate probabilities with numerical stability
                logits = logits - logits.logsumexp(dim=-1, keepdim=True)
                probs = torch.softmax(logits, dim=-1)
                
                # Create distribution only if probabilities are valid
                if not torch.isnan(probs).any() and not torch.isinf(probs).any():
                    dist = torch.distributions.Categorical(probs=probs)
                else:
                    logging.error("Invalid probabilities detected")
                    continue
                
                current_log_probs = dist.log_prob(actions)
                current_values = value_net(states).squeeze()
                
                # Calculate ratios and advantages
                ratios = torch.exp(current_log_probs - old_log_probs)
                ratios = torch.clamp(ratios, 0.0, 10.0)
                
                advantages = returns - old_values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Calculate losses
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-config['epsilon_clip'], 1+config['epsilon_clip']) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(current_values, returns)
                
                # Update networks if losses are valid
                if not torch.isnan(policy_loss) and not torch.isnan(value_loss):
                    optimizer_p.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
                    optimizer_p.step()
                    
                    optimizer_v.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=0.5)
                    optimizer_v.step()
                    
                    logging.info(f"Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")
                else:
                    logging.error("NaN losses detected")
                    
            except Exception as e:
                logging.error(f"Error in PPO update: {str(e)}")
                logging.error(traceback.format_exc())
                continue
                
    except Exception as e:
        logging.error(f"Error in train_ppo: {str(e)}")
        logging.error(traceback.format_exc())
        raise

# def save_game(board, result, game_index):
#     '''
#     Save the game as a PGN file.
    
#     Args:
#         board: Chess board object
#         result: Game result string
#         game_index: Index of the game
#     '''

#     game = chess.pgn.Game()
    
#     # Add game metadata
#     game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
#     game.headers["White"] = "ChessPPO"
#     game.headers["Black"] = "ChessPPO"
#     game.headers["Result"] = result
#     game.headers["Event"] = f"Self-play game {game_index}"
#     game.headers["Round"] = str(game_index)
    
#     # Add moves
#     node = game
#     for move in board.move_stack:
#         node = node.add_variation(move)
    
#     return game

def self_play_episode(env, policy_net, value_net, optimizer_p, optimizer_v, game_index):
    '''
    Play a single self-play game episode and train the policy network.
    
    Args:
        env: Chess environment object
        policy_net: Policy network model
        value_net: Value network model
        optimizer_p: Policy network optimizer
        optimizer_v: Value network optimizer
        game_index: Index of the game

    Returns:
        trajectory: Dictionary containing trajectory data
    '''

    states, actions, rewards, values, log_probs = [], [], [], [], []
    
    if config['print_self_play']:
        print(f"\nStarting game {game_index + 1}")
    
    # reset the environment for the start of a new game
    obs = env.reset()
    done = False
    episode_reward = 0
    
    while not done and len(actions) < config['max_moves']:
        state = encode_fen(env.board.fen())
        
        # Get action from policy
        with torch.no_grad():
            move = choose_move(env, policy_net)
            if move is None:
                break
                
            # Store state and action
            state_tensor = state.unsqueeze(0)
            value = value_net(state_tensor)
            logits = policy_net(state_tensor)
            action_idx = move.from_square * 64 + move.to_square
            
            # Store trajectory information
            states.append(state)
            actions.append(action_idx)
            values.append(value.item())
            log_prob = nn.LogSoftmax(dim=1)(logits)[0][action_idx]
            log_probs.append(log_prob.item())
            
            # Make move
            obs, reward, done = env.step(move)
            rewards.append(reward)
            episode_reward += reward
            
    trajectory = {
        'states': torch.stack(states),
        'actions': torch.tensor(actions),
        'rewards': torch.tensor(rewards),
        'values': torch.tensor(values),
        'log_probs': torch.tensor(log_probs),
        'episode_reward': episode_reward
    }
    
    # Add PPO training step
    train_ppo(trajectory, policy_net, value_net, optimizer_p, optimizer_v)
    
    # Before returning trajectory, save the game
    result = "1/2-1/2"  # Default to draw
    if done:
        if env.board.result() == "1-0":
            result = "1-0"
        elif env.board.result() == "0-1":
            result = "0-1"
    else:
        result = "*"  # Incomplete game
        
    game = save_game(env.board, result, game_index)
    
    # Save game - Simplified filename structure
    timestamp = datetime.now().strftime("%Y%m%d")
    batch_num = game_index // config['games_per_file']
    filename = f"games_{timestamp}_batch{batch_num}.pgn"
    filepath = os.path.join(config['games_dir'], filename)
    
    # Append game to file
    mode = 'a' if os.path.exists(filepath) else 'w'
    with open(filepath, mode) as f:
        print(game, file=f, end="\n\n")
    
    return trajectory

# def setup_games_directory():
#     '''
#     Create the games directory if it doesn't exist.
#     '''

#     if not os.path.exists(config['games_dir']):
#         os.makedirs(config['games_dir'])

# def plot_training_progress(metrics, save_path='training_progress.png'):
#     '''
#     Plot training progress using metrics data.

#     Args:
#         metrics: TrainingMetrics object
#         save_path: File path to save the plot
#     '''

#     if not metrics.episode_rewards:
#         logging.warning("No metrics to plot - skipping plot generation")
#         return
        
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
#     # Smooth metrics with validation
#     window = min(config['metrics_smoothing'], len(metrics.episode_rewards))
#     if window < 1:
#         window = 1
        
#     # Plot episode rewards if data exists
#     if len(metrics.episode_rewards) >= window:
#         smooth_rewards = np.convolve(metrics.episode_rewards, 
#                                    np.ones(window)/window, mode='valid')
#         ax1.plot(smooth_rewards)
#     ax1.set_title('Average Episode Reward')
#     ax1.set_xlabel('Episode')
#     ax1.set_ylabel('Reward')
    
#     # Plot game lengths if data exists
#     if len(metrics.game_lengths) >= window:
#         smooth_lengths = np.convolve(metrics.game_lengths,
#                                    np.ones(window)/window, mode='valid')
#         ax2.plot(smooth_lengths)
#     ax2.set_title('Average Game Length')
#     ax2.set_xlabel('Episode')
#     ax2.set_ylabel('Moves')
    
#     # Plot material advantage if data exists
#     if len(metrics.material_advantages) >= window:
#         smooth_material = np.convolve(metrics.material_advantages,
#                                     np.ones(window)/window, mode='valid')
#         ax3.plot(smooth_material)
#     ax3.set_title('Average Material Advantage')
#     ax3.set_xlabel('Episode')
#     ax3.set_ylabel('Material Score')
    
#     # Plot win rate
#     if metrics.win_rates:
#         ax4.plot(metrics.win_rates)
#     ax4.set_title('Win Rate')
#     ax4.set_xlabel('Episode')
#     ax4.set_ylabel('Win Rate')
    
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()

def evaluate_model(env, policy_net, value_net, num_games=config['eval_games']):
    '''
    Evaluate the model performance over a number of games.
    
    Args:
        env: Chess environment object
        policy_net: Policy network model
        value_net: Value network model
        num_games: Number of games to evaluate
        
    Returns:
        metrics: TrainingMetrics object
    '''
    
    metrics = TrainingMetrics()
    
    for game in range(num_games):
        obs = env.reset()  # Reset only at start of each evaluation game
        done = False
        episode_reward = 0
        moves = 0
        
        while not done and moves < config['max_moves']:
            move = choose_move(env, policy_net)
            if move is None:
                break
            
            obs, reward, done = env.step(move)
            episode_reward += reward
            moves += 1
        
        # Get final material advantage and result
        material_score = env.calculate_reward()[0]
        result = env.board.result() if env.board.is_game_over() else "*"
        
        metrics.update(episode_reward, moves, material_score, result)
    
    return metrics

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

def main():
    '''
    Main training loop for Chess PPO.
    '''

    try:
        setup_games_directory(config['games_dir'])
        if not os.path.exists(config['metrics_dir']):
            os.makedirs(config['metrics_dir'])
        
        env = ChessEnv(config)
        
        # Load or create models
        policy_net = load_or_create_model(ChessPolicy, config['load_model_path'], config['device'])
        value_net = load_or_create_model(ChessValue, config['load_model_path'].replace('checkpoint', 'value'), config['device'])
        
        optimizer_p = optim.Adam(policy_net.parameters(), lr=config['lr'])
        optimizer_v = optim.Adam(value_net.parameters(), lr=config['lr'])
        
        print("\nStarting training process...")
        print(f"Total games to play: {config['self_play_games']}")
        
        # Initial evaluation
        print("\nEvaluating initial model performance...")
        initial_metrics = evaluate_model(env, policy_net, value_net)
        print(f"Initial evaluation complete - Win rate: {initial_metrics.win_rates[-1]:.2f}")
        
        # Training metrics
        training_metrics = TrainingMetrics()
        
        # Training loop
        print("\nStarting training games...")
        for game_index in range(config['self_play_games']):
            try:
                # Print progress every 10 games
                if game_index % 10 == 0:
                    progress = (game_index) / config['self_play_games'] * 100
                    print(f"\nGame {game_index}/{config['self_play_games']} ({progress:.1f}% complete)")
                    if training_metrics.win_rates:
                        print(f"Current win rate: {training_metrics.win_rates[-1]:.2f}")
                        print(f"Games played: {sum(training_metrics.win_draw_loss)}")
                
                # Play game and get trajectory
                trajectory = self_play_episode(env, policy_net, value_net, 
                                            optimizer_p, optimizer_v, game_index)
                
                # Update metrics
                material_score = env.calculate_reward()[0]
                result = env.board.result() if env.board.is_game_over() else "*"
                training_metrics.update(
                    trajectory['episode_reward'],
                    len(trajectory['actions']),
                    material_score,
                    result
                )
                
                logging.info(f"Game {game_index} completed successfully")
                
            except Exception as e:
                logging.error(f"Error in game {game_index}: {str(e)}")
                logging.error(traceback.format_exc())
                continue
        
        # Final evaluation and plotting only if we have data
        if sum(training_metrics.win_draw_loss) > 0:
            if config['plot_metrics']:
                plot_training_progress(training_metrics, config['metrics_smoothing'])
                metrics_file = training_metrics.save_metrics(config['metrics_dir'])
                
                print("\nTraining Summary:")
                print(f"Total games completed: {sum(training_metrics.win_draw_loss)}")
                print(f"Wins: {training_metrics.win_draw_loss[0]}")
                print(f"Draws: {training_metrics.win_draw_loss[1]}")
                print(f"Losses: {training_metrics.win_draw_loss[2]}")
                print(f"\nDetailed metrics saved to: {metrics_file}")
        else:
            print("\nNo games completed successfully - no metrics to plot")
            
    except Exception as e:
        logging.error(f"Fatal error in main: {str(e)}")
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()

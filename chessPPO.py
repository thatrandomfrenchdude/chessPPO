import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime
import logging
import traceback

from src.ChessEnv import ChessEnv, save_game, setup_games_directory, setup_metrics_directory
from src.PPOModels import ChessPolicy, ChessValue, encode_fen, choose_move, load_or_create_model
from src.TrainingMetrics import TrainingMetrics, plot_training_progress, final_evaluation

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

def training_loop(
    env,
    policy_net,
    value_net,
    optimizer_p,
    optimizer_v,
    training_metrics
):
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

def main():
    '''
    Main training loop for Chess PPO.
    '''

    try:
        # create directories if they don't exist
        setup_games_directory(config['games_dir'])
        setup_metrics_directory(config['metrics_dir'])
        
        # initialize the environment
        env = ChessEnv(config)
        
        # TODO: review these
        # load or create the ppo policy networks
        policy_net = load_or_create_model(ChessPolicy, config['load_model_path'], config['device'])
        value_net = load_or_create_model(ChessValue, config['load_model_path'].replace('checkpoint', 'value'), config['device'])
        
        optimizer_p = optim.Adam(policy_net.parameters(), lr=config['lr'])
        optimizer_v = optim.Adam(value_net.parameters(), lr=config['lr'])
        
        print("\nStarting training process...")
        print(f"Total games to play: {config['self_play_games']}")
        
        # TODO: review these
        # Training metrics
        # initialize a set of training metrics for the training session
        training_metrics = TrainingMetrics()
        
        # run the ppo training loop
        # trains the agent in the environment
        training_loop(
            env,
            policy_net,
            value_net,
            optimizer_p,
            optimizer_v,
            training_metrics
        )
        
        # evaluate the agent performance during training session
        final_evaluation(
            training_metrics,
            config['plot_metrics'],
            config['metrics_smoothing'],
            config['metrics_dir']
        )
            
    except Exception as e:
        logging.error(f"Fatal error in main: {str(e)}")
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()

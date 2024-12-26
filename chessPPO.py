from datetime import datetime
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import traceback
import yaml
from pathlib import Path

from src.ChessEnv import ChessEnv, save_game, setup_games_directory, setup_metrics_directory
from src.PPOModels import ChessPolicy, ChessValue, encode_fen, choose_move, load_or_create_model
from src.TrainingMetrics import TrainingMetrics, final_evaluation

def load_config():
    """Load configuration from yaml file"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Flatten config for backwards compatibility
    flat_config = {}
    flat_config.update(config['ppo'])
    flat_config.update(config['paths'])
    flat_config.update(config['evaluation'])
    flat_config.update(config['agent'])
    
    return flat_config

# Load configuration
config = load_config()

##### LOGGING #####
logging.basicConfig(
    filename=f'logs/chess_ppo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
##### END LOGGING #####

def training_loop(
    env,
    policy_net,
    value_net,
    optimizer_p,
    optimizer_v,
    training_metrics
):
    print("\nStarting training games...")
    
    for game_index in range(config['self_play_games']):
        try:
            # Progress reporting
            if game_index % 10 == 0:
                progress = (game_index) / config['self_play_games'] * 100
                print(f"\nGame {game_index}/{config['self_play_games']} ({progress:.1f}% complete)")
                if training_metrics.win_rates:
                    print(f"Current win rate: {training_metrics.win_rates[-1]:.2f}")
                    print(f"Games played: {sum(training_metrics.win_draw_loss)}")

            # Game initialization
            if config['print_self_play']:
                print(f"\nStarting game {game_index + 1}")
            
            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            states, actions, rewards, values, log_probs = [], [], [], [], []

            # Play single game
            while not done and len(actions) < config['max_moves']:
                state = encode_fen(env.board.fen())
                
                with torch.no_grad():
                    move = choose_move(env, policy_net)
                    if move is None:
                        break
                    
                    state_tensor = state.unsqueeze(0)
                    value = value_net(state_tensor)
                    logits = policy_net(state_tensor)
                    action_idx = move.from_square * 64 + move.to_square
                    
                    states.append(state)
                    actions.append(action_idx)
                    values.append(value.item())
                    log_prob = nn.LogSoftmax(dim=1)(logits)[0][action_idx]
                    log_probs.append(log_prob.item())
                    
                    obs, reward, done = env.step(move)
                    rewards.append(reward)
                    episode_reward += reward

            # Prepare trajectory data
            trajectory = {
                'states': torch.stack(states),
                'actions': torch.tensor(actions),
                'rewards': torch.tensor(rewards),
                'values': torch.tensor(values),
                'log_probs': torch.tensor(log_probs),
                'episode_reward': episode_reward
            }

            # PPO Training
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + config['gamma'] * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32)
            
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            for _ in range(config['k_epochs']):
                try:
                    # Forward pass
                    logits = policy_net(trajectory['states'])
                    logits = logits / 10.0  # Temperature scaling
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
                    
                    if torch.isnan(logits).any():
                        logging.error("NaN values detected in logits")
                        continue
                    
                    logits = logits - logits.logsumexp(dim=-1, keepdim=True)
                    probs = torch.softmax(logits, dim=-1)
                    
                    if not torch.isnan(probs).any() and not torch.isinf(probs).any():
                        dist = torch.distributions.Categorical(probs=probs)
                    else:
                        logging.error("Invalid probabilities detected")
                        continue
                    
                    # Calculate policy updates
                    current_log_probs = dist.log_prob(trajectory['actions'])
                    current_values = value_net(trajectory['states']).squeeze()
                    
                    ratios = torch.exp(current_log_probs - trajectory['log_probs'])
                    ratios = torch.clamp(ratios, 0.0, 10.0)
                    
                    advantages = returns - trajectory['values']
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    
                    # Calculate losses
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1-config['epsilon_clip'], 1+config['epsilon_clip']) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = nn.MSELoss()(current_values, returns)
                    
                    # Update networks
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

            # Save game and update metrics
            material_score = env.calculate_reward()[0]
            result = env.board.result() if env.board.is_game_over() else "*"
            training_metrics.update(episode_reward, len(actions), material_score, result)
            
            save_game(env.board, result, game_index)
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
        
        # TODO: review this
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

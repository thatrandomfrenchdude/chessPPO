from datetime import datetime
import logging
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import traceback
import yaml
from pathlib import Path

from src.ChessEnv import ChessEnv, encode_state
from src.PPOModels import ChessPolicy, ChessValue, choose_move
from src.TrainingMetrics import TrainingMetrics, plot_training_progress

logging.basicConfig(
    filename=f'logs/chess_ppo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# make an app class from the code
class ChessPPOApp:
    def __init__(self):
        # you're on the clock!
        self.start_time = datetime.now()
        
        # load the configuration
        self.config = self.load_config()

        # load or create the ppo policy networks
        self.policy_net = self.load_or_create_model(ChessPolicy, self.config['load_model_path'])
        self.value_net = self.load_or_create_model(ChessValue, self.config['load_model_path'].replace('checkpoint', 'value'))

        # initialize the optimizers
        self.optimizer_p = optim.Adam(self.policy_net.parameters(), lr=self.config['lr']) # policy network optimizer
        self.optimizer_v = optim.Adam(self.value_net.parameters(), lr=self.config['lr']) # value network optimizer

        # initialize the environment
        self.env = ChessEnv(self.config)

        # initialize a set of training metrics for the app
        self.training_metrics = TrainingMetrics()

    def load_config(self):
        """Load configuration from yaml file"""
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        flat_config = {}
        flat_config.update(config['ppo'])
        flat_config.update(config['paths'])
        flat_config.update(config['metrics'])
        flat_config.update(config['agent'])

        return flat_config

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
    
    def setup(self):
        '''
        Pre-run setup for Chess PPO.
        '''
        # create directories if they don't exist
        if not os.path.exists(self.config['games_dir']):
            os.makedirs(self.config['games_dir'])
        # setup_games_directory(self.config['games_dir'])
        if not os.path.exists(self.config['metrics_dir']):
            os.makedirs(self.config['metrics_dir'])
        # setup_metrics_directory(self.config['metrics_dir'])
    
    def run(self) -> None:
        try:
            # setup the app
            self.setup()

            # print pre-training information
            print("\nStarting training process...")
            print(f"Total games to play: {self.config['self_play_games']}")
            print(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # trains the agent in the environment
            self.training_loop()
            
            # print post-training information
            end_time = datetime.now()
            total_runtime = end_time - self.start_time
            print(f"\nTraining completed!")
            print(f"Total runtime: {total_runtime}")
            print(f"Average games per second: {self.config['self_play_games'] / total_runtime.total_seconds():.2f}")
            
            # evaluate the agent performance during training session
            # TODO: review this
            self.final_evaluation()
            # self.final_evaluation(
            #     self.training_metrics,
            #     self.config['plot_metrics'],
            #     self.config['metrics_smoothing'],
            #     self.config['metrics_dir']
            # )

            # save the agent models for future use
            torch.save(self.policy_net.state_dict(), self.config['save_model_path'])
            torch.save(self.value_net.state_dict(), self.config['save_model_path'].replace('checkpoint', 'value'))
            print(f"Models saved to {self.config['save_model_path']}")
                
        except Exception as e:
            end_time = datetime.now()
            logging.error(f"Fatal error in main after {end_time - self.start_time}: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def training_loop(self) -> None:
        print("\nStarting training games...")
        last_progress_time = datetime.now()  # Track time between progress reports
        
        # play the games
        for game_index in range(0, self.config['self_play_games']):
            # Game initialization
            if self.config['print_self_play']:
                print(f"\nStarting game {game_index + 1}")
            
            try:
                # print progress readout every 25 games
                if game_index > 0 and game_index % 25 == 0:
                    self.training_readout(game_index+1, last_progress_time)

                # play a game of chess, one from the total number of games
                total_game_rewards, num_actions = self.game_loop(game_index + 1)

                # Save game and update metrics
                # TODO: this is not working
                material_score = self.env.calculate_reward()[0]
                result = self.env.board.result() if self.env.board.is_game_over() else "*"
                self.training_metrics.update(total_game_rewards, num_actions, material_score, result)
                
                # save game to a pgn file
                self.env.save_game(result, game_index)
                
            except Exception as e:
                logging.error(f"Error in game {game_index}: {str(e)}")
                logging.error(traceback.format_exc())
                continue

            if self.config['print_self_play']:
                print(f"Game {game_index+1} completed successfully")

    def final_evaluation(self) -> None:
        # Final evaluation and plotting only if games were completed
        if sum(self.training_metrics.win_draw_loss) > 0:
            if self.config['plot_metrics']:
                plot_training_progress(self.training_metrics, self.config['metrics_smoothing'], self.config['metrics_dir'])
                metrics_file = self.training_metrics.save_metrics(self.config['metrics_dir'])
                
                print("\nTraining Summary:")
                print(f"Total games completed: {sum(self.training_metrics.win_draw_loss)}")
                print(f"Wins: {self.training_metrics.win_draw_loss[0]}")
                print(f"Draws: {self.training_metrics.win_draw_loss[1]}")
                print(f"Losses: {self.training_metrics.win_draw_loss[2]}")
                print(f"\nDetailed metrics saved to: {metrics_file}")
        else:
            print("\nNo games completed successfully - no metrics to plot")

    def game_loop(
        self,
        game_index: int
    ) -> tuple:
        '''
        Play a single game of chess using PPO with real-time learning.

        The game alternates between white and black moves. For each move:
        1. Current board state is encoded to tensor format
        2. Policy network selects a move (validated by chess engine)
        3. Value network estimates state value
        4. Move is executed and reward calculated
        5. PPO update is performed immediately after each move:
            - Calculates advantages and returns
            - Updates policy and value networks using clipped PPO loss
            - Applies gradient clipping for stability
        6. Game continues until checkmate, stalemate, or max moves reached
        
        Returns:
            - total_episode_reward (float): Total accumulated reward for the game
            - num_actions (int): Number of moves made in the game
        Notes:
             - Each move triggers a PPO update (online learning)
             - Moves are validated by chess engine before execution
             - Temperature scaling applied to logits for exploration
             - Includes safety checks for numerical stability
             - Max moves limit prevents infinite games
        '''
        # Reset environment
        obs, done = self.env.reset()
        total_episode_reward = 0 # tracking the reward for the overall episode
        states, actions, rewards, values, log_probs = [], [], [], [], [] # initialize lists for trajectory storage

        move_count = 1
        while not done and len(actions) < self.config['max_moves']:
            logging.info(f"\nMove {move_count + 1}")
            try:
                # 1. State Processing - get and encode the state for the policy network
                state = encode_state(obs['fen']) # creates a flattened 12x8x8 tensor (768 values)
                state_tensor = state.unsqueeze(0) # Add batch dimension: [1, 768]

                # TODO: add additional state information, for example piece positions, material scores
                # state = {
                #     'fen': obs['fen'],
                # }
                
                # 2. Move Selection - get move and log probability with the policy network
                with torch.no_grad():
                    # illegal moves are filtered out by the chess engine
                    moves = self.env.get_legal_moves()
                    if not moves:
                        break

                    # select move and get a value estimate
                    move = choose_move(
                        moves, # list of legal moves
                        self.policy_net, # policy network for move selection
                        state, # state tensor for the current board position
                        self.config['temperature'] # temperature scaling for exploration
                    )
                    value = self.value_net(state_tensor)

                    # Get action probability distribution
                    action_logits = self.policy_net(state_tensor)
                    action_logits = action_logits / self.config['temperature']
                    probs = nn.Softmax(dim=1)(action_logits)
                    
                    # Get move index and log probability
                    action_idx = move.from_square * 64 + move.to_square
                    action_log_prob = torch.log(probs[0][action_idx] + self.config['epsilon_clip'])  # Add small epsilon for numerical stability

                    # somewhere in here, need to evaluate the move to provide the reward
                    # integrate evaluation like this:
                    # https://blog.propelauth.com/chess-analysis-in-python/
                    
                # 3. Execute Move - apply the chosen move to the environment to collect the reward    
                obs, reward, done = self.env.step(move)

                try:
                    total_episode_reward += reward # track overall episode rewards
                except Exception as e:
                    print(reward)
                    sys.exit(f"Error in reward calculation: {str(e)}")


                # Store trajectory data for PPO update
                actions.append(action_idx) # Keep track of moves for max_moves limit
                states.append(state)
                rewards.append(reward)
                values.append(value)
                log_probs.append(action_log_prob)
                
                # 4. Prepare PPO Update Data
                # reward = [
                #     float,
                #     float,
                #     float,
                #     float,
                #     float,
                #     float
                # ]
                step_rewards = torch.tensor([reward], dtype=torch.float32).reshape(1, 1)  # Shape: [1,1]

                # 5. PPO Update - calculate policy and value losses and update networks
                for _ in range(self.config['k_epochs']):
                    # Forward pass
                    logits = self.policy_net(state_tensor)
                    logits = logits / self.config['temperature']
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
                    logits = logits - logits.logsumexp(dim=-1, keepdim=True)
                    
                    probs = torch.softmax(logits, dim=-1)
                    
                    if torch.isnan(probs).any() and torch.isinf(probs).any():
                        raise ValueError("Invalid probabilities detected")
                    
                    dist = torch.distributions.Categorical(probs=probs)
                    
                    # Calculate PPO objectives
                    current_log_prob = dist.log_prob(torch.tensor(action_idx))
                    current_value = self.value_net(state_tensor)
                    
                    ratio = torch.clamp(
                        torch.exp(current_log_prob - action_log_prob),
                        0.0,
                        10.0
                    ) # ratio represents how much the current policy differs from the old policy
                    
                    advantage = step_rewards - value
                    
                    # Calculate losses in the networks
                    ## Step 1: Policy Loss
                    ### Strategy: use surrogates to catch large policy updates
                    surrogate_1 = ratio * advantage # unbounded policy gradient
                    surrogate_2 = torch.clamp(
                        ratio,
                        1-self.config['epsilon_clip'],
                        1+self.config['epsilon_clip']
                    ) * advantage # unbounded (clipped) policy gradient
                    # final policy is min of surrogate_1 and surrogate_2
                    policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                    ## Step 2: Value Loss
                    value_loss = nn.MSELoss()(current_value, step_rewards)
                    
                    # Update networks if losses are valid
                    if not torch.isnan(policy_loss) and not torch.isnan(value_loss):
                        # update the policy network
                        self.optimizer_p.zero_grad()
                        policy_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
                        self.optimizer_p.step()
                        
                        # update the value network
                        self.optimizer_v.zero_grad()
                        value_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
                        self.optimizer_v.step()
                        
                        logging.info(f"Move {len(actions)}: Policy Loss={policy_loss.item():.4f}, Value Loss={value_loss.item():.4f}")
                    else:
                        if torch.isnan(policy_loss):
                            raise ValueError("Policy loss is NaN in PPO Update loop")
                        if torch.isnan(value_loss):
                            raise ValueError("Value loss is NaN in PPO Update loop")
                        
                move_count += 1
                logging.info(f"Completed move {move_count}. Total moves: {len(actions)}")
            
            except Exception as e:
                logging.error(f"Error in game loop: {str(e)}")
                logging.error(traceback.format_exc())
                sys.exit("Error in game loop")

        print(f"Game {game_index} ended after {move_count} moves. Final reward: {total_episode_reward}")
        return total_episode_reward, len(actions)

    def training_readout(
        self,
        game_index: int,
        last_progress_time: datetime
    ):
        '''
        Print training progress and metrics.
        
        Args:
            game_index: Index of the current game
            last_progress_time: Time of the last progress report
        '''
        current_time = datetime.now()
        if game_index > 0:  # Skip first iteration
            time_elapsed = current_time - last_progress_time
            games_per_second = 25 / time_elapsed.total_seconds()
            print(f"Time for last 25 games: {time_elapsed.total_seconds():.1f}s ({games_per_second:.2f} games/s)")
        last_progress_time = current_time
        
        progress = (game_index-1) / self.config['self_play_games'] * 100
        print(f"\nGame {game_index-1}/{self.config['self_play_games']} ({progress:.1f}% complete)")
        if self.training_metrics.win_rates:
            print(f"Current win rate: {self.training_metrics.win_rates[-1]:.2f}")
            print(f"Games played: {sum(self.training_metrics.win_draw_loss)}")

if __name__ == "__main__":
    # main()
    app = ChessPPOApp()
    app.run()


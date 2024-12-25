import os
from datetime import datetime
import numpy as np
import json
import matplotlib.pyplot as plt
import logging 

def plot_training_progress(
    metrics,
    metrics_smoothing,
    save_path='training_progress.png'
):
    '''
    Plot training progress using metrics data.

    Args:
        metrics: TrainingMetrics object
        save_path: File path to save the plot
    '''

    if not metrics.episode_rewards:
        logging.warning("No metrics to plot - skipping plot generation")
        return
        
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Smooth metrics with validation
    window = min(metrics_smoothing, len(metrics.episode_rewards))
    if window < 1:
        window = 1
        
    # Plot episode rewards if data exists
    if len(metrics.episode_rewards) >= window:
        smooth_rewards = np.convolve(metrics.episode_rewards, 
                                   np.ones(window)/window, mode='valid')
        ax1.plot(smooth_rewards)
    ax1.set_title('Average Episode Reward')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    
    # Plot game lengths if data exists
    if len(metrics.game_lengths) >= window:
        smooth_lengths = np.convolve(metrics.game_lengths,
                                   np.ones(window)/window, mode='valid')
        ax2.plot(smooth_lengths)
    ax2.set_title('Average Game Length')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Moves')
    
    # Plot material advantage if data exists
    if len(metrics.material_advantages) >= window:
        smooth_material = np.convolve(metrics.material_advantages,
                                    np.ones(window)/window, mode='valid')
        ax3.plot(smooth_material)
    ax3.set_title('Average Material Advantage')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Material Score')
    
    # Plot win rate
    if metrics.win_rates:
        ax4.plot(metrics.win_rates)
    ax4.set_title('Win Rate')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Win Rate')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

class TrainingMetrics:
    def __init__(self):
        self.episode_rewards = []
        self.game_lengths = []
        self.material_advantages = []
        self.win_rates = []
        self.win_draw_loss = [0, 0, 0]  # [wins, draws, losses]
        self.incomplete_games = 0
        self.timestamp = datetime.now().strftime("%d%m%y_%H%M%S")

    def update(self, episode_reward, game_length, final_material, result):
        self.episode_rewards.append(episode_reward)
        self.game_lengths.append(game_length)
        self.material_advantages.append(final_material)
        
        # Update win/draw/loss counts including incomplete games
        if result == "1-0":
            self.win_draw_loss[0] += 1
        elif result == "1/2-1/2":
            self.win_draw_loss[1] += 1
        elif result == "0-1":
            self.win_draw_loss[2] += 1
        elif result == "*":
            self.incomplete_games += 1
            # Count incomplete games as draws for win rate calculation
            self.win_draw_loss[1] += 1
        
        # Calculate win rate using all games
        total_games = sum(self.win_draw_loss)
        win_rate = self.win_draw_loss[0] / total_games if total_games > 0 else 0
        self.win_rates.append(win_rate)

    def save_metrics(self, metrics_dir='training-metrics'):
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)

        metrics_data = {
            'episode_rewards': self.episode_rewards,
            'game_lengths': self.game_lengths,
            'material_advantages': self.material_advantages,
            'win_rates': self.win_rates,
            'win_draw_loss': self.win_draw_loss,
            'incomplete_games': self.incomplete_games,
            'summary': {
                'final_win_rate': self.win_rates[-1] if self.win_rates else 0,
                'avg_reward': float(np.mean(self.episode_rewards)),
                'avg_game_length': float(np.mean(self.game_lengths)),
                'total_games': sum(self.win_draw_loss),
                'wins': self.win_draw_loss[0],
                'draws': self.win_draw_loss[1],
                'losses': self.win_draw_loss[2],
                'incomplete': self.incomplete_games
            }
        }

        filename = f"metrics_{self.timestamp}.json"
        filepath = os.path.join(metrics_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=4)

        return filepath
from datetime import datetime
import json

from dataclasses import dataclass

@dataclass
class GameMetrics:
    """
    Module to track the metrics of a Chess PPO bot training session.
    
    Stores the metrics of a training session.
    """
    def __init__(self):
        self.start = datetime.now()
        self.timestamps = []
        self.start_positions = []
        self.actions = []
        self.rewards = []
        self.end_positions = []
        self.dones = []
        self.result = None

    def save_step(
        self,
        timestamp,
        start_position,
        action,
        reward,
        end_position,
        done
    ):
        """
        Save the metrics for a single step in the game.
        """
        self.timestamps.append(timestamp)
        self.start_positions.append(start_position)
        self.actions.append(action)
        self.rewards.append(reward)
        self.end_positions.append(end_position)
        self.dones.append(done)

    def save_result(self, result):
        """
        Save the result of the game.
        """
        self.result = result

    def save(
        self,
        file_path
    ):
        """
        Save the metrics to a file.
        """
        # create the metrics json data
        metrics_data = {
            'start': self.start.strftime('%Y-%m-%d %H:%M:%S'),
            'timestamps': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in self.timestamps],
            'start_positions': [str(sp) for sp in self.start_positions],
            'actions': [str(a) for a in self.actions],
            'rewards': self.rewards,
            'end_positions': [str(ep) for ep in self.end_positions],
            'dones': self.dones,
            'result': self.result
        }

        # save the metrics data to a file
        with open(file_path, 'w') as f:
            json.dump(metrics_data, f, indent=4)
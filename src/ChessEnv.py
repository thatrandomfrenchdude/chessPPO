import chess
import chess.pgn
from datetime import datetime
import os

class ChessEnv:
    def __init__(self, config):
        self.board = chess.Board()
        self.config = config
        self.last_piece_count = self.count_pieces()
        print("Initialized ChessEnv with a fresh board.")
    
    def count_pieces(self):
        """Count total pieces on board"""
        return sum(1 for _ in self.board.piece_map())

    def reset(self):
        '''
        Reset the environment to a new game.
        
        Returns:
            observation: Initial observation of the environment
            done: Boolean indicating if the game is over
        '''
        self.board.reset()

        if self.config['print_self_play']:
            print("\n" + "="*50)
            print("Environment reset: new chess board set up.")
            print("\nInitial board position:")
            print(self.board)
            print("="*50 + "\n")
        
        return self.get_observation(), False

    # TODO: update step to provide done in env observations
    def step(self, move):

        if self.config['print_self_play']:
            print("\n" + "="*50)
            print(f"Current board position:")
            print(self.board)
            print(f"\nExecuting move: {move}")
        
        self.board.push(move)

        if self.config['print_self_play']:
            print("\nBoard after move:")
            print(self.board)
            print("="*50 + "\n")
        
        reward, done = self.calculate_reward()
        
        if self.config['print_self_play']:
            print(f"Reward: {reward}")
        
        return self.get_observation(), reward, done

    def get_legal_moves(self):
        return list(self.board.legal_moves)

    # TODO: improve the observations about the environment
    # to give more context to the policy network
    def get_observation(self) -> dict:
        # Simple serialization of board (could be improved)
        obs = {
            'fen': self.board.fen(),
            # 'done': self.board.is_game_over(),
        }
        return obs

    def calculate_reward(self):
        done = False
        piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0}
        position_values = {
            'P': [  # Pawn position values
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                [1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0],
                [0.5, 0.5, 1.0, 2.5, 2.5, 1.0, 0.5, 0.5],
                [0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0],
                [0.5, -0.5, -1.0, 0.0, 0.0, -1.0, -0.5, 0.5],
                [0.5, 1.0, 1.0, -2.0, -2.0, 1.0, 1.0, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ],
            'N': [  # Knight position values (existing)
                [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0],
                [-4.0, -2.0, 0.0, 0.0, 0.0, 0.0, -2.0, -4.0],
                [-3.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -3.0],
                [-3.0, 0.5, 1.5, 2.0, 2.0, 1.5, 0.5, -3.0],
                [-3.0, 0.0, 1.5, 2.0, 2.0, 1.5, 0.0, -3.0],
                [-3.0, 0.5, 1.0, 1.5, 1.5, 1.0, 0.5, -3.0],
                [-4.0, -2.0, 0.0, 0.5, 0.5, 0.0, -2.0, -4.0],
                [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0]
            ],
            'B': [  # Bishop position values
                [-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0],
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, -1.0],
                [-1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, -1.0],
                [-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0],
                [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
                [-1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, -1.0],
                [-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0]
            ],
            'R': [  # Rook position values
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
                [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                [0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0]
            ],
            'Q': [  # Queen position values
                [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0],
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0],
                [-0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -0.5],
                [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -0.5],
                [-1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0],
                [-1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -1.0],
                [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0]
            ],
            'K': [  # King position values (middlegame)
                [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
                [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
                [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
                [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
                [-2.0, -3.0, -3.0, -4.0, -4.0, -3.0, -3.0, -2.0],
                [-1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0],
                [2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0],
                [2.0, 3.0, 1.0, 0.0, 0.0, 1.0, 3.0, 2.0]
            ]
        }
        
        if self.board.is_game_over():
            done = True
            result = self.board.result()
            if result == "1-0":
                return 10.0, done
            elif result == "0-1":
                return -10.0, done
            else:
                return 0.0, done
        
        # Track piece captures
        current_pieces = self.count_pieces()
        pieces_taken = self.last_piece_count - current_pieces
        self.last_piece_count = current_pieces
        
        # Aggression modifier
        aggression = self.config.get('aggression', 0.0)
        capture_reward = pieces_taken * (1.0 + aggression)
        
        # Material advantage with aggression
        material_score = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = piece_values[piece.symbol().upper()]
                if piece.color:  # White
                    material_score += value * (1.0 + aggression if aggression > 0 else 1.0)
                else:  # Black
                    material_score -= value * (1.0 - aggression if aggression < 0 else 1.0)
        
        # Position score with aggression modifier
        position_score = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                piece_type = piece.symbol().upper()
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                if piece.color:  # White
                    pos_value = position_values[piece_type][7-rank][file]
                    position_score += pos_value * (1.0 - abs(aggression))  # Less positional focus when aggressive
                else:  # Black
                    pos_value = position_values[piece_type][rank][file]
                    position_score -= pos_value * (1.0 - abs(aggression))
        
        # Combined reward with aggression weighting
        total_score = (
            0.3 * material_score + 
            0.15 * position_score + 
            0.2 * capture_reward
        )
        
        return float(total_score), done


def save_game(board, result, game_index):
    '''
    Save the game as a PGN file.
    
    Args:
        board: Chess board object
        result: Game result string
        game_index: Index of the game
    '''

    game = chess.pgn.Game()
    
    # Add game metadata
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = "ChessPPO"
    game.headers["Black"] = "ChessPPO"
    game.headers["Result"] = result
    game.headers["Event"] = f"Self-play game {game_index}"
    game.headers["Round"] = str(game_index)
    
    # Add moves
    node = game
    for move in board.move_stack:
        node = node.add_variation(move)
    
    return game

def setup_games_directory(
    games_dir='games'
):
    '''
    Create the games directory if it doesn't exist.
    '''

    if not os.path.exists(games_dir):
        os.makedirs(games_dir)

def setup_metrics_directory(
    metrics_dir='training-metrics'
):
    '''
    Create the metrics directory if it doesn't exist.
    '''

    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
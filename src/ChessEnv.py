import chess
import chess.pgn

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
        self.board.reset()
        if self.config['print_self_play']:
            print("\n" + "="*50)
            print("Environment reset: new chess board set up.")
            print("\nInitial board position:")
            print(self.board)
            print("="*50 + "\n")
        # else:
        #     print("Environment reset: new chess board set up.")
        return self.get_observation()

    def step(self, move):
        if self.config['print_self_play']:
            print("\n" + "="*50)
            print(f"Current board position:")
            print(self.board)
            print(f"\nExecuting move: {move}")
            self.board.push(move)
            print("\nBoard after move:")
            print(self.board)
            print("="*50 + "\n")
        else:
            self.board.push(move)
        reward, done = self.calculate_reward()
        if self.config['print_self_play']:
            print(f"Reward: {reward}, Done: {done}")
        return self.get_observation(), reward, done

    def get_legal_moves(self):
        return list(self.board.legal_moves)

    def get_observation(self):
        # Simple serialization of board (could be improved)
        return self.board.fen()

    def calculate_reward(self):
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
            result = self.board.result()
            if result == "1-0":
                return 10.0, True
            elif result == "0-1":
                return -10.0, True
            else:
                return 0.0, True
        
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
        
        return total_score, False
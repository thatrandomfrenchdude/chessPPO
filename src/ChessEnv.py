import chess
import chess.pgn
from datetime import datetime
import os
import torch

class ChessEnv:
    PIECE_VALUES = {
        'P': 1,
        'N': 3,
        'B': 3.1, # Bishops are slightly better than knights
        'R': 5,
        'Q': 9,
        'K': 0
    }
    POSITION_VALUES = {
        'P': [  # Pawn position values
            [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0], # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
    
    def __init__(
        self,
        config: dict
    ) -> None:
        self.board = chess.Board()
        self.config = config

        # enable reward modifiers
        self.enable_material_score = config.get('enable_material_score', 1.0)
        self.enable_position_score = config.get('enable_position_score', 1.0)
        self.enable_mobility_score = config.get('enable_mobility_score', 1.0)
        self.enable_pawn_structure_score = config.get('enable_pawn_structure_score', 1.0)
        self.enable_center_control_score = config.get('enable_center_control_score', 1.0)
        self.enable_game_over_score = config.get('enable_game_over_score', 1.0)

        # self.last_piece_count = self.count_pieces()
        print("Initialized ChessEnv with a fresh board.")

    def save_game(
        self,
        result: str,
        game_index: int
    ):
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
        for move in self.board.move_stack:
            node = node.add_variation(move)
        
        return game
    
    # def count_pieces(self):
    #     """Count total pieces on board"""
    #     return sum(1 for _ in self.board.piece_map())

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
    
    def calculate_reward(self) -> tuple[list[float],bool]:
        done = False
        
        # Aggression modifier
        # aggression = self.config.get('aggression', 0.0)
        # capture_reward = pieces_taken * (1.0 + aggression)
        
        # Material advantage with aggression modifier
        material_score = self._normalized_material_score()
        
        # Position score with aggression modifier
        position_score = self._normalized_position_score()

        # Mobility score
        mobility_score = self._normalized_mobility_score()

        # Pawn structure score
        pawn_structure_score = self._normalized_pawn_structure_score()

        # Center control score
        center_control_score = self._normalized_center_control_score()
        
        # Game win/loss/draw score
        game_over_score, done = self._normalized_game_over_reward()

        # Calculate game phase weights
        early_game, \
        mid_game, \
        late_game = self._game_phase_weights()

        # Custom ratios
        material_score_ratio = 1.0 # 0.7 # 0.5
        position_score_ratio = 1.0 # 0.6 # 0.5
        game_over_score_ratio = 1.0 # 0.8 # 1.0
        mobility_score_ratio = 1.0 # 0.5
        pawn_structure_score_ratio = 1.0 # 0.5
        center_control_score_ratio = 1.0 # 0.5

        # Combined reward calculation with both phase scaling and custom ratios
        # TODO: need to better relate these -- I am consider implementing a third network to learn these optimized weights
        total_score = (
            self.enable_material_score * material_score_ratio * mid_game * material_score +
            self.enable_position_score * position_score_ratio * early_game * position_score +
            self.enable_game_over_score * game_over_score_ratio * late_game * game_over_score +
            self.enable_mobility_score * mobility_score_ratio * mid_game * mobility_score +
            self.enable_pawn_structure_score * pawn_structure_score_ratio * mid_game * pawn_structure_score +
            self.enable_center_control_score * center_control_score_ratio * early_game * center_control_score
        )

        # scores = [
        #     float(self.enable_material_score * material_score_ratio * mid_game * material_score),
        #     float(self.enable_position_score * position_score_ratio * early_game * position_score),
        #     float(self.enable_game_over_score * game_over_score_ratio * late_game * game_over_score),
        #     float(self.enable_mobility_score * mobility_score_ratio * mid_game * mobility_score),
        #     float(self.enable_pawn_structure_score * pawn_structure_score_ratio * mid_game * pawn_structure_score),
        #     float(self.enable_center_control_score * center_control_score_ratio * early_game * center_control_score)
        # ]

        # # check the turn and invert the score if it is not white's turn
        # if not self.board.turn:
        #     total_score = -total_score
        
        return float(total_score), done
        # return scores, done
    
    def _normalized_game_over_reward(self):
        '''
        Check if the game is over and return the result.
        
        Returns:
            score: normalized score based on game result
            result: Result of the game'''
        score = 0.0
        done = False

        if self.board.is_game_over():
            done = True
            result = self.board.result()

            # white wins
            if result == "1-0":
                score = 1.0 # white wins
            elif result == "0-1":
                score = -1.0 # black wins
            else:
                score = 0.0 # draw, still get points to encourage faster games
            
            # invert the score if it is not white's turn
            if not self.board.turn:
                score = -score

            return score, done

        # game is not over
        return score, done
    
    def _normalized_material_score(self):
        '''
        Calculate the material score of the board.
        
        Returns:
            material_score: Normalized material score of the board
        '''
        material_score = 0

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = self.PIECE_VALUES[piece.symbol().upper()]
                if piece.color:  # White
                    material_score += value # * (1.0 + aggression if aggression > 0 else 1.0)
                else:  # Black
                    material_score -= value # * (1.0 - aggression if aggression < 0 else 1.0)

        # normalize the material score
        material_score = material_score / 39.0

        return material_score
    
    def _normalized_position_score(self):
        '''
        Calculate the positional score of the board.
        
        Returns:
            position_score: Normalized positional score of the board
        '''
        position_score = 0

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                piece_type = piece.symbol().upper()
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                if piece.color:  # White
                    pos_value = self.POSITION_VALUES[piece_type][7-rank][file]
                    position_score += pos_value # * (1.0 - abs(aggression))  # Less positional focus when aggressive
                else:  # Black
                    pos_value = self.POSITION_VALUES[piece_type][rank][file]
                    position_score -= pos_value # * (1.0 - abs(aggression))

        # normalize the position score
        position_score = position_score / 39.0

        return position_score
    
    # TODO: rough implementation, not yet used in reward calculation
    def _normalized_mobility_score(self):
        '''
        Calculate the mobility score of the board.
        
        Returns:
            mobility_score: Normalized mobility score of the board
        '''
        mobility_score = 0

        # Get legal moves for the current player
        legal_moves = len(list(self.board.legal_moves))

        # Get legal moves for the opponent
        # TODO: clean this up to avoid pushing and popping moves
        # self.board.push(chess.Move.null())
        # legal_moves_opponent = len(list(self.board.legal_moves))
        # self.board.pop()

        # Save current turn
        current_turn = self.board.turn
        
        # Switch turn to opponent
        self.board.turn = not current_turn
        legal_moves_opponent = len(list(self.board.legal_moves))
        
        # Restore original turn
        self.board.turn = current_turn

        # calculate the normalized mobility score
        mobility_score = (legal_moves - legal_moves_opponent) / (legal_moves + legal_moves_opponent)

        return mobility_score
    
    # TODO: rough implementation, not yet used in reward calculation
    def _normalized_pawn_structure_score(self):
        '''
        Calculate the pawn structure score of the board.
        
        Returns:
            pawn_structure_score: Normalized pawn structure score of the board
        '''
        pawn_structure_score = 0

        doubled_pawns = 0 # pawns in same file
        isolated_pawns = 0 # pawns with no adjacent pawns
        connected_pawns = 0 # pawns protected by other pawns

        # Check for doubled, isolated, and connected pawns
        for file in range(8):
            file_pawns = 0
            for rank in range(8):
                piece = self.board.piece_at(chess.square(file, rank))
                if piece and piece.symbol().upper() == 'P':
                    file_pawns += 1
            if file_pawns > 1:
                doubled_pawns += 1
            if file_pawns == 1:
                isolated_pawns += 1

        # calculate the normalized pawn structure score
        pawn_structure_score = (doubled_pawns + isolated_pawns + connected_pawns) / 24.0

        return pawn_structure_score
    
    # TODO: rough implementation, not yet used in reward calculation
    def _normalized_center_control_score(self):
        '''
        Calculate the center control score of the board.
        
        Returns:
            center_control: Normalized center control score of the board
        '''
        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        center_score = 0
        
        for square in center_squares:
            piece = self.board.piece_at(square)
            # Add points for piece occupation
            if piece:
                if piece.color == self.board.turn:
                    center_score += 1
                else:
                    center_score -= 1
            # Add points for attacks
            if self.board.is_attacked_by(self.board.turn, square):
                center_score += 0.5
            if self.board.is_attacked_by(not self.board.turn, square):
                center_score -= 0.5
                
        return center_score / 6.0  # Normalize score
    
    def _game_phase_weights(self):
        move_count = len(self.board.move_stack)

        early_game = max(0, 1 - move_count/30)
        late_game = min(1, (move_count-20)/20)
        mid_game = 1 - min(early_game + late_game, 1)
        
        return early_game, mid_game, late_game

def encode_state(fen) -> torch.Tensor:
    '''
    Create a binary vector representation of the chess position.

    Args:
        fen: FEN string of the board position

    Returns:
        board_tensor: Tensor representation of the board
    '''
    
    piece_map = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                 'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    
    board_tensor = torch.zeros(12, 8, 8)  # 12 piece types, 8x8 board
    
    # Parse the FEN string
    board_str = fen.split(' ')[0]
    row, col = 0, 0
    
    for char in board_str:
        if char == '/':
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        else:
            board_tensor[piece_map[char]][row][col] = 1
            col += 1
            
    return board_tensor.flatten()
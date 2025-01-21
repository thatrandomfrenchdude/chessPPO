import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ChessGame import ChessGame

def test_ChessGame():
    game = ChessGame(100)
    assert game is not None
    assert type(game) == ChessGame

def test_get_position():
    game = ChessGame(100)
    position = game.get_position()
    assert position == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

def test_get_reward():
    pass

def test_get_done_new_game():
    # new game
    game = ChessGame(100)
    done = game.get_done()
    assert done == False

def test_get_done_stalemate():
    # stalemate
    game = ChessGame(100)
    # moves to shortest known stalemate
    moves = [
        "e2e3", "a7a5",
        "d1h5", "a8a6",
        "h5a5", "h7h5",
        "h2h4", "a6h6",
        "a5c7", "f7f6",
        "c7d7", "e8f7",
        "d7b7", "d8d3",
        "b7b8", "d3h7",
        "b8c8", "f7g6",
        "c8e6"
    ]
    for move in moves:
        _, _, done = game.step(game.get_position(), move)
    assert done == True
    result = game.get_result()
    assert result == "1/2-1/2"

def test_get_done_checkmate():
    # checkmate
    game = ChessGame(100)
    # moves to shortest known checkmate
    moves = [
        "f2f3", "e7e5",
        "g2g4", "d8h4#"
    ]
    for move in moves:
        _, _, done = game.step(game.get_position(), move)
    assert done == True
    result = game.get_result()
    assert result == "0-1"

def test_get_done_out_of_moves():
    # draw - out of moves
    game = ChessGame(6)
    moves = [
        "e2e3", "a7a5",
        "d1h5", "a8a6",
        "h5a5", "h7h5"
    ]
    for move in moves:
        _, _, done = game.step(game.get_position(), move)
    assert done == True
    result = game.get_result()
    assert result == "*"
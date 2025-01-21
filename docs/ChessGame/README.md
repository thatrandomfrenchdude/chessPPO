# ChessGame Documentation
This file contains documentation for the `ChessGame` module. The ChessGame module represents the chess environment with reward calculation and game state management.

**Quick Jump:**
- [ChessGame Values](#chessgame-values)
- [ChessGame Functions](#chessgame-functions)
- [Developer Notes](#developer-notes)

## ChessGame Values
- Values
    - `board`: the game environment
    - `moves`: list of moves that have been played

## ChessGame Functions
- Functions
    - `save`: Save the game to a PGN file
    - `get_postion`: Get the current board as fen string
    - `get_rewards`: Get the rewards for a move
    - `get_done`: Check if the game is over
    - `get_results`: Get the game result
    - `step`: Play a move, update the game state, and return the new state, reward, and done flag
    - `calc_position_value`: Calculate the value of the current position

## Developer Notes
- cases represent the color of the piece
    - upper -> white
    - lower -> black
- all char/string evals are cast to capital letters

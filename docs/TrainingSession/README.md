# TrainingSession Documentation
This file contains documentation for the `TrainingSession` module. The TrainingSession module represents the training loop with self-play.

**Quick Jump:**
- [TrainingSession Values](#trainingsession-values)
- [TrainingSession Functions](#trainingsession-functions)
- [Developer Notes](#developer-notes)

## TrainingSession Values
- Values
    - `bot`: The ChessPPOBot agent
    - `session_length`: Number of games to play
    - `save_session`: Save the session to files
    - `save_models`: Save the bot models to files
    - `session_dir`: Directory to save the session files
    - `games_dir`: Directory to save the pgn game files
    - `metrics_dir`: Directory to save the game metric files

## TrainingSession Functions
- Functions
    - `run`: Run the training session
    - `save`: Save the session to files
    - `plot_session`: Plot the session

## Developer Notes
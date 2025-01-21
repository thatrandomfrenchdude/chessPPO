# GameMetrics Documentation
This file contains documentation for the `GameMetrics` module. The GameMetrics module tracks game statistics for a chess game.

**Quick Jump:**
- [GameMetrics Values](#gamemetrics-values)
- [Functions](#module-functions)
- [Developer Notes](#developer-notes)

## GameMetrics Values
- Values
    - `start`: game start time
    - `timestamps`: list of timestamps for each step
    - `start_positions`: starting positions for each step
    - `actions`: list of moves played at each step
    - `rewards`: list of rewards for each step
    - `end_positions`: ending positions for each step
    - `dones`: list of done flags for each step
    - `result`: game result

## GameMetrics Functions
- Functions
    - `save_step`: Save the current game step
    - `save_result`: Save the game result
    - `save`: Save the game metrics to a JSON file

## Developer Notes
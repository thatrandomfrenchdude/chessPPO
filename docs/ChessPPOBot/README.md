# ChessPPOBot Documentation
This file contains documentation for the `ChessPPOBot` module. The ChessPPOBot module represents the PPO agent with policy and value networks.

**Quick Jump:**
- [ChessPPOBot Values](#chessppobot-values)
- [ChessPPOBot Functions](#chessppobot-functions)
- [Developer Notes](#developer-notes)

## ChessPPOBot Values
- Values
    - `bot_name`
    - `Actor`: 768 -> 1024 -> 2048 -> 4096
    - `Critic`: 768 -> 1024 -> 512 -> 256 -> 64 -> 1
    - `actor_optimizer`
    - `critic_optimizer`

## ChessPPOBot Functions
- Functions
    - `load_or_create_model`: Load an existing model from file or create one if not found
    - `choose_move`: Choose a move based on the given state using the actor network
    - `evaluate_move`: Evaluate the chosen move using the critic network
    - `update`: Update the actor and critic networks
    - `save_bot`: Save the bot models to files

## Developer Notes
# Splendor/RL/training.py

import json
import numpy as np
import os

from Environment.Splendor_components.Player_components.strategy import ( # type: ignore
    BestStrategy, RandomStrategy, OffensiveStrategy, ResourceHog, ObliviousStrategy
)
from RL import RLAgent # type: ignore


def train_agent(base_save_path, log_path, layer_sizes, model_paths=None):
    from Environment.game import Game # type: ignore
    
    # Players and strategies (BestStrategy for training perfectly)
    player1_model = RLAgent(layer_sizes, model_paths[0])
    player2_model = RLAgent(layer_sizes, model_paths[1])
    
    players = [
        ('Player1', BestStrategy(), 1, player1_model, 0),
        ('Player2', BestStrategy(), 1, player2_model, 1)
    ]

    # Training loop
    for episode in range(100):  # Number of games
        logging = False
        game = Game(players)
        state = np.array(game.to_vector())

        # Log every 10 games
        if episode % 10 == 0: # LOGGING ALL EPISODES
            log_state = open(os.path.join(log_path, f"game_states_episode_{episode}.json"), 'w')
            log_move = open(os.path.join(log_path, f"moves_episode_{episode}.json"), 'w')
            logging = True

        while not game.victor and game.half_turns < 350:
            game.turn()  # Take a turn
            next_state = np.array(game.to_vector())

            # if logging: # LOGGING ALL EPISODES
            json.dump(game.get_state(), log_state)
            log_state.write('\n')
            log_move.write(str(game.active_player.chosen_move) + '\n')

            # Agent remembers
            active_player = game.active_player
            if not active_player.entered_loop:
                active_player.rl_model.remember(
                    state, 
                    active_player.move_index, 
                    game.reward, 
                    next_state, 
                    1 if game.victor else 0
                )
            active_player.entered_loop = False

            # Update state
            state = next_state
        
        print(f"Episode {episode+1} game complete and took {game.half_turns} turns. Replaying.")

        # Replay
        for player in game.players:
            rl_model = player.rl_model
            reversed_memory = list(reversed(player.rl_model.memory))

            # Reward token takes leading up to card purchases
            # Find last purchase
            for i, (_, move_index, _, _, _) in enumerate(reversed_memory):
                if 15 <= move_index < 45:
                    first_purchase_index = i
                    break

            count_since_purchase = 0
            reward = 1
            for state, move_index, _, next_state, _ in reversed_memory[first_purchase_index+1:]:
                if move_index >= 15 and move_index < 45: # Card puchases
                    count_since_purchase = 0
                elif move_index < 15: # Reward token takes
                    if count_since_purchase < 4:
                        rl_model.train(state, move_index, reward, next_state, 0)
                        if 5 <= move_index < 10:
                            reward *= .8
                            count_since_purchase += 1
                        else:
                            reward *= .9
                            count_since_purchase += 0.5
            
            # Regular replay
            player.rl_model.replay()

            # Winner/loser replay
            final_reward = 0.1 if player == game.victor else -0.1
            indices = np.random.permutation(len(player.rl_model.memory))
            for i in indices:
                state, move_index, reward, next_state, done = player.rl_model.memory[i]
                player.rl_model.train(state, move_index, final_reward, next_state, done)

            # Decay epsilon
            if player.rl_model.epsilon > player.rl_model.epsilon_min:
                player.rl_model.epsilon *= player.rl_model.epsilon_decay
        
        # Close logs
        if logging:
            log_state.close()
            log_move.close()

        # Log the progress
        print(f"Episode {episode+1} replayed")

        # Save models
        if episode%5 == 0:
            for player in game.players:
                player.rl_model.save_model(base_save_path, player.name)

    # Save models
    for player in game.players:
        player.rl_model.save_model(base_save_path, player.name)
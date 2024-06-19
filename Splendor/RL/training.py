# Splendor/RL/training.py

import json
import numpy as np
import os
import random

from Environment.Splendor_components.Player_components.strategy import ( # type: ignore
    BestStrategy, RandomStrategy, OffensiveStrategy, ResourceHog, ObliviousStrategy
)
from Environment.game import Game # type: ignore
from RL import RLAgent # type: ignore


def debug_game(base_save_path, log_path, layer_sizes, model_path):
     # Players and strategies (BestStrategy for training perfectly)
    player1_model = RLAgent(layer_sizes, model_path)
    player2_model = RLAgent(layer_sizes, model_path)
    
    players = [
        ('Player1', BestStrategy(), 1, player1_model, 0),
        ('Player2', BestStrategy(), 1, player2_model, 1)
    ]

    for episode in range(1000):
        # Enable logging for all games
        log_state = open(os.path.join(log_path, f"game_states_episode_{episode}.json"), 'w')
        log_move = open(os.path.join(log_path, f"moves_episode_{episode}.json"), 'w')

        game = simulate_game(players, True, log_state, log_move)
        print(f"Simulated game {episode}")

def simulate_game(players, logging, log_state, log_move):
    game = Game(players)
    state = np.array(game.to_vector())

    while not game.victor and game.half_turns < 350:
        # Take a turn
        game.turn()
        next_state = np.array(game.to_vector())

        # Log states and moves
        if logging:
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
    
    return game

def priority_play(base_save_path, log_path, layer_sizes, model_path):
    """searches tons of games and selects the out 10%s for training"""
    # Players and strategies (BestStrategy for training perfectly)
    player1_model = RLAgent(layer_sizes, model_path)
    player2_model = RLAgent(layer_sizes, model_path)
    
    players = [
        ('Player1', BestStrategy(), 1, player1_model, 0),
        ('Player2', BestStrategy(), 1, player2_model, 1)
    ]

    n_128 = 1
    sims = 1_000
    victor_memory = []
    loser_memory = []
    for i in range(sims):
        game = simulate_game(players, False, None, None)
        print(f"simulated game {i}")
        for player in game.players:
            if player.victor:
                victor_memory.append(player.rl_model.memory)
            else:
                loser_memory.append(player.rl_model.memory)
        
    # Sort and select memory
    victor_memory.sort(key=len, reverse=True)
    loser_memory.sort(key=len, reverse=True)

    shortest_victor = random.sample(victor_memory[ :sims//5], 128*n_128)
    longest_loser = random.sample(loser_memory[4*sims//5: ], 128*n_128)
    print("Average lengths of 10 percents:", np.mean([len(mem) for mem in shortest_victor]), np.mean([len(mem) for mem in longest_loser]))

    # Flatten the memory
    flattened_victor_memory = [memory for game in shortest_victor for memory in game]
    flattened_loser_memory = [memory for game in longest_loser for memory in game]

    # Train on memories
    print("Training on memories")
    player1_model.train_batch(flattened_victor_memory, 5)
    player1_model.train_batch(flattened_loser_memory, -5)

    # Save model
    print("Saving model")
    player1_model.save_model(base_save_path, "priority_play")

def train_agent(base_save_path, log_path, layer_sizes, model_paths=None):
    # Players and strategies (BestStrategy for training perfectly)
    player1_model = RLAgent(layer_sizes, model_paths[0])
    player2_model = RLAgent(layer_sizes, model_paths[1])
    
    players = [
        ('Player1', BestStrategy(), 1, player1_model, 0),
        ('Player2', BestStrategy(), 1, player2_model, 1)
    ]
    
    logging = False

    # Training loop
    for episode in range(1):  # Number of games
        # Log every 10 games
        if episode % 10 == 0:
            log_state = open(os.path.join(log_path, f"game_states_episode_{episode}.json"), 'w')
            log_move = open(os.path.join(log_path, f"moves_episode_{episode}.json"), 'w')
            logging = True

        game = simulate_game(players, logging, log_state, log_move)
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
            reward = 3
            for state, move_index, _, next_state, _ in reversed_memory[first_purchase_index+1:]:
                if move_index >= 15 and move_index < 45: # Card puchases
                    count_since_purchase = 0
                elif move_index < 15: # Reward token takes
                    if count_since_purchase < 4:
                        rl_model.train_individual(state, move_index, reward, next_state, 0)
                        if 5 <= move_index < 10:
                            reward *= .8
                            count_since_purchase += 1
                        else:
                            reward *= .9
                            count_since_purchase += 0.5
            
            # Regular replay
            player.rl_model.replay()

            # Winner/loser replay
            final_reward = 10 if player == game.victor else -10
            indices = np.random.permutation(len(player.rl_model.memory))
            for i in indices:
                state, move_index, reward, next_state, done = player.rl_model.memory[i]
                player.rl_model.train_individual(state, move_index, final_reward, next_state, done)

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
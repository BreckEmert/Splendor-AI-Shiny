# Splendor/Environment/Splendor_components/player.py

from itertools import combinations
import numpy as np


class Player:
    def __init__(self, name, strategy, strategy_strength, rl_model, turn_order_index):
        self.name: str = name
        self.turn_order_index = turn_order_index # Unused
        self.state_offset = 156 + 45*turn_order_index

        self.gems: np.ndarray = np.zeros(6, dtype=int)
        self.cards: np.ndarray = np.zeros(5, dtype=int)
        self.reserved_cards: list = []
        self.points: int = 0

        self.card_ids = [[[], [], [], [], []], [[], [], [], [], []], [[], [], [], [], []], [[], [], [], [], []]]
        self.rl_model = rl_model
        self.entered_loop = False
        self.victor = False
        #self.strategy: strategy = strategy
        #self.strategy_strength: int = strategy_strength
    
    def take_or_spend_gems(self, gems_to_change):
        self.gems += np.pad(gems_to_change, (0, 6-len(gems_to_change)))
        assert np.all(self.gems >= 0), "player changed gems to less than 0"
        assert np.sum(self.gems) <= 12, "player changed gems to more than 10" # RAISED GEM CAP 2

    def get_bought_card(self, card):
        self.cards[card.gem] += 1
        self.points += card.points
        self.card_ids[card.tier][card.gem].append(card.id)

    def choose_discard(self, game_state):
        # Set legal mask to only legal discards
        legal_mask = np.zeros(61, dtype=int)
        legal_mask[10:15] = game_state[self.state_offset:self.state_offset+5].astype(bool).astype(int) # Based on player's gems
        assert sum(legal_mask) > 1, "not enough legal moves in choose_discard"

        # Call the model to choose a discard
        rl_moves = self.rl_model.get_predictions(game_state, legal_mask)
        move_index = np.argmax(rl_moves)
        gem_index = move_index - 10

        next_state = game_state.copy()
        next_state[gem_index] += 1  # Update board's gems
        next_state[gem_index + self.state_offset] -= 1  # Update player's gems
        self.rl_model.remember(game_state, move_index, -1, next_state, 0) # Negative reward for inefficient move

        return gem_index, next_state

    def take_tokens_loop(self, game_state):
        total_gems = sum(self.gems)
        chosen_gems = np.zeros(5, dtype=int)

        takes_remaining = 3
        legal_selection = game_state[:5]

        while takes_remaining and np.any(legal_selection):
            # Discard if required
            if total_gems == 12:
                gem_index, next_state = self.choose_discard(game_state)

                # Implement move
                total_gems -= 1
                chosen_gems[gem_index] -= 1 # Update for apply_move()?
                legal_selection[gem_index] += 1 # Taking a second of the same gem is now legal
                game_state = next_state.copy()
            
            # Set legal mask to only legal takes
            legal_mask = np.zeros(61, dtype=int)
            legal_mask[:5] = (legal_selection > 0).astype(int)
            assert sum(legal_mask) > 0, "no legal moves in take tokens loop - take"

            # Call the model to choose a take
            rl_moves = self.rl_model.get_predictions(game_state, legal_mask)
            gem_index = np.argmax(rl_moves)

            next_state = game_state.copy()
            next_state[gem_index] -= 1 # Update board's gems
            next_state[gem_index+self.state_offset] += 1 # Update player's gems
            reward = -0.2 if takes_remaining == 3 else 0
            self.rl_model.remember(game_state, gem_index, reward, next_state, 0)
            
            # Implement move
            total_gems += 1
            takes_remaining -= 1
            chosen_gems[gem_index] += 1
            legal_selection[gem_index] *= 0 # Taking this gem again is now illegal unless previously discarded
            game_state = next_state.copy()

        return chosen_gems

    def buy_with_gold_loop(self, game_state, move_index, card):
        # Model remembers selecting the move and the reward first
        self.rl_model.remember(game_state, move_index, card.points, game_state, 0) # Same state for now

        chosen_gems = np.zeros(6, dtype=int)
        legal_mask = np.zeros(61, dtype=int) # Action vector size
        cost = np.append(card.cost, 0)

        while sum(cost) > 0:
            gems = self.gems + chosen_gems # Update the player's gems to a local variable

            # Legal tokens to spend
            legal_mask[10:15] = (gems*cost != 0).astype(int)[:5] # Can only spend gems where card cost remains
            legal_mask[60] = 1 if gems[5] else 0 # Enable spending gold as a legal move
            assert sum(legal_mask) > 0, "no legal moves in buy w/gold loop"

            # Predict token to spend
            rl_moves = self.rl_model.get_predictions(game_state, legal_mask)
            move_index = np.argmax(rl_moves)
            gem_index = 5 if move_index == 60 else move_index-10

            # Remember
            next_state = game_state.copy()
            next_state[gem_index] += 1 # Board gains the gem
            next_state[gem_index+self.state_offset] -= 1 # Player spends the gem
            self.rl_model.remember(game_state, move_index, 0.1, next_state, 0) # Small reward to shape towards legal moves, even though we have a mask

            # Propagate move
            chosen_gems[gem_index] -= 1
            cost[gem_index] -= 1
            game_state = next_state

        return chosen_gems

    def get_legal_moves(self, board):
        legal_moves = []

        # Take gems
        if sum(self.gems) <= 8: # We actually can take 2 if more than 8 but will need discard
            for gem, amount in enumerate(board.gems[:5]):
                if amount >= 4:
                    legal_moves.append(('take', (gem, 2)))
        legal_moves.append(('take', (0, 1))) # Random of the 5 moves because right now it gets overriden

        # Reserve card
        if len(self.reserved_cards) < 3:
            for tier_index, tier in enumerate(board.cards[:3]):
                for card_index, card in enumerate(tier):
                    if card:
                        legal_moves.append(('reserve', (tier_index, card_index)))
                if board.deck_mapping[tier_index].cards:
                    legal_moves.append(('reserve top', (tier_index, None))) # Setting card_index to None because it shouldn't be needed
        
        # Buy card
        for tier_index, tier in enumerate(board.cards[:3]):
            for card_index, card in enumerate(tier):
                if card:
                    can_afford = can_afford_with_gold = True
                    gold_needed = 0

                    for gem_index, amount in enumerate(card.cost):
                        if self.gems[gem_index] < amount:
                            can_afford = False
                            gold_needed += amount - self.gems[gem_index]
                            if gold_needed > self.gems[5]:
                                can_afford_with_gold = False
                                break

                    if can_afford:
                        legal_moves.append(('buy', (tier_index, card_index)))
                    elif can_afford_with_gold:
                        legal_moves.append(('buy with gold', (tier_index, card_index)))

        # Buy reserved card
        for card_index, card in enumerate(self.reserved_cards):
            can_afford = can_afford_with_gold = True
            gold_needed = 0

            for gem_index, amount in enumerate(card.cost):
                if self.gems[gem_index] < amount:
                    can_afford = False
                    gold_needed += amount - self.gems[gem_index]
                    if gold_needed > self.gems[5]:
                        can_afford_with_gold = False
                        break

            if can_afford:
                legal_moves.append(('buy reserved', (None, card_index)))
            elif can_afford_with_gold:
                legal_moves.append(('buy reserved with gold', (None, card_index)))
        
        return legal_moves
    
    def legal_to_vector(self, legal_moves):
        legal_mask = np.zeros(61, dtype=int)
        for move, details in legal_moves:
            tier, card_index = details
            match move:
                case 'take':
                    gem, amount = details # Overriding tier and card_index
                    if amount == 1:
                        legal_mask[gem] = 1
                    elif amount == 2:
                        legal_mask[gem+5] = 1
                    assert gem <= 5, f"take is off-index: {gem}"
                case 'buy':
                    assert 15 + 4*tier + card_index < 45, f"buy is off-index: {tier}, {card_index}"
                    legal_mask[15 + 4*tier + card_index] = 1
                case 'buy reserved':
                    assert 27 + card_index <= 30, f"buy reserved is off-index: {card_index}"
                    legal_mask[27 + card_index] = 1
                case 'buy with gold':
                    assert 30 + 4*tier + card_index <= 42, f"buy with gold is off-index: {tier}, {card_index}"
                    legal_mask[30 + 4*tier + card_index] = 1
                case 'buy reserved with gold':
                    assert 42 + card_index <= 45, f"buy reserved with gold is off-index: {card_index}"
                    legal_mask[42 + card_index] = 1
                case 'reserve':
                    assert 45 + 4*tier + card_index <= 57, f"reserve is off-index: {tier}, {card_index}"
                    legal_mask[45 + 4*tier + card_index] = 1
                case 'reserve top':
                    assert 57 + tier <= 60, f"reserve top is off-index: {tier}"
                    legal_mask[57 + tier] = 1

        return legal_mask
    
    def vector_to_details(self, board, game_state, move_index):
        tier = move_index % 15 // 4
        card_index = move_index % 15 % 4

        if move_index < 15:  # Take (includes discarding a gem)
            gem_index = move_index % 5
            gems_to_take = np.zeros(6, dtype=int)

            if move_index < 5:
                self.entered_loop = True
                move = ('take', (self.take_tokens_loop(game_state), None))
            elif move_index < 10:
                gems_to_take[gem_index] = 2
                move = ('take', (gems_to_take, None))
            else:
                raise Exception("Player chose to discard a gem")
                gems_to_take[gem_index] = -1
                move = ('take', (gems_to_take, None))
        elif move_index == 60: # Take gold
            raise Exception("Player chose to take a gold")
            move = ('take', ([0, 0, 0, 0, 0, 1], None))

        elif move_index < 45: # Buy
            if move_index < 27:
                move = ('buy', (tier, card_index))
            elif move_index < 30:
                move = ('buy reserved', (None, move_index-27))
            elif move_index < 42:
                card = board.cards[tier][card_index]

                # Remember move before entering loop
                next_state = game_state.copy()
                next_state[self.state_offset + card.gem] += 1
                next_state[self.state_offset + 45] += card.points
                self.rl_model.remember(game_state, move_index, card.points, next_state, 0)  # Remembering before the loop
                
                self.entered_loop = True # We'll remember() the submoves in the loop
                spent_gems = self.buy_with_gold_loop(game_state, move_index, card)
                move = ('buy with gold', ((tier, card_index), spent_gems))
            else:
                card_index = move_index - 42
                card = self.reserved_cards[card_index]

                # Remember move before entering loop
                next_state = game_state.copy()
                next_state[self.state_offset + card.gem] += 1
                next_state[self.state_offset + 45] += card.points
                self.rl_model.remember(game_state, move_index, card.points, next_state, 0)  # Remembering before the loop
                
                self.entered_loop = True # We'll remember() the submoves in the loop
                spent_gems = self.buy_with_gold_loop(game_state, move_index, card)
                move = ('buy reserved with gold', (card_index, spent_gems))

        else: # < 60 Reserve
            if move_index < 57:
                move = ('reserve', (tier, card_index))
            else: # 60 already covered
                move = ('reserve top', (move_index-57, None))
        
        return move
    
    def choose_move(self, board, game_state):
        legal_moves = self.get_legal_moves(board)
        legal_mask = self.legal_to_vector(legal_moves)
        rl_moves = self.rl_model.get_predictions(game_state, legal_mask)
        #strategic_moves = self.strategy.strategize(game_state, rl_moves, self.strategy_strength)
        
        self.move_index = np.argmax(rl_moves)
        self.chosen_move = self.vector_to_details(board, game_state, self.move_index)

        return self.chosen_move
    
    def get_state(self):
        return {
            'gems': self.gems.tolist(), 
            'cards': self.card_ids, 
            'reserved_cards': [(card.tier, card.id) for card in self.reserved_cards], 
            'points': self.points
        }

    def to_vector(self):
        reserved_cards_vector = []
        for card in self.reserved_cards:
            reserved_cards_vector.extend(card.vector)
        reserved_cards_vector.extend([0] * 11*(3-len(self.reserved_cards)))

        state_vector = np.concatenate((
            self.gems, # length 6
            self.cards, # length 5
            reserved_cards_vector, # length 11*3 = 33
            [self.points] # length 1
        ))

        assert len(state_vector) == 45, f"Player state vector is not 45, but {len(state_vector)}"
        return state_vector
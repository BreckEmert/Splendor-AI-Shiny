# Splendor/Environment/Splendor_components/player.py
from .strategy import *
from RL.model import rl_model


class Player:
    def __init__(self, name, strategy, strategy_strength):
        self.name: str = name
        self.gems: dict = {'white': 0, 'blue': 0, 'green': 0, 'red': 0, 'black': 0, 'gold': 0}
        self.gem_cards: dict = {'white': 0, 'blue': 0, 'green': 0, 'red': 0, 'black': 0}
        self.cards: list = []
        self.reserved_cards: list = []
        self.points: int = 0

        self.rl_model = rl_model()
        self.strategy: strategy = strategy
        self.strategy_strength: int = strategy_strength

    def get_state(self):
        return {
            'gems': self.gems, 
            'gem_cards': self.gem_cards, 
            'cards': self.cards, 
            'reserved_cards': self.reserved_cards, 
            'points:': self.points
        }

    def take_gems(self, gems_to_take):
        for gem, amount in gems_to_take.items():
            self.gems[gem] += amount

    def buy_card(self, card):
        for gem, amount in card.cost.items():
            self.gems[gem] -= amount
        self.gem_cards[card.gem_type] += 1
        self.cards.append(card)

    def reserve_card(self, card):
        self.reserved_cards.append(card)

    def get_legal_moves(self, board):
        best_moves = []
        total_gems = self.total_gems()
        if total_gems <= 8:
            if total_gems < 7:
                # Taking 3 different gems if available
                for gem1 in board.gems:
                    for gem2 in board.gems:
                        for gem3 in board.gems:
                            if gem1 != gem2 and gem2 != gem3 and gem1 != gem3:
                                min_count = min(board.gems[gem1] + board.gems[gem2] + board.gems[gem3])
                                if min_count > 0:
                                    best_moves.append(('take', {gem1: 1, gem2: 1, gem3: 1}))
            
            # Taking 2 of the same gem if at least 4 available
            for gem in board.gems:
                if board.gems[gem] >= 4:
                    best_moves.append(('take', {gem: 2}))
        
        worse_moves = []
        if len(best_moves) < 10:
            # Taking 2 different gems if available
            for gem1 in board.gems:
                for gem2 in board.gems:
                    if gem1 != gem2:
                        min_count = min(board.gems[gem1], board.gems[gem2])
                        if min_count > 0:
                            worse_moves.append(('take', {gem1: 1, gem2: 1}))

        if len(worse_moves) == 0:
            # Taking 1 gem if available
            for gem1 in board.gems:
                for gem2 in board.gems:
                    if gem1 != gem2:
                        min_count = min(board.gems[gem1], board.gems[gem2])
                        if min_count > 0:
                            worse_moves.append(('take', {gem1: 1, gem2: 1}))
        
        # Reserve card moves
        for card in board.get_reservable_cards():
            best_moves.append(('reserve', card))

        # Buy card moves
        for card in board.get_all_available_cards():
            for gem, amount in card.cost.items():
                if self.gems[gem] - amount >= 0:
                    legal_moves.append(('buy', card))                
        
        if len(best_moves) < 10:
            legal_moves = best_moves.extend(worse_moves)

        return legal_moves

    def move(self, board_state):
        legal_moves = self.get_legal_moves()
        rl_moves = self.rl_model(board_state, legal_moves)
        strategic_moves = self.strategy.strategize(board_state, rl_moves)
        return strategic_moves
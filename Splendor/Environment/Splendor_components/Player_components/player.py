# Splendor/Environment/Splendor_components/player.py

from RL import RLAgent # type: ignore


class Player:
    def __init__(self, name, strategy, strategy_strength):
        self.name: str = name
        self.gems: dict = {'white': 0, 'blue': 0, 'green': 0, 'red': 0, 'black': 0, 'gold': 0}
        self.cards: dict = {'white': 0, 'blue': 0, 'green': 0, 'red': 0, 'black': 0}
        self.reserved_cards: list = []
        self.points: int = 0

        self.rl_model = RLAgent()
        #self.strategy: strategy = strategy
        #self.strategy_strength: int = strategy_strength

    def get_state(self):
        return {
            'gems': self.gems, 
            'cards': self.cards, 
            'reserved_cards': self.reserved_cards, 
            'points:': self.points
        }

    def to_vector(self):
        return (
            list(self.gems.values()) + 
            list(self.cards.values()) + 
            [card.to_vector() for card in self.reserved_cards] + 
            [self.points])
    
    def take_gems(self, gems_to_take):
        for gem, amount in gems_to_take.items():
            self.gems[gem] += amount

    def buy_card(self, card):
        for gem, amount in card.cost.items():
            self.gems[gem] -= amount
        self.cards[card.gem_type] += 1

    def reserve_card(self, card):
        self.reserved_cards.append(card)

    def get_legal_moves(self, board):
        best_moves = []

        # Take gems moves
        total_gems = sum(self.gems.values())
        take_2_count = 0
        take_3_count = 0
        if total_gems <= 8:  # Max of 10 gems in inventory
            non_zero_gems = [gem for gem in board.gems if gem != 'gold' and board.gems[gem] > 0]
            len_nzg = len(non_zero_gems)

            if total_gems < 7:
                # Taking 3 different gems if available
                for i in range(len_nzg):
                    for j in range(i + 1, len_nzg):
                        for k in range(j + 1, len_nzg):
                            best_moves.append(('take', {non_zero_gems[i]: 1,
                                                        non_zero_gems[j]: 1, 
                                                        non_zero_gems[k]: 1}))
                            take_3_count += 1

            # Taking 2 of the same gem if at least 4 available
            for gem, count in board.gems.items():
                if gem != 'gold' and count >= 4:
                    best_moves.append(('take', {gem: 2}))

        worse_moves = []
        if take_3_count == 0:
            # Taking 2 different gems if available
            for i in range(len_nzg):
                for j in range(i + 1, len_nzg):
                    worse_moves.append(('take', {non_zero_gems[i]: 1,
                                                 non_zero_gems[j]: 1}))
                    take_2_count += 1

        if take_2_count == 0:
            # Taking 1 gem if available
            for gem, count in board.gems.items():
                if gem != 'gold' and count > 0:
                    worse_moves.append(('take', {gem: 1}))
        
        # Reserve card moves
        for card in board.get_reservable_cards():
            best_moves.append(('reserve', card))

        # Buy card moves
        for card in board.get_all_available_cards():
            for gem, amount in card.cost.items():
                if self.gems[gem] - amount >= 0:
                    best_moves.append(('buy', card))                

        return best_moves + worse_moves

    def choose_move(self, board, game_state):
        legal_moves = self.get_legal_moves(board)
        rl_moves = self.rl_model.get_predictions(game_state, legal_moves)
        #strategic_moves = self.strategy.strategize(game_state, rl_moves, self.strategy_strength)

        return max(rl_moves, key=lambda move: move[1])[0]
    
if __name__ == "__main__":
    import sys

    sys.path.append("C:/Users/Public/Documents/Python_Files/Splendor")

    from Environment.Splendor_components.Player_components.strategy import BestStrategy # type: ignore
    from RL.model import RLAgent # type: ignore

    bob = Player('Bob', BestStrategy(), 1)
    print(bob.to_vector())
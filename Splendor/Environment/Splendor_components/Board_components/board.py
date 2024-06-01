# Splendor/Environment/Splendor_components/Board_components/board.py
from deck import Tier1, Tier2, Tier3, Nobles


class Board:
    def __init__(self, num_players):
        # Gems
        gems = 4 + num_players//3
        self.gems = {'white': gems, 'blue': gems, 'green': gems, 'red': gems, 'black': gems, 'gold': 5}

        # Decks
        self.Tier1 = Tier1
        self.Tier2 = Tier2
        self.Tier3 = Tier3
        self.Nobles = Nobles

        # Active cards
        self.cards = {
            'tier1': [self.Tier1.draw() for _ in range(4)],
            'tier2': [self.Tier2.draw() for _ in range(4)],
            'tier3': [self.Tier3.draw() for _ in range(4)],
            'nobles': [self.Nobles.draw() for _ in range(num_players+1)]
        }

    def get_state(self):
        return {
            'Gems': self.gems, 
            'Cards': self.cards
        }
    
    def take_gem(self, gem, amount):
        self.gems[gem] -= amount

    def take_card(self, card):
        self.cards[card.tier].remove(card)
        self.cards[card.tier].append(self.tier.pop())

# board = Board(2)
# state = board.get_state()
# print(state.keys())
# Splendor/Environment/Splendor_components/Board_components/board.py

from .deck import Deck


class Board:
    def __init__(self, num_players):
        # Gems
        gems = 4 + num_players//3
        self.gems = {'white': gems, 'blue': gems, 'green': gems, 'red': gems, 'black': gems, 'gold': 5}

        # Decks
        self.Tier1 = Deck('Tier1')
        self.Tier2 = Deck('Tier2')
        self.Tier3 = Deck('Tier3')
        self.Nobles = Deck('Nobles')

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
    
    def to_vector(self):
        state_vector = list(self.gems.values())
        for tier in ['tier1', 'tier2', 'tier3', 'nobles']:
            for card in self.cards[tier]:
                state_vector.extend(card.to_vector())
        return state_vector
    
    def take_gem(self, gem, amount):
        self.gems[gem] -= amount

    def take_card(self, card):
        self.cards[card.tier].remove(card)
        self.cards[card.tier].append(self.tier.pop())

if __name__ == "__main__":
    import sys

    sys.path.append("C:/Users/Public/Documents/Python_Files/Splendor")

    from Environment.Splendor_components.Board_components.deck import Deck # type: ignore

    b1 = Board(2)
    print(b1.to_vector())
# Splendor/Environment/Splendor_components/Board_components/deck.py

import random
import pandas as pd


class Card:
    def __init__(self, id, tier, gem, points, cost):
        self.id = id
        self.tier = tier
        self.gem = gem
        self.points = points
        self.cost = cost  # Dictionary of gem costs

    def to_vector(self):
        return [self.id, self.points, self.cost['White'], self.cost['Blue'], self.cost['Green'], self.cost['Red'], self.cost['Black']]
    
    def __repr__(self):
        return f'Card(ID: {self.id}, Gem: {self.gem}, Points: {self.points}, Cost: {self.cost})'


class Deck:
    def __init__(self, Tier):
        self.tier = Tier
        self.cards = self.load_deck()

    def load_deck(self):
        path = 'C:/Users/Public/Documents/Python_Files/Splendor/Environment/Splendor_components/Board_components/Splendor_Cards.xlsx'
        deck = pd.read_excel(path, sheet_name=self.tier)

        cards = []
        for _, row in deck.iterrows():
            id, gem, points, white, blue, green, red, black = row
            cards.append(Card(id = id, tier = self.tier, gem = gem, points = points, 
                              cost = {'White': white, 'Blue': blue, 'Green': green, 'Red': red, 'Black': black}))
            
        random.shuffle(cards)
        
        return cards

    def draw(self):
        return self.cards.pop() if self.cards else None
    
    def __repr__(self):
        return f'Deck: {self.tier} with {len(self.cards)} cards remaining'
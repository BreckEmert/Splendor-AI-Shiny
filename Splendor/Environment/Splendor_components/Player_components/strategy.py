# Splendor/Environment/Splendor_components/Player_components folder - strategy.py
import numpy as np


class BestStrategy():
    """Return the raw moves, as that's what the model chose
    """
    def strategize(moves):
        return moves

class RandomStrategy():
    """Adjusts the logits randomly
    """
    def strategize(moves):
        noise = np.random.rand(*moves.shape) - 0.5
        return moves + noise

class OffensiveStrategy(Strategy):
    """Actively looks at other players' states and increases logits of interfering
    """
    def strategize(moves):
        for player in players:
            pass
            # Increments move choices that align with progress towards resource cards and nobles
            # Only distrupts if taking the gem would limit the other player's options

    def choosemove():
        # 1: Sum tier n gems * round weight for tier n
        # 2: Analyze reserving
        # 3: Analyze buying
            # Never buy if the card can't be bought or reserved, and we have good choices of gems left.
        pass

class ResourceHog():
    def __init__(self):
        pass

class ObliviousStrategy():
    def __init__(self):
        pass
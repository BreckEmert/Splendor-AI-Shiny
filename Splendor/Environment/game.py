# Splendor/Environment/game.py
from Splendor_components.board import Board
from Splendor_components.player import Player


class Game:
    def __init__(self, players):
        self.num_players = len(players)

        self.board = Board(self.num_players)
        self.players: list = [Player(name, strategy) for name, strategy in players]

        self.turn: int = 0
        self.turn_index: int = 0
        self.is_final_turn: bool = False

    def get_state(self):
        return {
            'board': self.board.get_state(),
            'players': {player.name: player.get__state() for player in self.players}, 
            'current_turn': self.turn, 
            'is_final_turn': self.is_final_turn
        }

    def turn(self):
        if self.is_final_turn:
            if self.turn_index == self.num_players-1:
                self.get_victor()

        self.turn_index = (self.turn) % self.num_players
        active_player = self.players[self.turn_index]

        active_player.move(self.get_state())
        self.check_noble_visit(active_player)
        self.check_15(active_player)

        self.turn_index = (self.turn_index + 1) % self.num_players

    def check_noble_visit(self, active_player):
        for noble in self.board.nobles:
            if all(active_player.gem_cards[gem] >= amount for gem, amount in noble.cost.items()):
                active_player.points += noble.points
                self.Board.deck.remove(noble)
                break # Implement logic to choose the noble if tied

    def check_15(self, active_player):
        if active_player.points >= 15:
            self.is_final_turn = True

    def get_victor(self):
        victor = max(self.players, key=lambda p: p.points)
        return victor
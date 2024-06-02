# Splendor/Environment/game.py

from Environment.Splendor_components import Board # type: ignore
from Environment.Splendor_components import Player # type: ignore

class Game:
    def __init__(self, players):
        self.num_players = len(players)

        self.board = Board(self.num_players)
        self.players: list = [Player(name, strategy, strategy_strength) 
                              for name, strategy, strategy_strength in players]

        self.turn_index: int = 0
        self.turn_order: int = 0
        self.is_final_turn: bool = False

    def get_state(self):
        return {
            'board': self.board.get_state(),
            'players': {player.name: player.get_state() for player in self.players}, 
            'current_turn': self.turn, 
            'is_final_turn': self.is_final_turn
        }

    def to_vector(self):
        state_vector = self.board.to_vector()
        for player in self.players:
            state_vector.extend(player.to_vector())
        state_vector.append(self.turn_order)
        state_vector.append(int(self.is_final_turn))
        return state_vector
    
    def turn(self):
        active_player = self.players[self.turn_order]
        prev_state = self.get_state()

        chosen_move = active_player.choose_move(self.board, prev_state)
        self.apply_move(active_player, chosen_move)

        self.check_noble_visit(active_player)
        if active_player.points >= 15:
            self.is_final_turn = True

        self.turn_order = (self.turn_order + 1) % self.num_players
        self.turn_index += 1

    def apply_move(self, player, move):
        action, details = move
        if action == 'take':
            player.take_gems(self.board, details)
        elif action == 'buy':
            player.buy_card(self.board, details)
        elif action == 'reserve':
            player.reserve_card(self.board, details)

    def check_noble_visit(self, active_player):
        for noble in self.board.nobles:
            if all(active_player.cards[gem] >= amount for gem, amount in noble.cost.items()):
                active_player.points += noble.points
                self.Board.deck.remove(noble)
                break # Implement logic to choose the noble if tied

    def get_victor(self):
        victor = max(self.players, key=lambda p: p.points)
        return victor
   

if __name__ == "__main__":
    import sys

    sys.path.append("C:/Users/Public/Documents/Python_Files/Splendor")

    from Environment.Splendor_components import Board # type: ignore
    from Environment.Splendor_components import Player # type: ignore
    from Environment.Splendor_components.Player_components.strategy import BestStrategy # type: ignore

    players = [('Player1', BestStrategy(), 1), ('Player2', BestStrategy(), 1)]
    g1 = Game(players)
    print(g1.to_vector())
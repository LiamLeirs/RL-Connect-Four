import numpy as np
import pygame


class Opponent:
    def select_action(self, observation, action_mask, rng):
        raise NotImplementedError

class RandomOpponent(Opponent):
    def select_action(self, observation, action_mask, rng):
        legal = np.flatnonzero(action_mask)
        return int(rng.choice(legal))

class HumanOpponent(Opponent):
    def __init__(self, renderer=None, game=None):
        self.renderer = renderer
        self.game = game

    def select_action(self, observation, action_mask, rng):
        if self.renderer is None or self.game is None:
            raise RuntimeError(
                "HumanOpponent requires a renderer and game."
            )

        while True:
            running = self.renderer.update()

            if not running:
                raise KeyboardInterrupt("Pygame window closed.")

            self.renderer.draw_human(
                board=self.game.get_board(),
                current_player=self.game.get_current_player(),
                winner=self.game.get_winner(),
            )

            column = self.renderer.consume_click()

            if column is None:
                continue

            if action_mask[column]:
                return column
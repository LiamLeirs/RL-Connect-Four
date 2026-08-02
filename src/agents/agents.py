import numpy as np
import pygame
from src.envs.connect_four.game import ConnectFour
from src.envs.connect_four.renderer import ConnectFourRenderer
import copy

class Agent:
    def attach(self, *, game: ConnectFour, renderer: ConnectFourRenderer | None) -> None:
        """
        Called when the agent is assigned to an environment.
        """
        pass

    def on_episode_start(self,rng: np.random.Generator,) -> None:
        """
        Called once at the beginning of every episode.
        """
        pass

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray, rng: np.random.Generator) -> int:
        raise NotImplementedError

class RandomAgent(Agent):
    def select_action(self, observation, action_mask, rng=np.random):
        legal = np.flatnonzero(action_mask)
        return int(rng.choice(legal))

class HumanAgent(Agent):
    def __init__(self):
        self.renderer = None
        self.game = None

    def attach(self, *, game, renderer) -> None:
        if renderer is None:
            raise ValueError(
                "HumanAgent requires a renderer."
            )

        self.game = game
        self.renderer = renderer

    def select_action(self, observation, action_mask, rng=np.random):
        if self.renderer is None or self.game is None:
            raise RuntimeError(
                "HumanAgent requires a renderer and game."
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

class ModelAgent(Agent):
    def __init__(self, model, deterministic=True):
        self.model = model
        self.deterministic = deterministic

    def select_action(self, observation, action_mask, rng=np.random):
        action, _ = self.model.predict(observation, action_masks=action_mask, deterministic=self.deterministic)
        return int(action)

class TacticalAgent(Agent):
    def __init__(self):
        self.game = None

    def attach(self, *, game, renderer) -> None:
        self.game = game

    def select_action(self, observation, action_mask, rng=np.random):
        legal_actions = np.flatnonzero(action_mask)

        if len(legal_actions) == 0:
            raise RuntimeError("No legal actions available.")

        tactical_player = self.game.get_current_player()
        other_player = -tactical_player

        # 1. Play an immediate winning move.
        for action in legal_actions:
            simulated_game = copy.deepcopy(self.game)
            result = simulated_game.make_move(int(action))

            if (
                result is not False
                and result.winner == tactical_player
            ):
                return int(action)

        # 2. Block the other player's immediate winning move.
        for action in legal_actions:
            simulated_game = copy.deepcopy(self.game)

            # Temporarily give the other player the turn.
            simulated_game.current_player = other_player

            result = simulated_game.make_move(int(action))

            if (
                result is not False
                and result.winner == other_player
            ):
                return int(action)

        # 3. Prefer the centre.
        centre_column = self.game.num_cols // 2

        if action_mask[centre_column]:
            return centre_column

        # 4. Otherwise choose randomly.
        return int(rng.choice(legal_actions))

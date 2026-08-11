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

class MiniMaxAgent(Agent):
    def __init__(self, *, game=None, renderer=None, depth=4):
        self.depth = depth
        self.game = None
        self.renderer = renderer

        self.window_rows = None
        self.window_cols = None

        if game is not None:
            self.attach(game=game, renderer=renderer)

    def attach(self, *, game, renderer) -> None:
        self.game = game
        self.renderer = renderer
        self.window_rows, self.window_cols = self._generate_windows()
    
    def minimax(self, game, depth, maximizingPlayer, agentPlayer, alpha=-float("inf"), beta=float("inf")):
        winner = game.get_winner()

        # Return score based on depth to encourage faster wins / slower losses
        if winner == agentPlayer:
            return 1000_000 + depth
        elif winner == -agentPlayer:
            return -1000_000 - depth
        elif winner is not None:
            return 0

        if depth == 0:
            return self.heuristic(game, agentPlayer)

        legal_actions = np.flatnonzero(game.get_legal_moves())

        if maximizingPlayer:
            bestScore = -float("inf")
            for action in legal_actions:
                simulatedGame = copy.deepcopy(game)
                simulatedGame.make_move(action)
                score = self.minimax(simulatedGame, depth - 1, False, agentPlayer, alpha, beta)
                bestScore = max(bestScore, score)
                alpha = max(alpha, bestScore)
                if alpha >= beta:
                    break
            return bestScore
        else:
            bestScore = float("inf")
            for action in legal_actions:
                simulatedGame = copy.deepcopy(game)
                simulatedGame.make_move(action)
                score = self.minimax(simulatedGame, depth - 1, True, agentPlayer, alpha, beta)
                bestScore = min(bestScore, score)
                beta = min(beta, bestScore)
                if alpha >= beta:
                    break
            return bestScore

    def select_action(self, observation, action_mask, rng=np.random):
        best_actions = []
        best_score = -float("inf")
        legal_actions = np.flatnonzero(action_mask)
        agent_player = self.game.get_current_player()
        alpha = -float("inf")
        beta = float("inf")
        for action in legal_actions:
            simulated_game = copy.deepcopy(self.game)
            simulated_game.make_move(action)
            score = self.minimax(simulated_game, self.depth-1, False, agent_player, alpha, beta)
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)
            alpha = max(alpha, best_score)
        return int(rng.choice(best_actions))

    def _generate_windows(self):
        rows = self.game.num_rows
        cols = self.game.num_cols
        k = self.game.win_req

        windows = []

        # Horizontal
        for row in range(rows):
            for col in range(cols - k + 1):
                row_indices = np.full(k, row, dtype=np.intp)
                col_indices = np.arange(col, col + k, dtype=np.intp)

                windows.append((row_indices, col_indices))

        # Vertical
        for row in range(rows - k + 1):
            for col in range(cols):
                row_indices = np.arange(row, row + k, dtype=np.intp)
                col_indices = np.full(k, col, dtype=np.intp)

                windows.append((row_indices, col_indices))

        # Diagonal \
        for row in range(rows - k + 1):
            for col in range(cols - k + 1):
                offsets = np.arange(k, dtype=np.intp)

                row_indices = row + offsets
                col_indices = col + offsets

                windows.append((row_indices, col_indices))

        # Diagonal /
        for row in range(k - 1, rows):
            for col in range(cols - k + 1):
                offsets = np.arange(k, dtype=np.intp)

                row_indices = row - offsets
                col_indices = col + offsets

                windows.append((row_indices, col_indices))

        row_indices = np.array(
            [window[0] for window in windows],
            dtype=np.intp,
        )

        col_indices = np.array(
            [window[1] for window in windows],
            dtype=np.intp,
        )

        return row_indices, col_indices

    def heuristic(self, game, agentPlayer):
        board = game.get_board()
        windows = board[self.window_rows, self.window_cols]

        my_counts = np.sum(windows == agentPlayer, axis=1)
        opponent_counts = np.sum(windows == -agentPlayer, axis=1)
        empty_counts = np.sum(windows == 0, axis=1)

        score = 0
        k = game.win_req
        # Score potential wins / losses
        score += np.sum((my_counts == k-1) & (empty_counts == 1)) * 100
        score += np.sum((my_counts == k-2) & (empty_counts == 2)) * 10
        score -= np.sum((opponent_counts == k-1) & (empty_counts == 1)) * 120
        score -= np.sum((opponent_counts == k-2) & (empty_counts == 2)) * 10

        # Score center control
        center_col = game.num_cols // 2
        center = board[:, center_col]

        score += np.sum(center == agentPlayer) * 3

        score -= np.sum(center == -agentPlayer) * 3

        return score


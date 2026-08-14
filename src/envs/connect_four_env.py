import gymnasium as gym
import numpy as np
from src.envs.connect_four.renderer import RendererEvent
from src.envs.connect_four.game import ConnectFour, MoveResult
from src.agents.agents import *
import torch
from sb3_contrib import MaskablePPO


class ConnectFourEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    def __init__(self, num_rows=6, num_cols=7, win_req=4, opponent=None, opponent_provider=None, render_mode=None):
        super().__init__()
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"Unsupported render mode: {render_mode}")

        self.game = ConnectFour(
            num_cols=num_cols, num_rows=num_rows, win_req=win_req)
        self.action_space = gym.spaces.Discrete(self.game.num_cols)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(
            2, self.game.num_rows, self.game.num_cols), dtype=np.float32)
        self.render_mode = render_mode
        self.renderer = None

        self.opponent_provider = opponent_provider
        self.fixed_opponent = opponent or RandomAgent()
        self.current_opponent = None

    def _ensure_renderer(self):
        if self.renderer is None:
            from src.envs.connect_four.renderer import ConnectFourRenderer

            self.renderer = ConnectFourRenderer(
                num_rows=self.game.num_rows,
                num_cols=self.game.num_cols,
                fps=self.metadata["render_fps"],
            )

    def _animate_move(self, result: MoveResult) -> None:
        if self.render_mode != "human":
            return

        self._ensure_renderer()

        self.renderer.start_drop_animation(
            row=result.row,
            column=result.col,
            player=result.player,
        )

        while self.renderer.is_animating:
            event = self.renderer.update()

            if event == RendererEvent.QUIT:
                self.close()
                raise KeyboardInterrupt("Pygame window closed.")
            elif event == RendererEvent.RESET:
                self.reset()
            self.renderer.draw_human(
                board=self.game.get_board(),
                current_player=self.game.get_current_player(),
                winner=self.game.get_winner(),
            )

    def render(self):
        if self.render_mode is None:
            return None

        self._ensure_renderer()

        if self.render_mode == "human":
            self.renderer.draw(
                board=self.game.get_board(),
                current_player=self.game.get_current_player(),
                winner=self.game.get_winner(),
            )
            return None

        if self.render_mode == "rgb_array":
            return self.renderer.render_rgb_array(
                board=self.game.get_board(),
                current_player=self.game.get_current_player(),
                winner=self.game.get_winner(),
            )

        raise RuntimeError(
            f"Unexpected render mode: {self.render_mode}"
        )

    def _get_observation_for(self, player: int) -> np.ndarray:
        board = self.game.get_board()

        own_channel = board == player
        opponent_channel = board == -player

        observation = np.stack(
            [own_channel, opponent_channel],
            axis=0,
        ).astype(np.float32)
        return observation

    def _get_info(self, result: MoveResult | None = None) -> dict:
        return {
            "action_mask": self.action_masks(),
            "winner": self.game.get_winner(),
            "agent_player": self.agent_player,
            "opponent_player": self.opponent_player,
            "result": result,
        }

    def action_masks(self) -> np.ndarray:
        return self.game.get_legal_moves().copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.render_mode == "human":
            self._ensure_renderer()

        self.game.reset()

        # Determine sides
        if options is not None and "learner_starts" in options:
            if options["learner_starts"]:
                self.agent_player = 1
                self.opponent_player = -1
            else:
                self.agent_player = -1
                self.opponent_player = 1
        else:
            # Randomize sides
            if self.np_random.random() < 0.5:
                self.agent_player = 1
                self.opponent_player = -1
            else:
                self.agent_player = -1
                self.opponent_player = 1

        # Generate varied opening
        varied_starting_state = (
            options is not None
            and options.get("varied_starting_state", False)
        )

        if varied_starting_state:
            num_opening_moves = self.np_random.choice([0, 2, 4, 6])
            for _ in range(num_opening_moves):
                legal_actions = np.flatnonzero(
                    self.game.get_legal_moves()
                )

                action = int(
                    self.np_random.choice(legal_actions)
                )

                self.game.make_move(action)

        # Choose opponent for this episode
        if self.opponent_provider is not None:
            opponent = self.opponent_provider.sample_opponent(self.np_random)
        else:
            opponent = self.fixed_opponent
        self.current_opponent = opponent
        self._attach_opponent()

        self.current_opponent.on_episode_start(self.np_random)

        result = None

        # If opponent is first player, make opponent move
        if self.agent_player == -1:
            opponent_action = self._choose_opponent_action()
            result = self.game.make_move(opponent_action)
            self._animate_move(result)

        if self.render_mode == "human":
            self.render()

        return self._get_observation_for(self.agent_player), self._get_info(result)

    def _reward_from_winner(self, winner: int | None) -> float:
        if winner is None or winner == 0:
            return 0.0

        if winner == self.agent_player:
            return 1.0

        return -1.0

    def set_opponent(self, opponent):
        self.current_opponent = opponent
        self._attach_opponent()

    def _attach_opponent(self) -> None:
        renderer = None

        if self.render_mode == "human":
            self._ensure_renderer()
            renderer = self.renderer

        if self.current_opponent is not None:
            self.current_opponent.attach(
                game=self.game,
                renderer=renderer,
            )

    def _choose_opponent_action(self) -> int:
        return self.current_opponent.select_action(self._get_observation_for(self.opponent_player), self.action_masks(), self.np_random)

    @property
    def agent_turn(self) -> bool:
        return self.game.get_current_player() == self.agent_player

    def _transition(self, reward: float, terminated: bool, info: dict,):
        if self.render_mode == "human":
            self.render()

        return (
            self._get_observation_for(self.agent_player),
            float(reward),
            bool(terminated),
            False,
            info,
        )

    def step(self, action: int):
        if not self.agent_turn:
            raise RuntimeError(
                "step() called when it is not the agent's turn.")

        action = int(action)

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # Check if action is legal
        if not self.action_masks()[action]:
            self.game.skip_turn()
            opponent_action = self._choose_opponent_action()
            opponent_result = self.game.make_move(opponent_action)
            self._animate_move(opponent_result)
            if opponent_result is False:
                raise RuntimeError("Opponent selected an illegal action.")
            terminated = opponent_result.winner is not None
            if terminated:
                reward = self._reward_from_winner(opponent_result.winner)
            else:
                reward = -0.1
            info = self._get_info(opponent_result)
            info["agent_action"] = action
            info["opponent_action"] = opponent_action
            info["illegal_action"] = True
            return self._transition(reward, terminated, info)

        # Learner move
        agent_result = self.game.make_move(action)
        self._animate_move(agent_result)

        if agent_result.winner is not None:
            return self._transition(
                self._reward_from_winner(agent_result.winner),
                True,
                self._get_info(agent_result)
            )

        # Opponent move
        opponent_action = self._choose_opponent_action()
        opponent_result = self.game.make_move(opponent_action)

        if opponent_result is False:
            raise RuntimeError("Opponent selected an illegal action.")

        self._animate_move(opponent_result)
        terminated = opponent_result.winner is not None
        reward = self._reward_from_winner(
            opponent_result.winner
        )

        info = self._get_info(opponent_result)
        info["agent_action"] = action
        info["opponent_action"] = opponent_action

        return self._transition(reward, terminated, info)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


if __name__ == "__main__":
    opponent = HumanAgent()
    env = ConnectFourEnv(render_mode="human", opponent=opponent)
    model = MaskablePPO.load("models/ppo_selfplay/final_model.zip")
    agent = ModelAgent(
        model=model, deterministic=True)
    obs, info = env.reset()
    terminated = False
    running = True
    terminated = False

    while running:
        event = env.renderer.update()
        if event == RendererEvent.QUIT:
            running = False
            continue
        elif event == RendererEvent.RESET:
            obs, info = env.reset()
            terminated = False
        if not terminated:
            legal = np.flatnonzero(info["action_mask"])
            action = agent.select_action(obs, info["action_mask"])
            obs, reward, terminated, truncated, info = env.step(action)
        env.render()
    env.close()

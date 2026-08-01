import gymnasium as gym
import numpy as np
from src.envs.connect_four.game import ConnectFour
from src.agents.opponents import RandomOpponent

class ConnectFourEnv(gym.Env):
    def __init__(self, num_rows=6, num_cols=7, win_req=4):
        super().__init__()
        self.game = ConnectFour(num_cols=num_cols, num_rows=num_rows, win_req=win_req)
        self.action_space = gym.spaces.Discrete(self.game.num_cols)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2, self.game.num_rows, self.game.num_cols), dtype=np.float32)
        self.opponent = RandomOpponent()

    def _get_observation_for(self, player: int) -> np.ndarray:
        board = self.game.get_board()

        own_channel = board == player
        opponent_channel = board == -player

        observation = np.stack(
            [own_channel, opponent_channel],
            axis=0,
        ).astype(np.float32)
        return observation

    def _get_info(self) -> dict:
        return {
        "action_mask": self.action_masks(),
        "winner": self.game.get_winner(),
        "agent_player": self.agent_player,
        "opponent_player": self.opponent_player,
    }

    def action_masks(self) -> np.ndarray:
        return self.game.get_legal_moves().copy()

    def reset(self,*, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        if self.np_random.random() < 0.5:
            self.agent_player = 1
            self.opponent_player = -1
        else:
            self.agent_player = -1
            self.opponent_player = 1
        if self.agent_player == -1:
            opponent_action = self._choose_opponent_action()
            result = self.game.make_move(opponent_action)
        return self._get_observation_for(self.agent_player), self._get_info()

    def _reward_from_winner(self, winner: int | None) -> float:
        if winner is None or winner == 0:
            return 0.0

        if winner == self.agent_player:
            return 1.0

        return -1.0

    def set_opponent(self, opponent):
        self.opponent = opponent

    def _choose_opponent_action(self) -> int:
        return self.opponent.select_action(self._get_observation_for(self.opponent_player), self.action_masks(), self.np_random)

    @property
    def agent_turn(self) -> bool:
        return self.game.get_current_player() == self.agent_player
    
    def step(self, action: int):
        if not self.agent_turn:
            raise RuntimeError("step() called when it is not the agent's turn.")

        action = int(action)

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Check if action is legal
        if not self.action_masks()[action]:
            self.game.skip_turn()
            opponent_action = self._choose_opponent_action()
            opponent_result = self.game.make_move(opponent_action)
            if opponent_result is False:
                raise RuntimeError("Opponent selected an illegal action.")
            terminated = opponent_result.winner is not None
            if terminated:
                reward = self._reward_from_winner(opponent_result.winner)
            else:
                reward = -0.1
            info = self._get_info()
            info["agent_action"] = action
            info["opponent_action"] = opponent_action
            info["illegal_action"] = True
            return (
                self._get_observation_for(self.agent_player),
                reward,
                terminated,
                False,
                info,
            )

        # Learner move
        agent_result = self.game.make_move(action)

        if agent_result.winner is not None:
            return (
                self._get_observation_for(self.agent_player),
                self._reward_from_winner(agent_result.winner),
                True,
                False,
                self._get_info(),
            )

        # Opponent move
        opponent_action = self._choose_opponent_action()
        opponent_result = self.game.make_move(opponent_action)

        if opponent_result is False:
            raise RuntimeError("Opponent selected an illegal action.")

        terminated = opponent_result.winner is not None
        reward = self._reward_from_winner(
            opponent_result.winner
        )

        info = self._get_info()
        info["agent_action"] = action
        info["opponent_action"] = opponent_action

        return (
            self._get_observation_for(self.agent_player),
            reward,
            terminated,
            False,
            info,
        )

if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    env = ConnectFourEnv()
    check_env(env, warn=True)
    for episode in range(1000):
        obs, info = env.reset(seed=episode)
        terminated = False

        while not terminated:
            legal = np.flatnonzero(info["action_mask"])
            action = int(env.np_random.choice(legal))

            obs, reward, terminated, truncated, info = env.step(action)

    print("Completed 1,000 episodes.")
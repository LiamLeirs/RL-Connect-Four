import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete
from stable_baselines3.common.env_checker import check_env

from src.envs.connect_four_env import ConnectFourEnv
from src.agents.opponents import HumanOpponent, Opponent


# ---------------------------------------------------------------------------
# Deterministic test opponents
# ---------------------------------------------------------------------------

class FirstLegalOpponent(Opponent):
    """Always selects the leftmost legal column."""

    def select_action(self, observation, action_mask, rng):
        legal_actions = np.flatnonzero(action_mask)

        if len(legal_actions) == 0:
            raise RuntimeError("No legal actions available.")

        return int(legal_actions[0])


class FixedColumnOpponent(Opponent):
    """Selects a preferred column when legal."""

    def __init__(self, column: int):
        self.column = column

    def select_action(self, observation, action_mask, rng):
        if action_mask[self.column]:
            return self.column

        legal_actions = np.flatnonzero(action_mask)
        return int(legal_actions[0])


class InvalidOpponent(Opponent):
    """Intentionally returns an illegal action."""

    def select_action(self, observation, action_mask, rng):
        return len(action_mask)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    environment = ConnectFourEnv(
        opponent=FirstLegalOpponent(),
        render_mode=None,
    )

    yield environment

    environment.close()


def force_agent_to_start(environment: ConnectFourEnv) -> None:
    """
    Place the environment in a fresh state where the learner is Player 1.
    """
    environment.game.reset()
    environment.agent_player = 1
    environment.opponent_player = -1

    assert environment.agent_turn


def force_agent_to_play_second(environment: ConnectFourEnv) -> None:
    """
    Place the environment in a fresh state where the learner is Player -1.

    The opponent makes one opening move so that control belongs to the agent.
    """
    environment.game.reset()
    environment.agent_player = -1
    environment.opponent_player = 1

    opening_action = environment._choose_opponent_action()
    result = environment.game.make_move(opening_action)

    assert result is not False
    assert environment.agent_turn


# ---------------------------------------------------------------------------
# Constructor and spaces
# ---------------------------------------------------------------------------

def test_default_action_space(env):
    assert isinstance(env.action_space, Discrete)
    assert env.action_space.n == 7


def test_default_observation_space(env):
    assert isinstance(env.observation_space, Box)
    assert env.observation_space.shape == (2, 6, 7)
    assert env.observation_space.dtype == np.float32
    assert np.all(env.observation_space.low == 0)
    assert np.all(env.observation_space.high == 1)


def test_custom_board_dimensions():
    env = ConnectFourEnv(
        num_rows=7,
        num_cols=8,
        win_req=5,
        opponent=FirstLegalOpponent(),
    )

    try:
        assert env.action_space.n == 8
        assert env.observation_space.shape == (2, 7, 8)
    finally:
        env.close()


@pytest.mark.parametrize("render_mode", ["invalid", "video", True])
def test_invalid_render_mode_raises(render_mode):
    with pytest.raises(ValueError, match="Unsupported render mode"):
        ConnectFourEnv(render_mode=render_mode)


def test_human_opponent_requires_human_render_mode():
    opponent = HumanOpponent(renderer=None, game=None)

    with pytest.raises(AssertionError):
        ConnectFourEnv(
            opponent=opponent,
            render_mode=None,
        )


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_reset_returns_observation_and_info(env):
    observation, info = env.reset(seed=42)

    assert isinstance(observation, np.ndarray)
    assert isinstance(info, dict)


def test_reset_observation_matches_space(env):
    observation, _ = env.reset(seed=42)

    assert env.observation_space.contains(observation)


def test_reset_assigns_opposite_player_ids(env):
    _, info = env.reset(seed=42)

    assert info["agent_player"] in {-1, 1}
    assert info["opponent_player"] == -info["agent_player"]


def test_reset_always_returns_on_agent_turn(env):
    for seed in range(100):
        env.reset(seed=seed)

        assert env.agent_turn


def test_opponent_opens_when_agent_is_second(env):
    found_second_player_episode = False

    for seed in range(100):
        observation, info = env.reset(seed=seed)

        if info["agent_player"] == -1:
            found_second_player_episode = True

            board = env.game.get_board()

            assert np.count_nonzero(board == 1) == 1
            assert np.count_nonzero(board == -1) == 0
            assert np.count_nonzero(observation[0]) == 0
            assert np.count_nonzero(observation[1]) == 1
            assert info["result"] is not None
            break

    assert found_second_player_episode


def test_empty_board_when_agent_starts(env):
    found_first_player_episode = False

    for seed in range(100):
        observation, info = env.reset(seed=seed)

        if info["agent_player"] == 1:
            found_first_player_episode = True

            assert np.count_nonzero(env.game.get_board()) == 0
            assert np.count_nonzero(observation) == 0
            assert info["result"] is None
            break

    assert found_first_player_episode


def test_side_randomization_produces_both_sides(env):
    assigned_players = set()

    for seed in range(100):
        _, info = env.reset(seed=seed)
        assigned_players.add(info["agent_player"])

    assert assigned_players == {-1, 1}


def test_reset_is_reproducible_for_same_seed():
    env_one = ConnectFourEnv(opponent=FirstLegalOpponent())
    env_two = ConnectFourEnv(opponent=FirstLegalOpponent())

    try:
        observation_one, info_one = env_one.reset(seed=123)
        observation_two, info_two = env_two.reset(seed=123)

        np.testing.assert_array_equal(
            observation_one,
            observation_two,
        )

        np.testing.assert_array_equal(
            info_one["action_mask"],
            info_two["action_mask"],
        )

        assert info_one["agent_player"] == info_two["agent_player"]
        assert info_one["opponent_player"] == info_two["opponent_player"]
    finally:
        env_one.close()
        env_two.close()


# ---------------------------------------------------------------------------
# Observation representation
# ---------------------------------------------------------------------------

def test_observation_has_two_binary_channels(env):
    observation, _ = env.reset(seed=42)

    assert observation.shape == (2, 6, 7)
    assert observation.dtype == np.float32
    assert np.all(np.isin(observation, [0.0, 1.0]))


def test_channels_do_not_overlap(env):
    observation, info = env.reset(seed=42)
    terminated = False

    while not terminated:
        legal_actions = np.flatnonzero(info["action_mask"])
        action = int(legal_actions[-1])

        observation, _, terminated, _, info = env.step(action)

        overlap = observation[0] * observation[1]

        assert np.count_nonzero(overlap) == 0


def test_observation_is_relative_to_requested_player(env):
    env.game.reset()
    env.game.board[5, 0] = 1
    env.game.board[5, 1] = -1

    player_one_observation = env._get_observation_for(1)
    player_two_observation = env._get_observation_for(-1)

    assert player_one_observation[0, 5, 0] == 1
    assert player_one_observation[1, 5, 1] == 1

    assert player_two_observation[0, 5, 1] == 1
    assert player_two_observation[1, 5, 0] == 1


def test_returned_observation_does_not_modify_game(env):
    observation, _ = env.reset(seed=42)

    observation[:] = 1

    assert not np.all(env.game.get_board() == 1)


# ---------------------------------------------------------------------------
# Action masks
# ---------------------------------------------------------------------------

def test_initial_action_mask_contains_all_columns(env):
    force_agent_to_start(env)

    expected = np.ones(7, dtype=bool)

    np.testing.assert_array_equal(
        env.action_masks(),
        expected,
    )


def test_action_mask_has_correct_type_and_shape(env):
    env.reset(seed=42)

    mask = env.action_masks()

    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.shape == (7,)


def test_full_column_is_masked(env):
    force_agent_to_start(env)

    for _ in range(env.game.num_rows):
        result = env.game.make_move(0)
        assert result is not False

    assert not env.action_masks()[0]
    assert np.all(env.action_masks()[1:])


def test_info_action_mask_matches_environment(env):
    _, info = env.reset(seed=42)

    np.testing.assert_array_equal(
        info["action_mask"],
        env.action_masks(),
    )


# ---------------------------------------------------------------------------
# Normal step behaviour
# ---------------------------------------------------------------------------

def test_step_returns_five_values(env):
    force_agent_to_start(env)

    result = env.step(3)

    assert len(result) == 5


def test_step_return_types(env):
    force_agent_to_start(env)

    observation, reward, terminated, truncated, info = env.step(3)

    assert isinstance(observation, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_non_terminal_step_contains_two_moves(env):
    force_agent_to_start(env)

    observation, reward, terminated, truncated, info = env.step(3)

    assert not terminated
    assert not truncated
    assert reward == 0.0
    assert np.count_nonzero(env.game.get_board()) == 2
    assert np.count_nonzero(observation) == 2


def test_control_returns_to_agent_after_step(env):
    force_agent_to_start(env)

    _, _, terminated, _, _ = env.step(3)

    if not terminated:
        assert env.agent_turn


def test_step_info_contains_actions(env):
    force_agent_to_start(env)

    _, _, _, _, info = env.step(3)

    assert info["agent_action"] == 3
    assert info["opponent_action"] == 0


def test_truncated_is_always_false(env):
    observation, info = env.reset(seed=42)

    terminated = False

    while not terminated:
        legal_actions = np.flatnonzero(info["action_mask"])
        action = int(legal_actions[-1])

        observation, _, terminated, truncated, info = env.step(action)

        assert truncated is False


@pytest.mark.parametrize("action", [-1, 7, 100])
def test_out_of_range_action_raises(env, action):
    force_agent_to_start(env)

    with pytest.raises(ValueError, match="Invalid action"):
        env.step(action)


def test_step_called_on_opponent_turn_raises(env):
    env.game.reset()
    env.agent_player = -1
    env.opponent_player = 1

    assert not env.agent_turn

    with pytest.raises(RuntimeError, match="not the agent's turn"):
        env.step(0)


# ---------------------------------------------------------------------------
# Reward logic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("agent_player", "winner", "expected_reward"),
    [
        (1, 1, 1.0),
        (1, -1, -1.0),
        (-1, -1, 1.0),
        (-1, 1, -1.0),
        (1, 0, 0.0),
        (-1, 0, 0.0),
        (1, None, 0.0),
    ],
)
def test_reward_is_relative_to_agent(
    env,
    agent_player,
    winner,
    expected_reward,
):
    env.agent_player = agent_player
    env.opponent_player = -agent_player

    assert env._reward_from_winner(winner) == expected_reward


def test_agent_win_returns_positive_reward(env):
    force_agent_to_start(env)

    env.game.board[5, 0] = 1
    env.game.board[5, 1] = 1
    env.game.board[5, 2] = 1
    env.game.current_player = 1

    observation, reward, terminated, truncated, info = env.step(3)

    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert info["winner"] == 1
    assert info["result"].winner == 1
    assert env.observation_space.contains(observation)


def test_opponent_win_returns_negative_reward():
    opponent = FixedColumnOpponent(column=0)
    env = ConnectFourEnv(opponent=opponent)

    try:
        force_agent_to_start(env)

        # Opponent (-1) already has three vertical pieces in column 0.
        env.game.board[5, 0] = -1
        env.game.board[4, 0] = -1
        env.game.board[3, 0] = -1

        # Give the learner valid pieces without creating a win.
        env.game.board[5, 2] = 1
        env.game.board[5, 4] = 1
        env.game.board[5, 6] = 1

        env.game.current_player = 1

        observation, reward, terminated, truncated, info = env.step(1)

        assert reward == -1.0
        assert terminated is True
        assert truncated is False
        assert info["winner"] == -1
        assert env.observation_space.contains(observation)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Invalid-action forfeiting
# ---------------------------------------------------------------------------

def test_illegal_action_forfeits_turn():
    opponent = FixedColumnOpponent(column=1)
    env = ConnectFourEnv(opponent=opponent)

    try:
        force_agent_to_start(env)

        # Fill column zero completely.
        for _ in range(env.game.num_rows):
            result = env.game.make_move(0)
            assert result is not False

        # The alternating direct moves leave Player 1 to move.
        assert env.agent_turn
        board_before = env.game.get_board()
        piece_count_before = np.count_nonzero(board_before)

        observation, reward, terminated, truncated, info = env.step(0)

        assert reward == pytest.approx(-0.1)
        assert terminated is False
        assert truncated is False
        assert info["illegal_action"] is True
        assert info["agent_action"] == 0
        assert info["opponent_action"] == 1

        # No learner piece was placed, but the opponent received one move.
        assert np.count_nonzero(env.game.get_board()) == piece_count_before + 1
        assert env.game.get_board()[5, 1] == -1
        assert env.agent_turn
        assert env.observation_space.contains(observation)
    finally:
        env.close()


def test_illegal_action_can_end_in_opponent_win():
    opponent = FixedColumnOpponent(column=1)
    env = ConnectFourEnv(opponent=opponent)

    try:
        force_agent_to_start(env)

        # Make column 0 full, so selecting it is illegal.
        for _ in range(env.game.num_rows):
            result = env.game.make_move(0)
            assert result is not False

        # Prepare an immediate vertical win for the opponent in column 1.
        env.game.board[5, 1] = -1
        env.game.board[4, 1] = -1
        env.game.board[3, 1] = -1
        env.game.current_player = env.agent_player

        observation, reward, terminated, truncated, info = env.step(0)

        assert reward == -1.0
        assert terminated is True
        assert truncated is False
        assert info["winner"] == env.opponent_player
        assert info["illegal_action"] is True
        assert env.observation_space.contains(observation)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Opponent behaviour and reproducibility
# ---------------------------------------------------------------------------

def test_random_opponent_only_chooses_legal_actions(env):
    for seed in range(100):
        _, info = env.reset(seed=seed)

        if env.game.get_winner() is not None:
            continue

        action = env._choose_opponent_action()

        assert env.action_space.contains(action)
        assert info["action_mask"][action]


def test_fixed_opponent_receives_its_own_perspective():
    class InspectingOpponent(Opponent):
        def __init__(self):
            self.observation = None
            self.mask = None

        def select_action(self, observation, action_mask, rng):
            self.observation = observation.copy()
            self.mask = action_mask.copy()
            return int(np.flatnonzero(action_mask)[0])

    opponent = InspectingOpponent()
    env = ConnectFourEnv(opponent=opponent)

    try:
        force_agent_to_start(env)
        env.step(3)

        assert opponent.observation is not None
        assert opponent.observation.shape == (2, 6, 7)

        # The learner's piece must appear in the opponent channel.
        assert np.count_nonzero(opponent.observation[0]) == 0
        assert np.count_nonzero(opponent.observation[1]) == 1
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_render_none_returns_none(env):
    env.reset(seed=42)

    assert env.render() is None


def test_rgb_array_render_shape_and_dtype():
    env = ConnectFourEnv(
        opponent=FirstLegalOpponent(),
        render_mode="rgb_array",
    )

    try:
        env.reset(seed=42)
        frame = env.render()

        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8
    finally:
        env.close()


def test_close_clears_renderer():
    env = ConnectFourEnv(
        opponent=FirstLegalOpponent(),
        render_mode="rgb_array",
    )

    env.reset(seed=42)
    env.render()

    assert env.renderer is not None

    env.close()

    assert env.renderer is None


# ---------------------------------------------------------------------------
# Stress tests and SB3 compatibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 42, 100])
def test_random_episodes_always_terminate(seed):
    env = ConnectFourEnv(
        opponent=FirstLegalOpponent(),
    )

    try:
        for episode in range(50):
            observation, info = env.reset(seed=seed + episode)

            terminated = False
            number_of_agent_steps = 0

            while not terminated:
                legal_actions = np.flatnonzero(
                    info["action_mask"]
                )

                assert len(legal_actions) > 0

                action = int(
                    env.np_random.choice(legal_actions)
                )

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                number_of_agent_steps += 1

                assert env.observation_space.contains(observation)
                assert reward in {-1.0, 0.0, 1.0}
                assert truncated is False
                assert number_of_agent_steps <= 21

                if not terminated:
                    assert env.agent_turn

            assert info["winner"] in {-1, 0, 1}
    finally:
        env.close()


def test_no_floating_pieces_after_random_episodes(env):
    for episode in range(100):
        observation, info = env.reset(seed=episode)
        terminated = False

        while not terminated:
            legal_actions = np.flatnonzero(info["action_mask"])
            action = int(env.np_random.choice(legal_actions))

            observation, _, terminated, _, info = env.step(action)

        board = env.game.get_board()

        for column in range(env.game.num_cols):
            for row in range(env.game.num_rows - 1):
                if board[row, column] != 0:
                    assert board[row + 1, column] != 0


def test_environment_passes_sb3_checker():
    env = ConnectFourEnv(
        opponent=FirstLegalOpponent(),
        render_mode=None,
    )

    try:
        check_env(env, warn=True)
    finally:
        env.close()
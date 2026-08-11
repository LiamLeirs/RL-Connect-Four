import numpy as np
import pytest

from src.agents.agents import MiniMaxAgent
from src.envs.connect_four.game import ConnectFour


@pytest.fixture
def game():
    return ConnectFour(
        num_rows=6,
        num_cols=7,
        win_req=4,
    )


@pytest.fixture
def agent(game):
    return MiniMaxAgent(
        game=game,
        renderer=None,
        depth=4,
    )


def play_moves(game, moves):
    """Play a sequence of columns starting from player 1."""
    for action in moves:
        result = game.make_move(action)
        assert result is not False


# -------------------------------------------------------------------------
# Window generation
# -------------------------------------------------------------------------


def test_generates_69_windows(agent):
    assert agent.window_rows.shape == (69, 4)
    assert agent.window_cols.shape == (69, 4)


def test_all_window_coordinates_are_inside_board(agent, game):
    assert np.all(agent.window_rows >= 0)
    assert np.all(agent.window_rows < game.num_rows)

    assert np.all(agent.window_cols >= 0)
    assert np.all(agent.window_cols < game.num_cols)


def test_all_windows_contain_four_unique_cells(agent):
    for rows, cols in zip(
        agent.window_rows,
        agent.window_cols,
    ):
        coordinates = list(zip(rows, cols))

        assert len(set(coordinates)) == 4


def test_windows_are_unique(agent):
    windows = []

    for rows, cols in zip(
        agent.window_rows,
        agent.window_cols,
    ):
        window = tuple(zip(rows.tolist(), cols.tolist()))
        windows.append(window)

    assert len(windows) == len(set(windows))


# -------------------------------------------------------------------------
# Heuristic
# -------------------------------------------------------------------------


def test_empty_board_has_zero_heuristic(agent, game):
    score = agent.heuristic(
        game,
        agentPlayer=1,
    )

    assert score == 0


def test_center_piece_is_preferred_over_edge_piece(game):
    center_game = copy_game(game)
    edge_game = copy_game(game)

    center_agent = MiniMaxAgent(
        game=center_game,
        renderer=None,
        depth=4,
    )

    edge_agent = MiniMaxAgent(
        game=edge_game,
        renderer=None,
        depth=4,
    )

    center_game.make_move(3)
    edge_game.make_move(0)

    center_score = center_agent.heuristic(
        center_game,
        agentPlayer=1,
    )

    edge_score = edge_agent.heuristic(
        edge_game,
        agentPlayer=1,
    )

    assert center_score > edge_score


def test_three_in_row_scores_higher_than_two_in_row(game):
    two_game = copy_game(game)
    three_game = copy_game(game)

    two_agent = MiniMaxAgent(
        game=two_game,
        renderer=None,
        depth=4,
    )

    three_agent = MiniMaxAgent(
        game=three_game,
        renderer=None,
        depth=4,
    )

    # Player 1 gets pieces in columns 0 and 1.
    play_moves(two_game, [0, 6, 1])

    # Player 1 gets pieces in columns 0, 1 and 2.
    play_moves(three_game, [0, 6, 1, 6, 2])

    two_score = two_agent.heuristic(
        two_game,
        agentPlayer=1,
    )

    three_score = three_agent.heuristic(
        three_game,
        agentPlayer=1,
    )

    assert three_score > two_score


def test_opponent_threat_produces_negative_score(agent, game):
    # Player -1 receives three horizontal pieces.
    play_moves(
        game,
        [
            6, 0,
            6, 1,
            5, 2,
        ],
    )

    score = agent.heuristic(
        game,
        agentPlayer=1,
    )

    assert score < 0


# -------------------------------------------------------------------------
# Terminal minimax evaluation
# -------------------------------------------------------------------------


def test_minimax_returns_large_positive_score_for_win(agent, game):
    # Player 1 wins horizontally.
    play_moves(
        game,
        [
            0, 6,
            1, 6,
            2, 5,
            3,
        ],
    )

    score = agent.minimax(
        game,
        depth=3,
        maximizingPlayer=False,
        agentPlayer=1,
    )

    assert score == 1_000_003


def test_minimax_returns_large_negative_score_for_loss(agent, game):
    # Player 1 wins, but evaluate from player -1's perspective.
    play_moves(
        game,
        [
            0, 6,
            1, 6,
            2, 5,
            3,
        ],
    )

    score = agent.minimax(
        game,
        depth=3,
        maximizingPlayer=True,
        agentPlayer=-1,
    )

    assert score == -1_000_003


# -------------------------------------------------------------------------
# Action selection
# -------------------------------------------------------------------------


def test_select_action_returns_legal_action(agent, game):
    rng = np.random.default_rng(42)

    action_mask = game.get_legal_moves()

    action = agent.select_action(
        observation=None,
        action_mask=action_mask,
        rng=rng,
    )

    assert action_mask[action]


def test_minimax_takes_immediate_win(agent, game):
    # Player 1:
    #
    # X X X _ . . .
    #
    # It is player 1's turn and column 3 wins.
    play_moves(
        game,
        [
            0, 6,
            1, 6,
            2, 5,
        ],
    )

    assert game.get_current_player() == 1

    action = agent.select_action(
        observation=None,
        action_mask=game.get_legal_moves(),
        rng=np.random.default_rng(42),
    )

    assert action == 3


def test_minimax_blocks_immediate_loss(agent, game):
    # Player -1 has:
    #
    # O O O _ . . .
    #
    # and it is player 1's turn.
    #
    # Player 1 must play column 3.
    play_moves(
        game,
        [
            6, 0,
            6, 1,
            5, 2,
        ],
    )

    assert game.get_current_player() == 1

    action = agent.select_action(
        observation=None,
        action_mask=game.get_legal_moves(),
        rng=np.random.default_rng(42),
    )

    assert action == 3


def test_select_action_never_selects_full_column(agent, game):
    # Fill column 0 completely.
    for _ in range(game.num_rows):
        result = game.make_move(0)
        assert result is not False

    action_mask = game.get_legal_moves()

    assert not action_mask[0]

    action = agent.select_action(
        observation=None,
        action_mask=action_mask,
        rng=np.random.default_rng(42),
    )

    assert action != 0
    assert action_mask[action]


def test_same_seed_produces_same_tie_break(agent, game):
    action_mask = game.get_legal_moves()

    action_1 = agent.select_action(
        observation=None,
        action_mask=action_mask,
        rng=np.random.default_rng(123),
    )

    action_2 = agent.select_action(
        observation=None,
        action_mask=action_mask,
        rng=np.random.default_rng(123),
    )

    assert action_1 == action_2


def copy_game(game):
    import copy
    return copy.deepcopy(game)
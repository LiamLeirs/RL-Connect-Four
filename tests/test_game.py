import numpy as np
import pytest

from src.envs.connect_four.game import ConnectFour


# ---------------------------------------------------------------------------
# Initialization and reset
# ---------------------------------------------------------------------------

def test_default_board_shape():
    game = ConnectFour()

    assert game.get_board().shape == (6, 7)


def test_custom_board_shape():
    game = ConnectFour(
        num_cols=8,
        num_rows=7,
        win_req=5,
    )

    assert game.get_board().shape == (7, 8)


def test_board_starts_empty():
    game = ConnectFour()

    assert np.all(game.get_board() == 0)


def test_player_one_starts():
    game = ConnectFour()

    assert game.get_current_player() == 1


def test_game_starts_without_winner():
    game = ConnectFour()

    assert game.get_winner() is None


def test_reset_restores_initial_state():
    game = ConnectFour()

    game.make_move(3)
    game.make_move(4)
    game.reset()

    assert np.all(game.get_board() == 0)
    assert game.get_current_player() == 1
    assert game.get_winner() is None


def test_board_must_be_large_enough_for_win_requirement():
    with pytest.raises(AssertionError):
        ConnectFour(
            num_cols=3,
            num_rows=6,
            win_req=4,
        )

    with pytest.raises(AssertionError):
        ConnectFour(
            num_cols=7,
            num_rows=3,
            win_req=4,
        )


# ---------------------------------------------------------------------------
# Legal moves and gravity
# ---------------------------------------------------------------------------

def test_all_columns_initially_legal():
    game = ConnectFour()

    expected = np.ones(7, dtype=bool)

    np.testing.assert_array_equal(
        game.get_legal_moves(),
        expected,
    )


@pytest.mark.parametrize("column", range(7))
def test_first_piece_falls_to_bottom(column):
    game = ConnectFour()

    game.make_move(column)

    assert game.get_board()[5, column] == 1
    assert np.count_nonzero(game.get_board()) == 1


def test_pieces_stack_from_bottom_up():
    game = ConnectFour()

    game.make_move(2)  # Player 1
    game.make_move(2)  # Player -1
    game.make_move(2)  # Player 1

    board = game.get_board()

    assert board[5, 2] == 1
    assert board[4, 2] == -1
    assert board[3, 2] == 1
    assert np.all(board[:3, 2] == 0)


def test_column_becomes_illegal_when_full():
    game = ConnectFour()

    for _ in range(game.num_rows):
        game.make_move(0)

    assert not game.get_legal_moves()[0]


def test_other_columns_remain_legal_when_one_column_is_full():
    game = ConnectFour()

    for _ in range(game.num_rows):
        game.make_move(0)

    legal_moves = game.get_legal_moves()

    assert legal_moves[0] == np.bool_(False)
    assert np.all(legal_moves[1:])


def test_move_in_full_column_does_not_change_board():
    game = ConnectFour()

    for _ in range(game.num_rows):
        game.make_move(0)

    board_before = game.get_board().copy()
    player_before = game.get_current_player()

    game.make_move(0)

    np.testing.assert_array_equal(
        game.get_board(),
        board_before,
    )
    assert game.get_current_player() == player_before


@pytest.mark.parametrize("column", [-1, 7, 20])
def test_column_outside_board_raises_value_error(column):
    game = ConnectFour()

    with pytest.raises(ValueError):
        game.make_move(column)


# ---------------------------------------------------------------------------
# Player switching
# ---------------------------------------------------------------------------

def test_player_switches_after_valid_move():
    game = ConnectFour()

    assert game.get_current_player() == 1

    game.make_move(0)

    assert game.get_current_player() == -1

    game.make_move(1)

    assert game.get_current_player() == 1


def test_player_does_not_switch_after_illegal_full_column_move():
    game = ConnectFour()

    for _ in range(game.num_rows):
        game.make_move(0)

    current_player = game.get_current_player()

    game.make_move(0)

    assert game.get_current_player() == current_player


# ---------------------------------------------------------------------------
# Horizontal wins
# ---------------------------------------------------------------------------

def test_player_one_horizontal_win():
    game = ConnectFour()

    # Player 1: 0, 1, 2, 3
    # Player -1: 6, 6, 6
    moves = [0, 6, 1, 6, 2, 6, 3]

    for move in moves:
        game.make_move(move)

    assert game.get_winner() == 1


def test_player_two_horizontal_win():
    game = ConnectFour()

    # Player -1 occupies columns 0, 1, 2 and 3 on the bottom row.
    moves = [6, 0, 6, 1, 5, 2, 5, 3]

    for move in moves:
        game.make_move(move)

    assert game.get_winner() == -1


def test_horizontal_gap_is_not_a_win():
    game = ConnectFour()

    game.board[5, 0] = 1
    game.board[5, 1] = 1
    game.board[5, 3] = 1
    game.board[5, 4] = 1
    game.current_player = 1

    assert not game._check_win(5, 4)


def test_horizontal_sequence_longer_than_four_is_a_win():
    game = ConnectFour()

    game.board[5, 0:5] = 1
    game.current_player = 1

    assert game._check_win(5, 4)


# ---------------------------------------------------------------------------
# Vertical wins
# ---------------------------------------------------------------------------

def test_player_one_vertical_win():
    game = ConnectFour()

    moves = [0, 1, 0, 1, 0, 1, 0]

    for move in moves:
        game.make_move(move)

    assert game.get_winner() == 1


def test_player_two_vertical_win():
    game = ConnectFour()

    moves = [1, 0, 1, 0, 2, 0, 2, 0]

    for move in moves:
        game.make_move(move)

    assert game.get_winner() == -1


def test_three_vertical_pieces_are_not_a_win():
    game = ConnectFour()

    game.board[5, 0] = 1
    game.board[4, 0] = 1
    game.board[3, 0] = 1
    game.current_player = 1

    assert not game._check_win(3, 0)


# ---------------------------------------------------------------------------
# Diagonal wins
# ---------------------------------------------------------------------------

def test_down_right_diagonal_win():
    """
    Tests this diagonal:

    X . . .
    . X . .
    . . X .
    . . . X
    """
    game = ConnectFour()

    game.board[2, 0] = 1
    game.board[3, 1] = 1
    game.board[4, 2] = 1
    game.board[5, 3] = 1
    game.current_player = 1

    assert game._check_win(5, 3)


def test_up_right_diagonal_win():
    """
    Tests this diagonal:

    . . . X
    . . X .
    . X . .
    X . . .
    """
    game = ConnectFour()

    game.board[5, 0] = 1
    game.board[4, 1] = 1
    game.board[3, 2] = 1
    game.board[2, 3] = 1
    game.current_player = 1

    assert game._check_win(2, 3)


def test_player_two_diagonal_win():
    game = ConnectFour()

    game.board[5, 0] = -1
    game.board[4, 1] = -1
    game.board[3, 2] = -1
    game.board[2, 3] = -1
    game.current_player = -1

    assert game._check_win(2, 3)


def test_diagonal_with_gap_is_not_a_win():
    game = ConnectFour()

    game.board[5, 0] = 1
    game.board[4, 1] = 1
    # Missing board[3, 2]
    game.board[2, 3] = 1
    game.board[1, 4] = 1
    game.current_player = 1

    assert not game._check_win(1, 4)


def test_diagonal_at_left_edge():
    game = ConnectFour()

    game.board[5, 0] = 1
    game.board[4, 1] = 1
    game.board[3, 2] = 1
    game.board[2, 3] = 1
    game.current_player = 1

    assert game._check_win(5, 0)


def test_diagonal_at_right_edge():
    game = ConnectFour()

    game.board[5, 6] = 1
    game.board[4, 5] = 1
    game.board[3, 4] = 1
    game.board[2, 3] = 1
    game.current_player = 1

    assert game._check_win(5, 6)


# ---------------------------------------------------------------------------
# Terminal game behaviour
# ---------------------------------------------------------------------------

def test_player_does_not_switch_after_winning_move():
    game = ConnectFour()

    moves = [0, 6, 1, 6, 2, 5, 3]

    for move in moves:
        game.make_move(move)

    assert game.get_winner() == 1
    assert game.get_current_player() == 1


def test_moves_after_win_raises_runtime_error():
    game = ConnectFour()

    moves = [0, 6, 1, 6, 2, 5, 3]

    for move in moves:
        game.make_move(move)

    with pytest.raises(RuntimeError):
        game.make_move(4)
    


def test_draw_is_detected():
    game = ConnectFour()

    # This sequence fills the entire board without producing a four-in-a-row.
    draw_sequence = [
        3, 2, 5, 4, 0, 1, 6,
        1, 1, 0, 0, 6, 6, 4,
        6, 0, 5, 0, 1, 5, 0,
        3, 3, 5, 2, 3, 1, 1,
        3, 5, 3, 6, 5, 2, 4,
        6, 4, 2, 4, 2, 2, 4,
    ]

    for move in draw_sequence:
        game.make_move(move)

    assert game.get_winner() == 0
    assert not np.any(game.get_legal_moves())
    assert np.all(game.get_board() != 0)


def test_moves_after_draw_are_ignored():
    game = ConnectFour()

    draw_sequence = [
        3, 2, 5, 4, 0, 1, 6,
        1, 1, 0, 0, 6, 6, 4,
        6, 0, 5, 0, 1, 5, 0,
        3, 3, 5, 2, 3, 1, 1,
        3, 5, 3, 6, 5, 2, 4,
        6, 4, 2, 4, 2, 2, 4,
    ]

    for move in draw_sequence:
        game.make_move(move)

    with pytest.raises(RuntimeError):
        game.make_move(0)

    assert game.get_winner() == 0


# ---------------------------------------------------------------------------
# Configurable win requirements
# ---------------------------------------------------------------------------

def test_three_in_a_row_with_win_requirement_three():
    game = ConnectFour(
        num_cols=5,
        num_rows=4,
        win_req=3,
    )

    moves = [0, 4, 1, 4, 2]

    for move in moves:
        game.make_move(move)

    assert game.get_winner() == 1


def test_three_in_a_row_is_not_enough_with_default_rules():
    game = ConnectFour()

    moves = [0, 6, 1, 6, 2]

    for move in moves:
        game.make_move(move)

    assert game.get_winner() is None


def test_five_in_a_row_with_win_requirement_five():
    game = ConnectFour(
        num_cols=8,
        num_rows=6,
        win_req=5,
    )

    game.board[5, 0:5] = 1
    game.current_player = 1

    assert game._check_win(5, 4)


# ---------------------------------------------------------------------------
# Randomized robustness tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [1, 2, 3, 42, 100])
def test_random_games_always_terminate(seed):
    rng = np.random.default_rng(seed)

    for _ in range(100):
        game = ConnectFour()
        number_of_moves = 0

        while game.get_winner() is None:
            legal_columns = np.flatnonzero(
                game.get_legal_moves()
            )

            assert len(legal_columns) > 0

            column = int(rng.choice(legal_columns))
            game.make_move(column)
            number_of_moves += 1

            assert number_of_moves <= (
                game.num_rows * game.num_cols
            )

        assert game.get_winner() in {-1, 0, 1}


def test_piece_count_increases_by_one_after_valid_move():
    game = ConnectFour()

    for column in [3, 2, 3, 4, 1]:
        count_before = np.count_nonzero(game.get_board())

        game.make_move(column)

        count_after = np.count_nonzero(game.get_board())

        assert count_after == count_before + 1


def test_board_only_contains_valid_values():
    rng = np.random.default_rng(42)

    for _ in range(100):
        game = ConnectFour()

        while game.get_winner() is None:
            legal_columns = np.flatnonzero(
                game.get_legal_moves()
            )

            column = int(rng.choice(legal_columns))
            game.make_move(column)

            assert np.all(
                np.isin(
                    game.get_board(),
                    [-1, 0, 1],
                )
            )


def test_piece_counts_never_differ_by_more_than_one():
    rng = np.random.default_rng(42)

    for _ in range(100):
        game = ConnectFour()

        while game.get_winner() is None:
            legal_columns = np.flatnonzero(
                game.get_legal_moves()
            )

            game.make_move(
                int(rng.choice(legal_columns))
            )

            player_one_count = np.count_nonzero(
                game.get_board() == 1
            )
            player_two_count = np.count_nonzero(
                game.get_board() == -1
            )

            assert player_one_count in {
                player_two_count,
                player_two_count + 1,
            }


def test_no_floating_pieces_exist_after_random_games():
    rng = np.random.default_rng(123)

    for _ in range(100):
        game = ConnectFour()

        while game.get_winner() is None:
            legal_columns = np.flatnonzero(
                game.get_legal_moves()
            )

            game.make_move(
                int(rng.choice(legal_columns))
            )

        board = game.get_board()

        for column in range(game.num_cols):
            for row in range(game.num_rows - 1):
                if board[row, column] != 0:
                    assert board[row + 1, column] != 0
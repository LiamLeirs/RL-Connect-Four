import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class MoveResult:
    row: int
    col: int
    player: int
    winner: int | None


class ConnectFour:
    def __init__(self, num_cols=7, num_rows=6, win_req=4):
        assert num_cols >= win_req and num_rows >= win_req
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.win_req = win_req
        self.reset()

    def reset(self):
        self.board = np.zeros((self.num_rows, self.num_cols), dtype=np.int8)
        self.current_player = 1
        self.winner = None

    def skip_turn(self):
        self.current_player *= -1

    def get_board(self):
        return self.board.copy()

    def get_winner(self):
        return self.winner

    def get_current_player(self):
        return self.current_player

    def get_legal_moves(self):
        return self.board[0] == 0

    def make_move(self, col):
        if not 0 <= col < self.num_cols:
            raise ValueError(f"Column must be between 0 and {self.num_cols - 1}.")

        if self.winner is not None:
            raise RuntimeError("The game has already ended.")
        
        if not self.get_legal_moves()[col]:
            return False

        player = self.current_player
        row = 0
        while self.board[row, col] == 0:
            if row == self.num_rows-1 or self.board[row+1, col] != 0:
                break
            row += 1
        self.board[row, col] = self.current_player
        if self._check_win(row, col):
            self.winner = self.current_player
        elif not np.any(self.get_legal_moves()):
            self.winner = 0 # Draw
        else:
            self.current_player *= -1

        return MoveResult(
        row=row,
        col=col,
        player=player,
        winner=self.winner,
    )

    def _has_win(self, line):
        if len(line) < self.win_req:
            return False
        mask = (np.array(line) == self.current_player)
        # Length matches the number of possible starting positions for a win
        final_mask = np.ones(len(mask)-self.win_req+1, dtype=bool)
        # A position in final_mask stays True ONLY if the player occupies
        # that index and all subsequent indices required to win.
        for i in range(self.win_req):
            final_mask &= mask[i:len(mask)-self.win_req + i + 1]
        return np.any(final_mask)

    def _check_win(self, row, col):
        # Define boundaries for local search windows (up to win_req steps away)
        # to avoid searching the entire board unnecessarily.
        minRow = row-self.win_req+1 if row-self.win_req+1 >= 0 else 0
        maxRow = row+self.win_req if row+self.win_req < self.num_rows else self.num_rows
        minCol = col-self.win_req+1 if col-self.win_req+1 >= 0 else 0
        maxCol = col+self.win_req if col+self.win_req < self.num_cols else self.num_cols
        vertical = self.board[minRow:maxRow, col]
        horizontal = self.board[row, minCol:maxCol]
        # Diagonal 1 (/): Bottom-Left to Top-Right
        diag1 = [self.board[row+i, col-i] for i in range(-self.win_req+1, self.win_req) if row+i < self.num_rows and col-i >= 0 and row+i >= 0 and col-i < self.num_cols]
        # Diagonal 2 (\): Top-Left to Bottom-Right
        diag2 = [self.board[row+i, col+i] for i in range(-self.win_req+1, self.win_req) if row+i < self.num_rows and col+i < self.num_cols and row+i >= 0 and col+i >= 0]
        return self._has_win(vertical) or self._has_win(horizontal) or self._has_win(diag1) or self._has_win(diag2)


if __name__ == "__main__":
    env = ConnectFour()
    env.reset()
    env.make_move(3)
    print(env.get_board())
    env.make_move(3)
        
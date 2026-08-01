from __future__ import annotations

import numpy as np
import pygame


class ConnectFourRenderer:
    """Draws a Connect Four board using Pygame."""

    BACKGROUND_COLOR = (235, 235, 235)
    BOARD_COLOR = (25, 85, 190)
    EMPTY_COLOR = (245, 245, 245)
    PLAYER_ONE_COLOR = (220, 45, 45)
    PLAYER_TWO_COLOR = (245, 205, 35)
    TEXT_COLOR = (30, 30, 30)

    def __init__(
        self,
        num_rows: int = 6,
        num_cols: int = 7,
        cell_size: int = 100,
        top_margin: int = 100,
        fps: int = 60,
    ) -> None:
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.cell_size = cell_size
        self.top_margin = top_margin
        self.fps = fps

        self.board_width = self.num_cols * self.cell_size
        self.board_height = self.num_rows * self.cell_size

        self.window_width = self.board_width
        self.window_height = self.top_margin + self.board_height

        self.window: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.font: pygame.font.Font | None = None

        self.hovered_column: int | None = None
        self.hover_alpha = 0.0
        self.hover_target_alpha = 0.0
        self.hover_fade_speed = 500.0  # alpha units per second
        self.delta_time = 0.0

        self.falling_piece = None
        self.drop_speed = 1400.0  # pixels per second

    def open(self) -> None:
        """Initializes Pygame and opens the window."""
        if self.window is not None:
            return

        pygame.init()

        self.window = pygame.display.set_mode(
            (self.window_width, self.window_height)
        )
        pygame.display.set_caption("Connect Four")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 40)

    def draw(
        self,
        board: np.ndarray,
        current_player: int | None = None,
        winner: int | None = None,
    ) -> None:
        """Draws one complete frame."""
        self.open()

        assert self.window is not None
        assert self.clock is not None
        assert self.font is not None

        delta_ms = self.clock.tick(self.fps)
        self.delta_time = delta_ms / 1000.0

        self.update_hover(pygame.mouse.get_pos())
        self.update_hover_animation(self.delta_time)

        self.update_drop_animation(self.delta_time)

        self.window.fill(self.BACKGROUND_COLOR)

        self._draw_status(current_player, winner)
        self._draw_board(board)
        self._draw_hover()
        self._draw_falling_piece()

        pygame.display.flip()
        
    def _draw_status(
        self,
        current_player: int | None,
        winner: int | None,
    ) -> None:
        assert self.window is not None
        assert self.font is not None

        if winner == 1:
            message = "Player 1 wins"
        elif winner == -1:
            message = "Player 2 wins"
        elif winner == 0:
            message = "Draw"
        elif current_player == 1:
            message = "Player 1's turn"
        elif current_player == -1:
            message = "Player 2's turn"
        else:
            message = "Connect Four"

        text_surface = self.font.render(
            message,
            True,
            self.TEXT_COLOR,
        )

        text_rect = text_surface.get_rect(
            center=(self.window_width // 2, self.top_margin // 2)
        )

        self.window.blit(text_surface, text_rect)

    def _draw_board(self, board: np.ndarray) -> None:
        assert self.window is not None

        board_rect = pygame.Rect(
            0,
            self.top_margin,
            self.board_width,
            self.board_height,
        )

        pygame.draw.rect(
            self.window,
            self.BOARD_COLOR,
            board_rect,
        )

        radius = self.cell_size // 2 - 10

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                value = board[row, col]

                is_falling_target = (
                    self.falling_piece is not None
                    and row == self.falling_piece["row"]
                    and col == self.falling_piece["column"]
                )

                if is_falling_target:
                    value = 0

                center_x = col * self.cell_size + self.cell_size // 2
                center_y = (
                    self.top_margin
                    + row * self.cell_size
                    + self.cell_size // 2
                )

                if value == 1:
                    color = self.PLAYER_ONE_COLOR
                elif value == -1:
                    color = self.PLAYER_TWO_COLOR
                else:
                    color = self.EMPTY_COLOR

                pygame.draw.circle(
                    self.window,
                    color,
                    (center_x, center_y),
                    radius,
                )

    def start_drop_animation(
    self,
    row: int,
    column: int,
    player: int,
) -> None:
        self.falling_piece = {
            "row": row,
            "column": column,
            "player": player,
            "y": self.top_margin - self.cell_size // 2,
        }

    def update_drop_animation(self, delta_time: float) -> None:
        if self.falling_piece is None:
            return

        target_y = (
            self.top_margin
            + self.falling_piece["row"] * self.cell_size
            + self.cell_size // 2
        )

        self.falling_piece["y"] += self.drop_speed * delta_time

        if self.falling_piece["y"] >= target_y:
            self.falling_piece["y"] = target_y
            self.falling_piece = None

    def _draw_falling_piece(self) -> None:
        if self.falling_piece is None:
            return

        assert self.window is not None

        player = self.falling_piece["player"]

        if player == 1:
            color = self.PLAYER_ONE_COLOR
        else:
            color = self.PLAYER_TWO_COLOR

        center_x = (
            self.falling_piece["column"] * self.cell_size
            + self.cell_size // 2
        )

        center_y = int(self.falling_piece["y"])

        pygame.draw.circle(
            self.window,
            color,
            (center_x, center_y),
            self.cell_size // 2 - 10,
        )

    def _draw_hover(self) -> None:
        if self.hovered_column is None or self.hover_alpha <= 0:
            return

        assert self.window is not None

        overlay = pygame.Surface(
            (self.cell_size, self.window_height),
            pygame.SRCALPHA,
        )

        overlay.fill(
            (255, 255, 255, int(self.hover_alpha))
        )

        x = self.hovered_column * self.cell_size

        self.window.blit(overlay, (x, 0))

    def update_hover_animation(self, delta_time: float) -> None:
        difference = self.hover_target_alpha - self.hover_alpha

        max_change = self.hover_fade_speed * delta_time

        if abs(difference) <= max_change:
            self.hover_alpha = self.hover_target_alpha
        elif difference > 0:
            self.hover_alpha += max_change
        else:
            self.hover_alpha -= max_change

    def update_hover(self, mouse_position: tuple[int, int]) -> None:
        mouse_x, mouse_y = mouse_position

        if (
            0 <= mouse_x < self.window_width
            and 0 <= mouse_y < self.window_height
        ):
            self.hovered_column = mouse_x // self.cell_size
            self.hover_target_alpha = 80.0
        else:
            self.hovered_column = None
            self.hover_target_alpha = 0.0

    def process_events(self) -> bool:
        """
        Processes window events.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        return True

    def close(self) -> None:
        """Closes the Pygame window."""
        pygame.quit()
        self.window = None
        self.clock = None
        self.font = None
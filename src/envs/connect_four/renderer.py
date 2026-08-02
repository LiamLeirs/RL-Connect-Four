from __future__ import annotations

from typing import TypedDict
from enum import Enum, auto
import numpy as np
import pygame

class RendererEvent(Enum):
    NONE = auto()
    QUIT = auto()
    RESET = auto()

class FallingPiece(TypedDict):
    row: int
    column: int
    player: int
    y: float
    velocity: float


class ConnectFourRenderer:
    """Render a Connect Four board with Pygame."""

    BACKGROUND_COLOR = (235, 235, 235)
    BOARD_COLOR = (25, 85, 190)
    EMPTY_COLOR = (245, 245, 245)
    PLAYER_ONE_COLOR = (220, 45, 45)
    PLAYER_TWO_COLOR = (245, 205, 35)
    TEXT_COLOR = (30, 30, 30)

    HOVER_COLOR = (255, 255, 255)

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

        # Hover animation
        self.hovered_column: int | None = None
        self.hover_alpha = 0.0
        self.hover_target_alpha = 0.0
        self.hover_fade_speed = 500.0

        # Falling-piece animation
        self.falling_piece: FallingPiece | None = None
        self.drop_gravity = 3_000.0

        self.clicked_column: int | None = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        """
        Initialize the Pygame components needed for drawing.

        This does not open a display window, so it can also be used for
        rgb_array rendering.
        """
        if not pygame.get_init():
            pygame.init()

        if not pygame.font.get_init():
            pygame.font.init()

        if self.font is None:
            self.font = pygame.font.Font(None, 40)

        if self.clock is None:
            self.clock = pygame.time.Clock()

    def open(self) -> None:
        """Open the live Pygame display window."""
        self._ensure_initialized()

        if self.window is not None:
            return

        self.window = pygame.display.set_mode(
            (self.window_width, self.window_height)
        )

        pygame.display.set_caption("Connect Four")

    # ------------------------------------------------------------------
    # Public rendering methods
    # ------------------------------------------------------------------

    def draw_human(
    self,
    board: np.ndarray,
    current_player: int | None = None,
    winner: int | None = None,
) -> None:
        """Draw the current visual state to the Pygame window."""
        self.open()

        assert self.window is not None

        frame = self._create_frame(
            board=board,
            current_player=current_player,
            winner=winner,
            include_hover=True,
            include_animation=True,
        )

        self.window.blit(frame, (0, 0))
        pygame.display.flip()

        

    def render_rgb_array(
        self,
        board: np.ndarray,
        current_player: int | None = None,
        winner: int | None = None,
        include_animation: bool = True,
    ) -> np.ndarray:
        """
        Return the current rendered frame as an RGB NumPy array.

        This method does not open a window and does not advance animation
        time automatically. Call `update(delta_time)` externally when
        recording an animated sequence.
        """
        self._ensure_initialized()

        frame_surface = self._create_frame(
            board=board,
            current_player=current_player,
            winner=winner,
            include_hover=False,
            include_animation=include_animation,
        )

        # Pygame returns arrays in (width, height, channels) order.
        frame = pygame.surfarray.array3d(frame_surface)

        # Video and Gymnasium tools normally expect (height, width, channels).
        frame = np.transpose(frame, (1, 0, 2))

        return np.ascontiguousarray(
            frame,
            dtype=np.uint8,
        )

    def draw(
        self,
        board: np.ndarray,
        current_player: int | None = None,
        winner: int | None = None,
    ) -> None:
        self.draw_human(
            board=board,
            current_player=current_player,
            winner=winner,
        )

    # ------------------------------------------------------------------
    # Frame creation
    # ------------------------------------------------------------------

    def _create_frame(
        self,
        board: np.ndarray,
        current_player: int | None,
        winner: int | None,
        *,
        include_hover: bool,
        include_animation: bool,
    ) -> pygame.Surface:
        """Create and return one complete Pygame frame."""
        self._ensure_initialized()

        expected_shape = (self.num_rows, self.num_cols)

        if board.shape != expected_shape:
            raise ValueError(
                f"Expected board shape {expected_shape}, "
                f"but received {board.shape}."
            )

        canvas = pygame.Surface(
            (self.window_width, self.window_height)
        )

        canvas.fill(self.BACKGROUND_COLOR)

        self._draw_status(
            surface=canvas,
            current_player=current_player,
            winner=winner,
        )

        self._draw_board(
            surface=canvas,
            board=board,
            hide_falling_target=include_animation,
        )

        if include_hover:
            self._draw_hover(canvas)

        if include_animation:
            self._draw_falling_piece(canvas)

        return canvas

    # ------------------------------------------------------------------
    # Static drawing
    # ------------------------------------------------------------------

    def _draw_status(
        self,
        surface: pygame.Surface,
        current_player: int | None,
        winner: int | None,
    ) -> None:
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
            center=(
                self.window_width // 2,
                self.top_margin // 2,
            )
        )

        surface.blit(text_surface, text_rect)

    def _draw_board(
        self,
        surface: pygame.Surface,
        board: np.ndarray,
        *,
        hide_falling_target: bool,
    ) -> None:
        board_rect = pygame.Rect(
            0,
            self.top_margin,
            self.board_width,
            self.board_height,
        )

        pygame.draw.rect(
            surface,
            self.BOARD_COLOR,
            board_rect,
        )

        radius = self.cell_size // 2 - 10

        for row in range(self.num_rows):
            for column in range(self.num_cols):
                value = int(board[row, column])

                is_falling_target = (
                    hide_falling_target
                    and self.falling_piece is not None
                    and row == self.falling_piece["row"]
                    and column == self.falling_piece["column"]
                )

                # The game already contains the placed piece. Hide it while
                # its animated version is falling.
                if is_falling_target:
                    value = 0

                center_x = (
                    column * self.cell_size
                    + self.cell_size // 2
                )

                center_y = (
                    self.top_margin
                    + row * self.cell_size
                    + self.cell_size // 2
                )

                color = self._color_for_value(value)

                pygame.draw.circle(
                    surface,
                    color,
                    (center_x, center_y),
                    radius,
                )

    def _color_for_value(
        self,
        value: int,
    ) -> tuple[int, int, int]:
        if value == 1:
            return self.PLAYER_ONE_COLOR

        if value == -1:
            return self.PLAYER_TWO_COLOR

        return self.EMPTY_COLOR

    # ------------------------------------------------------------------
    # Hover animation
    # ------------------------------------------------------------------

    def update_hover(
        self,
        mouse_position: tuple[int, int],
    ) -> None:
        mouse_x, mouse_y = mouse_position

        mouse_inside_window = (
            0 <= mouse_x < self.window_width
            and 0 <= mouse_y < self.window_height
        )

        if mouse_inside_window:
            self.hovered_column = (
                mouse_x // self.cell_size
            )
            self.hover_target_alpha = 80.0
        else:
            self.hovered_column = None
            self.hover_target_alpha = 0.0

    def _update_hover_animation(
        self,
        delta_time: float,
    ) -> None:
        difference = (
            self.hover_target_alpha
            - self.hover_alpha
        )

        max_change = (
            self.hover_fade_speed
            * delta_time
        )

        if abs(difference) <= max_change:
            self.hover_alpha = self.hover_target_alpha
        elif difference > 0:
            self.hover_alpha += max_change
        else:
            self.hover_alpha -= max_change

    def _draw_hover(
        self,
        surface: pygame.Surface,
    ) -> None:
        if (
            self.hovered_column is None
            or self.hover_alpha <= 0
        ):
            return

        overlay = pygame.Surface(
            (self.cell_size, self.window_height),
            pygame.SRCALPHA,
        )

        overlay.fill(
            (
                self.HOVER_COLOR[0],
                self.HOVER_COLOR[1],
                self.HOVER_COLOR[2],
                int(self.hover_alpha),
            )
        )

        x_position = (
            self.hovered_column
            * self.cell_size
        )

        surface.blit(
            overlay,
            (x_position, 0),
        )

    # ------------------------------------------------------------------
    # Falling-piece animation
    # ------------------------------------------------------------------

    def start_drop_animation(
        self,
        row: int,
        column: int,
        player: int,
    ) -> None:
        """
        Begin animating a piece toward its final board location.

        The board should already contain the placed piece. The renderer
        temporarily hides that final board position until the animation
        finishes.
        """
        if not 0 <= row < self.num_rows:
            raise ValueError(f"Invalid target row: {row}")

        if not 0 <= column < self.num_cols:
            raise ValueError(
                f"Invalid target column: {column}"
            )

        if player not in {-1, 1}:
            raise ValueError(
                f"Player must be -1 or 1, received {player}."
            )

        self.falling_piece = {
            "row": row,
            "column": column,
            "player": player,
            "y": float(
                self.top_margin
                - self.cell_size // 2
            ),
            "velocity": 0.0,
        }

    def _update_drop_animation(
        self,
        delta_time: float,
    ) -> None:
        if self.falling_piece is None:
            return

        target_y = (
            self.top_margin
            + self.falling_piece["row"]
            * self.cell_size
            + self.cell_size // 2
        )

        self.falling_piece["velocity"] += (
            self.drop_gravity * delta_time
        )

        self.falling_piece["y"] += (
            self.falling_piece["velocity"]
            * delta_time
        )

        if self.falling_piece["y"] >= target_y:
            self.falling_piece = None

    def _draw_falling_piece(
        self,
        surface: pygame.Surface,
    ) -> None:
        if self.falling_piece is None:
            return

        color = self._color_for_value(
            self.falling_piece["player"]
        )

        center_x = (
            self.falling_piece["column"]
            * self.cell_size
            + self.cell_size // 2
        )

        center_y = int(
            self.falling_piece["y"]
        )

        pygame.draw.circle(
            surface,
            color,
            (center_x, center_y),
            self.cell_size // 2 - 10,
        )

    @property
    def is_animating(self) -> bool:
        return self.falling_piece is not None

    # ------------------------------------------------------------------
    # Animation update
    # ------------------------------------------------------------------

    def update(self) -> bool:
        """
        Process input and advance animations by one frame.

        Returns False if the user closes the window.
        """
        self.open()

        assert self.clock is not None

        delta_time = self.clock.tick(self.fps) / 1000.0

        event = self.process_events()

        self.update_hover(pygame.mouse.get_pos())
        self._update_hover_animation(delta_time)
        self._update_drop_animation(delta_time)

        return event

    # ------------------------------------------------------------------
    # Events and cleanup
    # ------------------------------------------------------------------

    def process_events(self) -> bool:
        """
        Process basic window events.
        """
        self.open()
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                return RendererEvent.QUIT

            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.clicked_column = event.pos[0] // self.cell_size

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return RendererEvent.RESET

        return RendererEvent.NONE

    def consume_click(self):
        click = self.clicked_column
        self.clicked_column = None
        return click

    def close(self) -> None:
        """Release Pygame resources."""
        if self.window is not None:
            pygame.display.quit()

        if pygame.get_init():
            pygame.quit()

        self.window = None
        self.clock = None
        self.font = None

        self.hovered_column = None
        self.hover_alpha = 0.0
        self.hover_target_alpha = 0.0

        self.falling_piece = None
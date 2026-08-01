from src.envs.connect_four.game import ConnectFour, MoveResult
from src.envs.connect_four.renderer import ConnectFourRenderer
import pygame

def main():
    game = ConnectFour()
    renderer = ConnectFourRenderer(
        num_rows=game.num_rows,
        num_cols=game.num_cols,
    )

    renderer.open()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and game.get_winner() is None and renderer.falling_piece is None:
                column = event.pos[0] // renderer.cell_size
                if (0 <= column < game.num_cols) and game.get_legal_moves()[column]:
                    result =game.make_move(column)
                    if result:
                        row, col, player, winner = result.row, result.col, result.player, result.winner
                        renderer.start_drop_animation(row, col, player)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                game.reset()
        renderer.draw(
            game.get_board(),
            game.get_current_player(),
            game.get_winner(),
        )
    renderer.close()

if __name__ == "__main__":
    main()
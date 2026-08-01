import time

from src.envs.connect_four.game import ConnectFour
from src.envs.connect_four.renderer import ConnectFourRenderer


def main() -> None:
    game = ConnectFour()
    renderer = ConnectFourRenderer(
        num_rows=game.num_rows,
        num_cols=game.num_cols,
    )

    test_moves = [3, 2, 3, 2, 4, 2, 5]

    running = True

    renderer.draw(
        game.get_board(),
        game.get_current_player(),
        game.get_winner(),
    )

    for move in test_moves:
        if not running:
            break

        game.make_move(move)

        renderer.draw(
            game.get_board(),
            game.get_current_player(),
            game.get_winner(),
        )

        start_time = time.time()

        while time.time() - start_time < 0.5:
            running = renderer.process_events()

            if not running:
                break

    while running:
        running = renderer.process_events()

        renderer.draw(
            game.get_board(),
            game.get_current_player(),
            game.get_winner(),
        )

    renderer.close()


if __name__ == "__main__":
    main()
import argparse
import json
from pathlib import Path

from sb3_contrib import MaskablePPO

from src.agents.agents import (
    MiniMaxAgent,
    ModelAgent,
    RandomAgent,
    TacticalAgent,
)
from src.envs.connect_four_env import ConnectFourEnv
from src.evaluation.evaluator import evaluate_agent
from src.self_play.self_play_manager import (
    SelfPlayCallback,
    SelfPlayManager,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Number of environment transitions.",
    )

    parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50_000,
    )

    parser.add_argument(
        "--rating-freq",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=200.0,
    )

    parser.add_argument(
        "--K",
        type=float,
        default=8,
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/ppo_selfplay/final_model"),
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("models/ppo_selfplay/checkpoints"),
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/ppo_selfplay"),
    )

    parser.add_argument(
        "--tensorboard-log",
        type=Path,
        default=Path("runs/ppo_selfplay"),
    )

    return parser.parse_args()


def evaluate_final_model(
    model_path,
    num_episodes,
    seed,
):
    model = MaskablePPO.load(model_path)

    agent = ModelAgent(
        model=model,
        deterministic=True,
    )

    opponents = {
        "random": RandomAgent(),
        "tactical": TacticalAgent(),
        "minimax_2": MiniMaxAgent(depth=2),
        "minimax_4": MiniMaxAgent(depth=4),
    }

    results = {}

    for index, (name, opponent) in enumerate(opponents.items()):
        env = ConnectFourEnv(
            opponent=opponent,
            render_mode=None,
        )

        try:
            results[name] = evaluate_agent(
                env=env,
                agent=agent,
                num_episodes=num_episodes,
                seed=seed + index * 10_000,
                varied_starting_states=True
            )
        finally:
            env.close()

    return results


def main():
    args = parse_args()

    args.model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.tensorboard_log.mkdir(
        parents=True,
        exist_ok=True,
    )

    manager = SelfPlayManager(
        window_size=args.window_size,
        temperature=args.temperature,
        K=args.K,

    )

    train_env = ConnectFourEnv(
        opponent_provider=manager,
        render_mode=None,
    )

    model = MaskablePPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.lr,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=args.gamma,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(args.tensorboard_log),
        policy_kwargs={
            "net_arch": {
                "pi": [128, 128],
                "vf": [128, 128],
            }
        },
    )

    callback = SelfPlayCallback(
        manager=manager,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_dir=args.checkpoint_dir,
        rating_freq=args.rating_freq,
        results_path=args.results_dir / "training_history.csv"
    )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            progress_bar=True,
            tb_log_name="maskable_ppo_selfplay",
            callback=callback,
        )

        model.save(args.model_path)

    finally:
        train_env.close()

    results = evaluate_final_model(
        model_path=args.model_path,
        num_episodes=args.num_eval_episodes,
        seed=args.seed + 10_000,
    )

    output = {
        "training": {
            "seed": args.seed,
            "total_timesteps": args.total_timesteps,
            "learning_rate": args.lr,
            "gamma": args.gamma,
            "batch_size": model.batch_size,
            "n_epochs": model.n_epochs,
            "checkpoint_freq": args.checkpoint_freq,
            "rating_freq": args.rating_freq,
            "Elo_K": args.K,
            "window_size": args.window_size,
            "temperature": args.temperature,
            "final_learner_elo": manager.learner_elo,
            "model_path": str(args.model_path),
        },
        "evaluation": results,
    }

    save_path = (
        args.results_dir
        / "final_evaluation.json"
    )

    with save_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=4,
        )

    print(
        json.dumps(
            output,
            indent=4,
        )
    )


if __name__ == "__main__":
    main()

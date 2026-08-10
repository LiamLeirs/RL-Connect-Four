import argparse
import json
from pathlib import Path

from sb3_contrib import MaskablePPO

from src.agents.agents import ModelAgent, RandomAgent
from src.envs.connect_four_env import ConnectFourEnv
from src.evaluation.evaluator import evaluate_agent
from src.self_play.self_play_manager import SelfPlayManager, SelfPlayCallback


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)

    # This is the number of environment transitions, not episodes.
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=300_000,
    )

    parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=5_000,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/ppo_selfplay/final_model"),
    )

    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("results/ppo_vs_random.json"),
    )

    parser.add_argument(
        "--tensorboard-log",
        type=Path,
        default=Path("runs/ppo_random"),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    args.model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.tensorboard_log.mkdir(
        parents=True,
        exist_ok=True,
    )

    manager = SelfPlayManager()

    train_env = ConnectFourEnv(
        opponent_provider=manager,
        render_mode=None,
    )

    model = MaskablePPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
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

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            progress_bar=True,
            tb_log_name="maskable_ppo_random",
            callback=SelfPlayCallback(manager=manager, checkpoint_freq=50_000, checkpoint_dir="models/ppo_selfplay/checkpoints"),
        )

        model.save(args.model_path)

    finally:
        train_env.close()

    eval_env = ConnectFourEnv(
        opponent=RandomAgent(),
        render_mode=None,
    )

    trained_agent = ModelAgent(
        model=model,
        deterministic=True,
    )

    results = evaluate_agent(
        env=eval_env,
        agent=trained_agent,
        opponent=RandomAgent(),
        num_episodes=args.num_eval_episodes,
        seed=args.seed + 10_000,
        )

    eval_env.close()

    results["training"] = {
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "gamma": args.gamma,
        "model_path": str(args.model_path),
    }

    with args.save_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=4)

    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sb3_contrib import MaskablePPO

from src.agents.agents import (
    Agent,
    ModelAgent,
    RandomAgent,
    TacticalAgent,
    MiniMaxAgent,
)
from src.envs.connect_four_env import ConnectFourEnv
from src.evaluation.evaluator import evaluate_agent


SUPPORTED_AGENT_TYPES = (
    "random",
    "tactical",
    "minimax",
    "ppo",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Connect Four matchup."
    )

    parser.add_argument(
        "--agent",
        choices=SUPPORTED_AGENT_TYPES,
        default="ppo",
        help="Agent being evaluated.",
    )

    parser.add_argument(
        "--opponent",
        choices=SUPPORTED_AGENT_TYPES,
        default="minimax",
        help="Agent used as the opponent.",
    )

    parser.add_argument(
        "--agent-model-path",
        type=Path,
        default=Path("models/ppo_selfplay/final_model"),
        help="Model path when --agent is ppo.",
    )

    parser.add_argument(
        "--opponent-model-path",
        type=Path,
        default=None,
        help="Model path when --opponent is ppo.",
    )

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )

    return parser.parse_args()


def resolve_model_path(model_path: Path | None) -> Path:
    if model_path is None:
        raise ValueError(
            "A model path is required for PPO agents."
        )

    if model_path.exists():
        return model_path

    zip_path = model_path.with_suffix(".zip")

    if zip_path.exists():
        return zip_path

    raise FileNotFoundError(
        f"Model not found at {model_path} or {zip_path}."
    )


def create_agent(
    agent_type: str,
    model_path: Path | None = None,
) -> Agent:
    if agent_type == "random":
        return RandomAgent()

    if agent_type == "tactical":
        return TacticalAgent()

    if agent_type == "minimax":
        return MiniMaxAgent(depth=2)

    if agent_type == "ppo":
        resolved_path = resolve_model_path(model_path)

        model = MaskablePPO.load(
            str(resolved_path)
        )

        return ModelAgent(
            model=model,
            deterministic=True,
        )

    raise ValueError(
        f"Unsupported agent type: {agent_type!r}"
    )


def default_save_path(
    agent_type: str,
    opponent_type: str,
) -> Path:
    return (
        Path("results")
        / f"{agent_type}_vs_{opponent_type}.json"
    )


def save_results(
    results: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )


def main() -> None:
    args = parse_args()

    if args.num_episodes <= 0:
        raise ValueError(
            "--num-episodes must be greater than zero."
        )

    evaluated_agent = create_agent(
        agent_type=args.agent,
        model_path=args.agent_model_path,
    )

    opponent = create_agent(
        agent_type=args.opponent,
        model_path=args.opponent_model_path,
    )

    env = ConnectFourEnv(
        opponent=opponent,
        render_mode=None,
    )

    try:
        results = evaluate_agent(
            env=env,
            agent=evaluated_agent,
            num_episodes=args.num_episodes,
            seed=args.seed,
        )
    finally:
        env.close()

    results["evaluation"] = {
        "agent_type": args.agent,
        "opponent_type": args.opponent,
        "agent_model_path": (
            str(args.agent_model_path)
            if args.agent == "ppo"
            else None
        ),
        "opponent_model_path": (
            str(args.opponent_model_path)
            if args.opponent == "ppo"
            else None
        ),
    }

    save_path = (
        args.save_path
        if args.save_path is not None
        else default_save_path(
            agent_type=args.agent,
            opponent_type=args.opponent,
        )
    )

    save_results(
        results=results,
        path=save_path,
    )

    print(json.dumps(results, indent=4))
    print(f"\nSaved results to: {save_path}")


if __name__ == "__main__":
    main()

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_PATH = Path("results/ppo_selfplay/training_history.csv")
OUTPUT_DIR = Path("results/plots")


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["games"] = (
        df["wins"]
        + df["draws"]
        + df["losses"]
    )

    # Elo-style match score:
    # win = 1, draw = 0.5, loss = 0
    df["score_rate"] = (
        df["wins"] + 0.5 * df["draws"]
    ) / df["games"]

    df["win_rate"] = (
        df["wins"] / df["games"]
    )

    df["draw_rate"] = (
        df["draws"] / df["games"]
    )

    df["loss_rate"] = (
        df["losses"] / df["games"]
    )

    return df


def plot_learner_elo(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    # learner_elo occurs multiple times per timestep because
    # there is one row per evaluated opponent.
    elo_df = (
        df[["timestep", "learner_elo"]]
        .drop_duplicates(subset="timestep")
        .sort_values("timestep")
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        elo_df["timestep"],
        elo_df["learner_elo"],
        linewidth=2,
    )

    ax.set_title("Learner Elo During Self-Play Training")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Elo rating")
    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir / "learner_elo.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_baseline_performance(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    baselines = [
        "Random",
        "Tactical",
        "MiniMax2",
        "MiniMax4",
    ]

    baseline_df = df[
        df["opponent"].isin(baselines)
    ]

    fig, ax = plt.subplots(figsize=(11, 6))

    for opponent in baselines:
        opponent_df = (
            baseline_df[
                baseline_df["opponent"] == opponent
            ]
            .sort_values("timestep")
        )

        if opponent_df.empty:
            continue

        ax.plot(
            opponent_df["timestep"],
            opponent_df["score_rate"] * 100,
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=opponent,
        )

    ax.axhline(
        50,
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )

    ax.set_title(
        "PPO Performance Against Fixed Opponents"
    )
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Match score (%)")
    ax.set_ylim(0, 100)

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir / "baseline_performance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_minimax_performance(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    minimax_agents = [
        "MiniMax2",
        "MiniMax4",
    ]

    fig, ax = plt.subplots(figsize=(11, 6))

    for opponent in minimax_agents:
        opponent_df = (
            df[df["opponent"] == opponent]
            .sort_values("timestep")
        )

        if opponent_df.empty:
            continue

        ax.plot(
            opponent_df["timestep"],
            opponent_df["score_rate"] * 100,
            marker="o",
            markersize=3,
            linewidth=2,
            label=opponent,
        )

    ax.axhline(
        50,
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label="Equal performance",
    )

    ax.set_title(
        "PPO Performance Against Minimax"
    )
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Match score (%)")
    ax.set_ylim(0, 100)

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir / "minimax_performance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_historical_ppo_performance(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    ppo_df = df[
        df["opponent"].str.startswith(
            "PPO_",
            na=False,
        )
    ].copy()

    if ppo_df.empty:
        return

    ppo_df = ppo_df.sort_values("timestep")

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.scatter(
        ppo_df["timestep"],
        ppo_df["score_rate"] * 100,
        alpha=0.7,
    )

    ax.axhline(
        50,
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )

    ax.set_title(
        "Performance Against Historical PPO Checkpoints"
    )
    ax.set_xlabel("Current learner timestep")
    ax.set_ylabel("Match score (%)")
    ax.set_ylim(0, 100)

    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir / "historical_ppo_performance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_results(
        RESULTS_PATH
    )

    plot_learner_elo(
        df,
        OUTPUT_DIR,
    )

    plot_baseline_performance(
        df,
        OUTPUT_DIR,
    )

    plot_minimax_performance(
        df,
        OUTPUT_DIR,
    )

    plot_historical_ppo_performance(
        df,
        OUTPUT_DIR,
    )

    print(
        f"Plots saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()

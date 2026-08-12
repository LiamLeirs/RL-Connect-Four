from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from loguru import logger
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from src.agents.agents import *
from src.evaluation.evaluator import evaluate_agent
from src.envs.connect_four_env import ConnectFourEnv


@dataclass
class PlayerEntry:
    name: str
    agent_factory: Callable[[], Agent]
    timestep: int | None = None
    kind: str = "checkpoint"
    elo: float = 1200
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    def create_agent(self) -> Agent:
        return self.agent_factory()


class SelfPlayCallback(BaseCallback):
    def __init__(
        self,
        manager,
        checkpoint_freq,
        rating_freq,
        checkpoint_dir,
        verbose=0,
    ):
        super().__init__(verbose=verbose)

        self.manager: SelfPlayManager = manager
        self.checkpoint_freq = checkpoint_freq
        self.rating_freq = rating_freq
        self.checkpoint_dir = Path(checkpoint_dir)
        self.next_checkpoint = checkpoint_freq
        self.next_rating = rating_freq

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save_checkpoint(self):
        timestep = self.num_timesteps

        logger.info(
            f"Saving checkpoint | "
            f"timestep={timestep} | "
            f"elo={self.manager.learner_elo:.1f}"
        )

        save_path = (
            self.checkpoint_dir
            / f"checkpoint_{timestep}.zip"
        )
        self.model.save(save_path)
        frozen_model = MaskablePPO.load(save_path)
        self.manager.add_checkpoint(
            name=f"PPO_{timestep}",
            model=frozen_model,
            timestep=timestep
        )

        logger.info(
            f"Checkpoint saved | "
            f"name=PPO_{timestep} | "
            f"league_size={len(self.manager.league)}"
        )

    def _on_step(self) -> bool:
        timestep = self.num_timesteps

        rating_due = timestep >= self.next_rating
        checkpoint_due = timestep >= self.next_checkpoint

        # A checkpoint always gets a fresh rating first.
        if rating_due or checkpoint_due:
            self.run_rating_period()

        if rating_due:
            while self.next_rating <= timestep:
                self.next_rating += self.rating_freq

        if checkpoint_due:
            self.save_checkpoint()

            while self.next_checkpoint <= timestep:
                self.next_checkpoint += self.checkpoint_freq

        return True

    def run_rating_period(self):
        timestep = self.num_timesteps

        # Freeze the learner Elo for this entire rating period.
        learner_elo_before = self.manager.learner_elo

        logger.info(
            f"Rating period | "
            f"timestep={timestep} | "
            f"learner_elo={learner_elo_before:.1f}"
        )

        learner_agent = ModelAgent(
            model=self.model,
            deterministic=False,
        )

        opponent_entries = (
            self.manager.sample_evaluation_league(
                num_opponents=4,
            )
        )

        num_eval_games = 100
        rating_results = []

        for i, opponent_entry in enumerate(opponent_entries):
            evaluation_opponent = opponent_entry.create_agent()

            env = ConnectFourEnv(
                opponent=evaluation_opponent,
                render_mode=None,
            )

            try:
                elo_scores = evaluate_agent(
                    env=env,
                    agent=learner_agent,
                    num_episodes=num_eval_games,
                    get_elo_scores=True,
                    seed=timestep + i * 10_000,
                )
            finally:
                env.close()

            # Match statistics from the learner's perspective.
            wins = sum(
                score == 1.0
                for score in elo_scores
            )
            draws = sum(
                score == 0.5
                for score in elo_scores
            )
            losses = sum(
                score == 0.0
                for score in elo_scores
            )

            self.manager.record_results(
                opponent_entry,
                elo_scores,
            )

            # Calculate but DO NOT apply the Elo change yet.
            elo_delta = self.manager.calculate_elo_delta(
                learner_elo=learner_elo_before,
                opponent_elo=opponent_entry.elo,
                results=elo_scores,
            )

            rating_results.append(elo_delta)

            logger.info(
                f"vs {opponent_entry.name:<12} | "
                f"W/D/L={wins}/{draws}/{losses} | "
                f"opponent_elo={opponent_entry.elo:.1f} | "
                f"proposed_delta={elo_delta:+.1f}"
            )

        # Apply all Elo changes only after every matchup has
        # been evaluated, removing opponent-order dependence.
        total_delta = sum(rating_results)

        self.manager.learner_elo = (
            learner_elo_before + total_delta
        )

        for opponent_entry, elo_delta in zip(
            opponent_entries,
            rating_results,
        ):
            opponent_entry.elo -= elo_delta

        logger.info(
            f"Rating period complete | "
            f"learner_elo={learner_elo_before:.1f}"
            f"->{self.manager.learner_elo:.1f} | "
            f"delta={total_delta:+.1f}"
        )

        logger.info(
            f"Rating period complete | "
            f"timestep={timestep} | "
            f"elo={self.manager.learner_elo:.1f} | "
            f"league_size={len(self.manager.league)}"
        )


class SelfPlayManager:
    def __init__(self, window_size=8, temperature=200, K=8):
        self.league = [
            PlayerEntry(
                name="Random",
                agent_factory=RandomAgent,
                kind="baseline",
            ),
            PlayerEntry(
                name="Tactical",
                agent_factory=TacticalAgent,
                kind="baseline",
            ),
            PlayerEntry(
                name="MiniMax2",
                agent_factory=lambda: MiniMaxAgent(depth=2),
                kind="baseline",
            ),
            PlayerEntry(
                name="MiniMax4",
                agent_factory=lambda: MiniMaxAgent(depth=4),
                kind="baseline",
            ),
        ]

        self.window_size = window_size
        self.learner_elo = 1200
        self.temperature = temperature
        self.K = K

    def add_checkpoint(self, name, model, timestep):
        self.league.append(
            PlayerEntry(
                name=name,
                agent_factory=lambda model=model: ModelAgent(
                    model=model,
                    deterministic=False,
                ),
                timestep=timestep,
                kind="checkpoint",
                elo=self.learner_elo
            )
        )

    def expected_score(self, learner_elo, opponent_elo):
        return 1 / (
            1 + 10 ** ((opponent_elo - learner_elo) / 400)
        )

    def calculate_elo_delta(self, learner_elo, opponent_elo, results):
        initial_learner_elo = learner_elo

        temp_learner_elo = learner_elo
        temp_opponent_elo = opponent_elo

        for elo_score in results:
            expected_learner = self.expected_score(
                temp_learner_elo,
                temp_opponent_elo,
            )

            delta = self.K * (
                elo_score - expected_learner
            )

            temp_learner_elo += delta
            temp_opponent_elo -= delta

        return temp_learner_elo - initial_learner_elo

    def record_results(self, opponent_entry, results):
        for score in results:
            if score == 1.0:
                opponent_entry.losses += 1
            elif score == 0.0:
                opponent_entry.wins += 1
            elif score == 0.5:
                opponent_entry.draws += 1
            else:
                raise ValueError(
                    f"Invalid Elo score: {score}"
                )

            opponent_entry.games_played += 1

    def sample_evaluation_league(self, num_opponents=4):
        num_opponents = min(
            num_opponents,
            len(self.league),
        )

        sorted_league = sorted(
            self.league,
            key=lambda entry: entry.elo,
        )

        league_lte = []
        league_gt = []

        for entry in sorted_league:
            if entry.elo <= self.learner_elo:
                league_lte.append(entry)
            else:
                league_gt.append(entry)

        half = num_opponents // 2

        below = list(reversed(league_lte))
        above = league_gt

        eval_entries = below[:half] + above[:half]

        remaining = num_opponents - len(eval_entries)

        if remaining > 0:
            unused = below[half:] + above[half:]

            unused.sort(
                key=lambda entry: abs(
                    entry.elo - self.learner_elo
                )
            )

            eval_entries += unused[:remaining]

        return eval_entries

    def sample_opponent(self, rng=np.random):
        baselines = [
            entry
            for entry in self.league
            if entry.kind == "baseline"
        ]

        checkpoints = [
            entry
            for entry in self.league
            if entry.kind == "checkpoint"
        ]

        checkpoints = sorted(
            checkpoints,
            key=lambda entry: entry.timestep,
        )

        eligible_entries = (
            checkpoints[-self.window_size:]
            + baselines
        )

        distances = np.array(
            [
                abs(self.learner_elo - entry.elo)
                for entry in eligible_entries
            ],
            dtype=np.float64,
        )

        weights = np.exp(
            -distances / self.temperature
        )

        probabilities = (
            weights / weights.sum()
        )

        entry = rng.choice(
            eligible_entries,
            p=probabilities,
        )

        # Every episode gets its own Agent wrapper/instance.
        return entry.create_agent()

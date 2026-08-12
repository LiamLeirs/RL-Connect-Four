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
    def __init__(self, manager, checkpoint_freq, checkpoint_dir, verbose=0):
        super().__init__(verbose=verbose)

        self.manager: SelfPlayManager = manager
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_dir = Path(checkpoint_dir)
        self.next_checkpoint = checkpoint_freq

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_checkpoint:
            return True

        timestep = self.num_timesteps
        save_path = self.checkpoint_dir / f"checkpoint_{timestep}.zip"

        logger.info(
            f"Checkpoint evaluation | timestep={timestep} | "
            f"learner_elo={self.manager.learner_elo:.1f}"
        )

        # Freeze the current learner.
        self.model.save(save_path)
        frozen_model = MaskablePPO.load(save_path)

        learner_agent = ModelAgent(
            model=frozen_model,
            deterministic=True,
        )

        opponent_entries = self.manager.sample_evaluation_league(
            num_opponents=4
        )

        num_eval_games = 100

        for opponent_entry in opponent_entries:
            # IMPORTANT:
            # Evaluation gets a fresh opponent instance.
            evaluation_opponent = opponent_entry.create_agent()

            env = ConnectFourEnv(
                opponent=evaluation_opponent,
                render_mode=None,
            )

            learner_elo_before = self.manager.learner_elo
            opponent_elo_before = opponent_entry.elo

            try:
                elo_scores = evaluate_agent(
                    env=env,
                    agent=learner_agent,
                    num_episodes=num_eval_games,
                    get_elo_scores=True,
                )
            finally:
                env.close()

            wins = sum(score == 1.0 for score in elo_scores)
            draws = sum(score == 0.5 for score in elo_scores)
            losses = sum(score == 0.0 for score in elo_scores)

            self.manager.update_elo(
                opponent_entry,
                elo_scores,
            )

            logger.info(
                f"vs {opponent_entry.name:<12} | "
                f"W/D/L={wins}/{draws}/{losses} | "
                f"opponent_elo={opponent_elo_before:.1f}"
                f"->{opponent_entry.elo:.1f} | "
                f"learner_elo={learner_elo_before:.1f}"
                f"->{self.manager.learner_elo:.1f}"
            )

        self.manager.add_checkpoint(
            name=f"PPO_{timestep}",
            model=frozen_model,
            timestep=timestep,
        )

        logger.info(
            f"Checkpoint complete | timestep={timestep} | "
            f"elo={self.manager.learner_elo:.1f} | "
            f"league_size={len(self.manager.league)}"
        )

        self.next_checkpoint += self.checkpoint_freq

        return True


class SelfPlayManager:
    def __init__(self, window_size=8, temperature=200):
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

    def add_checkpoint(self, name, model, timestep):
        self.league.append(
            PlayerEntry(
                name=name,
                agent_factory=lambda model=model: ModelAgent(
                    model=model,
                    deterministic=True,
                ),
                timestep=timestep,
                kind="checkpoint",
                elo=self.learner_elo,
            )
        )

    def expected_score(self, learner_elo, opponent_elo):
        return 1 / (
            1 + 10 ** ((opponent_elo - learner_elo) / 400)
        )

    def update_elo(self, opponent_entry, results, K=32):
        learner_elo = self.learner_elo

        for elo_score in results:
            expected_learner = self.expected_score(
                learner_elo,
                opponent_entry.elo,
            )

            expected_opponent = 1 - expected_learner

            new_learner_elo = learner_elo + K * (
                elo_score - expected_learner
            )

            new_opponent_elo = opponent_entry.elo + K * (
                (1 - elo_score) - expected_opponent
            )

            learner_elo = new_learner_elo
            opponent_entry.elo = new_opponent_elo

            if elo_score == 1.0:
                opponent_entry.losses += 1
            elif elo_score == 0.0:
                opponent_entry.wins += 1
            elif elo_score == 0.5:
                opponent_entry.draws += 1
            else:
                raise ValueError(
                    f"Invalid Elo score: {elo_score}"
                )

            opponent_entry.games_played += 1

        self.learner_elo = learner_elo

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
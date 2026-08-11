from dataclasses import dataclass
from src.agents.agents import *
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from pathlib import Path
from loguru import logger

@dataclass
class PlayerEntry:
    name: str
    agent: Agent
    timestep: int
    kind: str = "checkpoint"
    elo: float = 1000
    games_played: int = 0

class SelfPlayCallback(BaseCallback):
    def __init__(self, manager, checkpoint_freq, checkpoint_dir, verbose=0):
        super().__init__(verbose=verbose)
        self.manager = manager
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_dir = checkpoint_dir
        self.next_checkpoint = checkpoint_freq
        self.timestep = 0

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_checkpoint:
            timestep = self.num_timesteps

            save_path = (
                Path(self.checkpoint_dir)
                / f"checkpoint_{timestep}.zip"
            )

            logger.info(f"Saving checkpoint to {save_path}")
            logger.info(f"Time elapsed: {timestep}")

            self.model.save(save_path)

            self.manager.add_checkpoint(
                name=f"checkpoint_{timestep}",
                model_path=save_path,
                timestep=timestep,
            )

            self.next_checkpoint += self.checkpoint_freq

        return True
    
class SelfPlayManager:
    def __init__(self, window_size=8):
        self.league = list[PlayerEntry]()
        self.league.append(PlayerEntry(name="Random", agent=RandomAgent(), timestep=0, kind="baseline"))
        self.league.append(PlayerEntry(name="Tactical", agent=TacticalAgent(), timestep=0, kind="baseline"))
        self.window_size = window_size
        self.learner_elo = 1000

    def add_checkpoint(self, name, model_path, timestep):
        agent = ModelAgent(model=MaskablePPO.load(model_path), deterministic=True)
        self.league.append(PlayerEntry(name=name, agent=agent, timestep=timestep, kind="checkpoint", elo=self.learner_elo))

    def sample_opponent(self, rng=np.random):
        if len(self.pool) == 0:
            entry = rng.choice(self.anchors)
        else:
            prob = rng.uniform(0, 1)
            if prob < self.anchor_weight:
                entry = rng.choice(self.anchors)
            else:
                entry = rng.choice(self.pool[-self.window_size:])
                logger.info(f"Selected opponent {entry.name}")

        return entry.agent

    def update_weights(self, timestep):
        pass
            
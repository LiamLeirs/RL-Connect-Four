from dataclasses import dataclass
from src.agents.agents import *
import numpy as np

@dataclass
class PlayerEntry:
    name: str
    player: Agent
    timestep: int
    kind: str = "checkpoint"

class SelfPlayManager:
    def __init__(self, num_timesteps=100_000, window_size=8, anchor_weight=0.5):
        self.pool: list[PlayerEntry] = []
        self.anchors: list[PlayerEntry] = []
        self.anchors.append(PlayerEntry(name="Random", player=RandomAgent(), timestep=0, kind="baseline"))
        self.anchors.append(PlayerEntry(name="Tactical", player=TacticalAgent(), timestep=0, kind="baseline"))
        self.current_player = 0
        self.timestep = 0
        self.anchor_weight = anchor_weight
        self.window_size = window_size
        self.num_timesteps = num_timesteps

    def add_player(self, name, player, timestep):
        self.pool.append(PlayerEntry(name=name, player=player, timestep=timestep, kind="checkpoint"))

    def sample_opponent(self, rng=np.random):
        if len(self.pool) == 0:
            return rng.choice(self.anchors)
        prob = rng.uniform(0, 1)
        if prob < self.anchor_weight:
            return rng.choice(self.anchors)
        else:
            return rng.choice(self.pool[-self.window_size:])

    def update_weights(self, timestep):
        pass
            
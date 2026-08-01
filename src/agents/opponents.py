import numpy as np

class Opponent:
    def select_action(self, observation, action_mask, rng):
        raise NotImplementedError

class RandomOpponent(Opponent):
    def select_action(self, observation, action_mask, rng):
        legal = np.flatnonzero(action_mask)
        return int(rng.choice(legal))
import numpy as np
import pytest

from src.agents.agents import Agent, RandomAgent
from src.self_play.self_play_manager import (
    PlayerEntry,
    SelfPlayManager,
)


class DummyAgent(Agent):
    def select_action(self, observation, action_mask, rng):
        legal_actions = np.flatnonzero(action_mask)
        return int(legal_actions[0])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager():
    return SelfPlayManager(
        window_size=3,
        temperature=200,
    )


# ---------------------------------------------------------------------------
# Expected score
# ---------------------------------------------------------------------------


def test_expected_score_equal_elo(manager):
    score = manager.expected_score(
        learner_elo=1200,
        opponent_elo=1200,
    )

    assert score == pytest.approx(0.5)


def test_expected_score_higher_rated_learner(manager):
    score = manager.expected_score(
        learner_elo=1400,
        opponent_elo=1200,
    )

    assert score > 0.5


def test_expected_score_lower_rated_learner(manager):
    score = manager.expected_score(
        learner_elo=1000,
        opponent_elo=1200,
    )

    assert score < 0.5


def test_expected_scores_are_complementary(manager):
    learner_score = manager.expected_score(
        learner_elo=1300,
        opponent_elo=1100,
    )

    opponent_score = manager.expected_score(
        learner_elo=1100,
        opponent_elo=1300,
    )

    assert learner_score + opponent_score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Elo updates
# ---------------------------------------------------------------------------


def test_equal_elo_learner_win_updates_to_1216_and_1184(manager):
    manager.learner_elo = 1200

    opponent = PlayerEntry(
        name="Opponent",
        agent=DummyAgent(),
        elo=1200,
    )

    manager.update_elo(
        opponent,
        results=[1.0],
        K=32,
    )

    assert manager.learner_elo == pytest.approx(1216)
    assert opponent.elo == pytest.approx(1184)


def test_equal_elo_learner_loss_updates_to_1184_and_1216(manager):
    manager.learner_elo = 1200

    opponent = PlayerEntry(
        name="Opponent",
        agent=DummyAgent(),
        elo=1200,
    )

    manager.update_elo(
        opponent,
        results=[0.0],
        K=32,
    )

    assert manager.learner_elo == pytest.approx(1184)
    assert opponent.elo == pytest.approx(1216)


def test_equal_elo_draw_does_not_change_ratings(manager):
    manager.learner_elo = 1200

    opponent = PlayerEntry(
        name="Opponent",
        agent=DummyAgent(),
        elo=1200,
    )

    manager.update_elo(
        opponent,
        results=[0.5],
        K=32,
    )

    assert manager.learner_elo == pytest.approx(1200)
    assert opponent.elo == pytest.approx(1200)


def test_multiple_games_update_sequentially(manager):
    manager.learner_elo = 1200

    opponent = PlayerEntry(
        name="Opponent",
        agent=DummyAgent(),
        elo=1200,
    )

    manager.update_elo(
        opponent,
        results=[1.0, 1.0],
        K=32,
    )

    # First game is exactly +16.
    # Second game should be less than +16 because learner is now favored.
    assert manager.learner_elo > 1216
    assert manager.learner_elo < 1232

    assert opponent.elo < 1184
    assert opponent.elo > 1168


def test_elo_is_zero_sum(manager):
    manager.learner_elo = 1200

    opponent = PlayerEntry(
        name="Opponent",
        agent=DummyAgent(),
        elo=1200,
    )

    initial_total = (
        manager.learner_elo
        + opponent.elo
    )

    manager.update_elo(
        opponent,
        results=[1.0, 0.0, 0.5, 1.0],
        K=32,
    )

    final_total = (
        manager.learner_elo
        + opponent.elo
    )

    assert final_total == pytest.approx(initial_total)


# ---------------------------------------------------------------------------
# Opponent statistics
# ---------------------------------------------------------------------------


def test_update_elo_tracks_opponent_stats(manager):
    opponent = PlayerEntry(
        name="Opponent",
        agent=DummyAgent(),
        elo=1200,
    )

    manager.update_elo(
        opponent,
        results=[
            1.0,   # opponent loses
            0.0,   # opponent wins
            0.5,   # draw
            1.0,   # opponent loses
        ],
    )

    assert opponent.games_played == 4
    assert opponent.wins == 1
    assert opponent.losses == 2
    assert opponent.draws == 1


# ---------------------------------------------------------------------------
# Checkpoint registration
# ---------------------------------------------------------------------------


def test_checkpoint_inherits_current_learner_elo(manager):
    manager.learner_elo = 1375.5

    agent = DummyAgent()

    manager.add_checkpoint(
        name="PPO_50000",
        agent=agent,
        timestep=50_000,
    )

    checkpoint = manager.league[-1]

    assert checkpoint.name == "PPO_50000"
    assert checkpoint.agent is agent
    assert checkpoint.timestep == 50_000
    assert checkpoint.kind == "checkpoint"
    assert checkpoint.elo == pytest.approx(1375.5)


def test_checkpoint_is_added_to_league(manager):
    league_size_before = len(manager.league)

    manager.add_checkpoint(
        name="PPO_50000",
        agent=DummyAgent(),
        timestep=50_000,
    )

    assert len(manager.league) == league_size_before + 1


# ---------------------------------------------------------------------------
# Evaluation league sampling
# ---------------------------------------------------------------------------


def test_sample_evaluation_league_returns_player_entries(manager):
    entries = manager.sample_evaluation_league(
        num_opponents=4
    )

    assert len(entries) == 4

    for entry in entries:
        assert isinstance(entry, PlayerEntry)


def test_sample_evaluation_league_does_not_exceed_league_size(manager):
    entries = manager.sample_evaluation_league(
        num_opponents=100
    )

    assert len(entries) == len(manager.league)


def test_sample_evaluation_league_prefers_nearby_ratings():
    manager = SelfPlayManager()

    manager.league = [
        PlayerEntry(
            name="VeryLow",
            agent=DummyAgent(),
            elo=700,
        ),
        PlayerEntry(
            name="Low",
            agent=DummyAgent(),
            elo=1100,
        ),
        PlayerEntry(
            name="NearLow",
            agent=DummyAgent(),
            elo=1190,
        ),
        PlayerEntry(
            name="NearHigh",
            agent=DummyAgent(),
            elo=1210,
        ),
        PlayerEntry(
            name="High",
            agent=DummyAgent(),
            elo=1300,
        ),
        PlayerEntry(
            name="VeryHigh",
            agent=DummyAgent(),
            elo=1700,
        ),
    ]

    manager.learner_elo = 1200

    entries = manager.sample_evaluation_league(
        num_opponents=4
    )

    names = {
        entry.name
        for entry in entries
    }

    assert "NearLow" in names
    assert "NearHigh" in names

    assert "VeryLow" not in names
    assert "VeryHigh" not in names


def test_sample_evaluation_league_fills_from_available_side():
    manager = SelfPlayManager()

    manager.league = [
        PlayerEntry(
            name="A",
            agent=DummyAgent(),
            elo=900,
        ),
        PlayerEntry(
            name="B",
            agent=DummyAgent(),
            elo=1000,
        ),
        PlayerEntry(
            name="C",
            agent=DummyAgent(),
            elo=1100,
        ),
        PlayerEntry(
            name="D",
            agent=DummyAgent(),
            elo=1300,
        ),
    ]

    manager.learner_elo = 1200

    entries = manager.sample_evaluation_league(
        num_opponents=4
    )

    assert len(entries) == 4

    assert {
        entry.name
        for entry in entries
    } == {
        "A",
        "B",
        "C",
        "D",
    }


# ---------------------------------------------------------------------------
# Training opponent sampling
# ---------------------------------------------------------------------------


def test_sample_opponent_returns_agent(manager):
    opponent = manager.sample_opponent(
        rng=np.random.default_rng(42)
    )

    assert isinstance(opponent, Agent)


def test_sample_opponent_only_uses_latest_checkpoint_window():
    manager = SelfPlayManager(
        window_size=2,
        temperature=200,
    )

    # Remove default league to make the test controlled.
    manager.league = []

    baseline = PlayerEntry(
        name="Baseline",
        agent=DummyAgent(),
        kind="baseline",
        elo=1200,
    )

    old_checkpoint = PlayerEntry(
        name="Old",
        agent=DummyAgent(),
        timestep=10_000,
        kind="checkpoint",
        elo=1200,
    )

    recent_checkpoint_1 = PlayerEntry(
        name="Recent1",
        agent=DummyAgent(),
        timestep=20_000,
        kind="checkpoint",
        elo=1200,
    )

    recent_checkpoint_2 = PlayerEntry(
        name="Recent2",
        agent=DummyAgent(),
        timestep=30_000,
        kind="checkpoint",
        elo=1200,
    )

    manager.league.extend(
        [
            baseline,
            old_checkpoint,
            recent_checkpoint_1,
            recent_checkpoint_2,
        ]
    )

    rng = np.random.default_rng(123)

    selected_agents = {
        manager.sample_opponent(rng)
        for _ in range(500)
    }

    assert baseline.agent in selected_agents
    assert recent_checkpoint_1.agent in selected_agents
    assert recent_checkpoint_2.agent in selected_agents

    assert old_checkpoint.agent not in selected_agents


def test_closer_elo_is_sampled_more_often():
    manager = SelfPlayManager(
        window_size=8,
        temperature=100,
    )

    manager.league = []

    close_agent = DummyAgent()
    far_agent = DummyAgent()

    manager.league.extend(
        [
            PlayerEntry(
                name="Close",
                agent=close_agent,
                kind="baseline",
                elo=1210,
            ),
            PlayerEntry(
                name="Far",
                agent=far_agent,
                kind="baseline",
                elo=1700,
            ),
        ]
    )

    manager.learner_elo = 1200

    rng = np.random.default_rng(42)

    close_count = 0
    far_count = 0

    for _ in range(5_000):
        opponent = manager.sample_opponent(rng)

        if opponent is close_agent:
            close_count += 1

        elif opponent is far_agent:
            far_count += 1

    assert close_count > far_count


def test_same_seed_produces_same_sampling_sequence():
    manager_one = SelfPlayManager(
        temperature=200
    )

    manager_two = SelfPlayManager(
        temperature=200
    )

    rng_one = np.random.default_rng(123)
    rng_two = np.random.default_rng(123)

    sequence_one = [
        type(manager_one.sample_opponent(rng_one))
        for _ in range(50)
    ]

    sequence_two = [
        type(manager_two.sample_opponent(rng_two))
        for _ in range(50)
    ]

    assert sequence_one == sequence_two
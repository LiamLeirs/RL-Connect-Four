import numpy as np
import pytest

from src.agents.agents import Agent, ModelAgent
from src.self_play.self_play_manager import (
    PlayerEntry,
    SelfPlayManager,
)


class DummyAgent(Agent):
    def __init__(self, tag="dummy"):
        self.tag = tag

    def select_action(self, observation, action_mask, rng):
        legal_actions = np.flatnonzero(action_mask)
        return int(legal_actions[0])


@pytest.fixture
def manager():
    return SelfPlayManager(
        window_size=3,
        temperature=200,
        K=32,
    )


def make_entry(
    name,
    elo=1200,
    kind="checkpoint",
    timestep=None,
):
    return PlayerEntry(
        name=name,
        agent_factory=lambda name=name: DummyAgent(tag=name),
        timestep=timestep,
        kind=kind,
        elo=elo,
    )


# ---------------------------------------------------------------------------
# PlayerEntry / factory
# ---------------------------------------------------------------------------


def test_player_entry_creates_agent():
    entry = make_entry("Test")

    agent = entry.create_agent()

    assert isinstance(agent, DummyAgent)
    assert agent.tag == "Test"


def test_player_entry_factory_creates_fresh_instances():
    entry = make_entry("Test")

    agent_one = entry.create_agent()
    agent_two = entry.create_agent()

    assert agent_one is not agent_two
    assert agent_one.tag == agent_two.tag == "Test"


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
# Elo delta calculation
# ---------------------------------------------------------------------------


def test_equal_elo_learner_win_produces_positive_16_delta(manager):
    delta = manager.calculate_elo_delta(
        learner_elo=1200,
        opponent_elo=1200,
        results=[1.0],
    )

    assert delta == pytest.approx(16.0)


def test_equal_elo_learner_loss_produces_negative_16_delta(manager):
    delta = manager.calculate_elo_delta(
        learner_elo=1200,
        opponent_elo=1200,
        results=[0.0],
    )

    assert delta == pytest.approx(-16.0)


def test_equal_elo_draw_produces_zero_delta(manager):
    delta = manager.calculate_elo_delta(
        learner_elo=1200,
        opponent_elo=1200,
        results=[0.5],
    )

    assert delta == pytest.approx(0.0)


def test_calculate_elo_delta_does_not_mutate_manager(manager):
    manager.learner_elo = 1200

    manager.calculate_elo_delta(
        learner_elo=1200,
        opponent_elo=1200,
        results=[1.0, 1.0],
    )

    assert manager.learner_elo == pytest.approx(1200)


def test_multiple_games_are_processed_sequentially(manager):
    delta = manager.calculate_elo_delta(
        learner_elo=1200,
        opponent_elo=1200,
        results=[1.0, 1.0],
    )

    # First win gives +16.
    # The learner is then favored in game two,
    # so the second increase is less than +16.
    assert delta > 16
    assert delta < 32


def test_win_then_loss_has_smaller_total_than_single_win(manager):
    single_win_delta = manager.calculate_elo_delta(
        learner_elo=1200,
        opponent_elo=1200,
        results=[1.0],
    )

    win_loss_delta = manager.calculate_elo_delta(
        learner_elo=1200,
        opponent_elo=1200,
        results=[1.0, 0.0],
    )

    assert win_loss_delta < single_win_delta


def test_calculate_elo_delta_uses_manager_k_factor():
    manager = SelfPlayManager(K=8)

    delta = manager.calculate_elo_delta(
        learner_elo=1200,
        opponent_elo=1200,
        results=[1.0],
    )

    assert delta == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Opponent statistics
# ---------------------------------------------------------------------------


def test_record_results_tracks_opponent_stats(manager):
    opponent = make_entry(
        name="Opponent",
        elo=1200,
    )

    manager.record_results(
        opponent,
        results=[
            1.0,  # learner wins -> opponent loss
            0.0,  # learner loses -> opponent win
            0.5,  # draw
            1.0,  # learner wins -> opponent loss
        ],
    )

    assert opponent.games_played == 4
    assert opponent.wins == 1
    assert opponent.losses == 2
    assert opponent.draws == 1


def test_record_results_accumulates_statistics(manager):
    opponent = make_entry("Opponent")

    manager.record_results(
        opponent,
        results=[1.0, 0.5],
    )

    manager.record_results(
        opponent,
        results=[0.0],
    )

    assert opponent.games_played == 3
    assert opponent.wins == 1
    assert opponent.losses == 1
    assert opponent.draws == 1


def test_record_results_rejects_invalid_score(manager):
    opponent = make_entry("Opponent")

    with pytest.raises(ValueError):
        manager.record_results(
            opponent,
            results=[0.25],
        )


# ---------------------------------------------------------------------------
# Checkpoint registration
# ---------------------------------------------------------------------------


def test_checkpoint_inherits_current_learner_elo(manager):
    manager.learner_elo = 1375.5

    dummy_model = object()

    manager.add_checkpoint(
        name="PPO_50000",
        model=dummy_model,
        timestep=50_000,
    )

    checkpoint = manager.league[-1]

    assert checkpoint.name == "PPO_50000"
    assert checkpoint.timestep == 50_000
    assert checkpoint.kind == "checkpoint"
    assert checkpoint.elo == pytest.approx(1375.5)


def test_checkpoint_is_added_to_league(manager):
    league_size_before = len(manager.league)

    manager.add_checkpoint(
        name="PPO_50000",
        model=object(),
        timestep=50_000,
    )

    assert len(manager.league) == league_size_before + 1


def test_checkpoint_factory_creates_model_agent(manager):
    dummy_model = object()

    manager.add_checkpoint(
        name="PPO_50000",
        model=dummy_model,
        timestep=50_000,
    )

    checkpoint = manager.league[-1]

    agent = checkpoint.create_agent()

    assert isinstance(agent, ModelAgent)


def test_checkpoint_factory_creates_fresh_wrappers(manager):
    dummy_model = object()

    manager.add_checkpoint(
        name="PPO_50000",
        model=dummy_model,
        timestep=50_000,
    )

    checkpoint = manager.league[-1]

    agent_one = checkpoint.create_agent()
    agent_two = checkpoint.create_agent()

    assert agent_one is not agent_two


# ---------------------------------------------------------------------------
# Evaluation league sampling
# ---------------------------------------------------------------------------


def test_sample_evaluation_league_returns_player_entries(manager):
    entries = manager.sample_evaluation_league(
        num_opponents=4,
    )

    assert len(entries) == 4

    for entry in entries:
        assert isinstance(entry, PlayerEntry)


def test_sample_evaluation_league_does_not_exceed_league_size(manager):
    entries = manager.sample_evaluation_league(
        num_opponents=100,
    )

    assert len(entries) == len(manager.league)


def test_sample_evaluation_league_prefers_nearby_ratings():
    manager = SelfPlayManager()

    manager.league = [
        make_entry("VeryLow", elo=700),
        make_entry("Low", elo=1100),
        make_entry("NearLow", elo=1190),
        make_entry("NearHigh", elo=1210),
        make_entry("High", elo=1300),
        make_entry("VeryHigh", elo=1700),
    ]

    manager.learner_elo = 1200

    entries = manager.sample_evaluation_league(
        num_opponents=4,
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
        make_entry("A", elo=900),
        make_entry("B", elo=1000),
        make_entry("C", elo=1100),
        make_entry("D", elo=1300),
    ]

    manager.learner_elo = 1200

    entries = manager.sample_evaluation_league(
        num_opponents=4,
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


def test_sample_evaluation_league_handles_single_entry():
    manager = SelfPlayManager()

    manager.league = [
        make_entry("Only", elo=1200),
    ]

    manager.learner_elo = 1200

    entries = manager.sample_evaluation_league(
        num_opponents=4,
    )

    assert len(entries) == 1
    assert entries[0].name == "Only"


# ---------------------------------------------------------------------------
# Training opponent sampling
# ---------------------------------------------------------------------------


def test_sample_opponent_returns_agent(manager):
    opponent = manager.sample_opponent(
        rng=np.random.default_rng(42),
    )

    assert isinstance(opponent, Agent)


def test_sample_opponent_returns_fresh_instance():
    manager = SelfPlayManager()
    manager.league = [
        make_entry(
            name="Only",
            kind="baseline",
            elo=1200,
        )
    ]

    rng = np.random.default_rng(42)

    first = manager.sample_opponent(rng)
    second = manager.sample_opponent(rng)

    assert first is not second
    assert first.tag == second.tag == "Only"


def test_sample_opponent_only_uses_latest_checkpoint_window():
    manager = SelfPlayManager(
        window_size=2,
        temperature=200,
    )

    manager.league = [
        make_entry(
            name="Baseline",
            kind="baseline",
            elo=1200,
        ),
        make_entry(
            name="Old",
            kind="checkpoint",
            timestep=10_000,
            elo=1200,
        ),
        make_entry(
            name="Recent1",
            kind="checkpoint",
            timestep=20_000,
            elo=1200,
        ),
        make_entry(
            name="Recent2",
            kind="checkpoint",
            timestep=30_000,
            elo=1200,
        ),
    ]

    rng = np.random.default_rng(123)

    selected_tags = {
        manager.sample_opponent(rng).tag
        for _ in range(500)
    }

    assert "Baseline" in selected_tags
    assert "Recent1" in selected_tags
    assert "Recent2" in selected_tags

    assert "Old" not in selected_tags


def test_closer_elo_is_sampled_more_often():
    manager = SelfPlayManager(
        window_size=8,
        temperature=100,
    )

    manager.league = [
        make_entry(
            name="Close",
            kind="baseline",
            elo=1210,
        ),
        make_entry(
            name="Far",
            kind="baseline",
            elo=1700,
        ),
    ]

    manager.learner_elo = 1200

    rng = np.random.default_rng(42)

    close_count = 0
    far_count = 0

    for _ in range(5_000):
        opponent = manager.sample_opponent(rng)

        if opponent.tag == "Close":
            close_count += 1

        elif opponent.tag == "Far":
            far_count += 1

    assert close_count > far_count


def test_equal_elo_entries_have_roughly_equal_sampling_probability():
    manager = SelfPlayManager(
        temperature=100,
    )

    manager.league = [
        make_entry(
            name="A",
            kind="baseline",
            elo=1200,
        ),
        make_entry(
            name="B",
            kind="baseline",
            elo=1200,
        ),
    ]

    manager.learner_elo = 1200

    rng = np.random.default_rng(42)

    counts = {
        "A": 0,
        "B": 0,
    }

    for _ in range(5_000):
        opponent = manager.sample_opponent(rng)
        counts[opponent.tag] += 1

    assert abs(counts["A"] - counts["B"]) < 500


def test_same_seed_produces_same_sampling_sequence():
    manager_one = SelfPlayManager(
        temperature=200,
    )

    manager_two = SelfPlayManager(
        temperature=200,
    )

    manager_one.league = [
        make_entry(
            name="A",
            kind="baseline",
            elo=1150,
        ),
        make_entry(
            name="B",
            kind="baseline",
            elo=1250,
        ),
    ]

    manager_two.league = [
        make_entry(
            name="A",
            kind="baseline",
            elo=1150,
        ),
        make_entry(
            name="B",
            kind="baseline",
            elo=1250,
        ),
    ]

    rng_one = np.random.default_rng(123)
    rng_two = np.random.default_rng(123)

    sequence_one = [
        manager_one.sample_opponent(rng_one).tag
        for _ in range(50)
    ]

    sequence_two = [
        manager_two.sample_opponent(rng_two).tag
        for _ in range(50)
    ]

    assert sequence_one == sequence_two

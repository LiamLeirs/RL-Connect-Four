# RL-Connect-Four

A reinforcement learning project for training an agent to play **Connect Four** using **PPO, self-play, action masking, and Elo-based opponent selection**.

The project explores how a reinforcement learning agent can progress from playing simple baseline opponents to competing against increasingly strong opponents and historical versions of itself.

## Overview

The main learning agent is trained using **Maskable PPO** from `sb3-contrib`. Illegal actions are handled through action masking, ensuring that the policy only selects columns in which a move can legally be made.

Training uses a self-play league containing several fixed baseline agents as well as historical PPO checkpoints.

The current baseline agents include:

- Random Agent
- Tactical Agent
- Minimax Agent (depth 2)
- Minimax Agent (depth 4)

As training progresses, snapshots of the PPO policy are added to the league. Each opponent is assigned an **Elo rating**, which is used to estimate opponent strength and influence matchmaking.

## Key Features

- Custom Connect Four Gymnasium environment
- Maskable PPO reinforcement learning agent
- Legal-action masking
- Self-play training
- Historical PPO checkpoint league
- Elo-based player ratings
- Elo-based opponent matchmaking
- Minimax with alpha-beta pruning
- Heuristic Connect Four evaluation
- Random and tactical baseline agents
- Deterministic and stochastic model evaluation
- Automated checkpoint evaluation
- TensorBoard training metrics
- Pytest test suite

## Self-Play Architecture

Training and evaluation are deliberately separated.

During normal training, PPO plays against opponents sampled from the self-play league. These games are used to update the neural network but do **not** directly update Elo.

At fixed rating intervals, the current PPO policy is evaluated against opponents with ratings close to the learner's current Elo.

```text
                    ┌─────────────────┐
                    │   Maskable PPO  │
                    │     Learner     │
                    └────────┬────────┘
                             │
                             │ training games
                             ▼
                    ┌─────────────────┐
                    │   Self-Play     │
                    │     League      │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
          Baselines     PPO Checkpoints   Minimax
```

The league allows the training distribution to evolve together with the learner instead of relying on a single fixed opponent.

## Elo Rating System

Every player in the league starts with an Elo rating of:

```text
1200
```

The expected score between players is calculated using the standard Elo expectation:

\[
E_A = \frac{1}{1 + 10^{(R_B-R_A)/400}}
\]

Game results are represented as:

| Result | Score |
| ------ | ----: |
| Win    |   1.0 |
| Draw   |   0.5 |
| Loss   |   0.0 |

Elo evaluation is performed separately from PPO training.

During a rating period:

1. The learner's Elo is frozen.
2. Several opponents near the learner's rating are selected.
3. The current policy plays evaluation matches against each opponent.
4. Elo changes are calculated for each matchup.
5. The changes are applied after all matchups have finished.

This avoids making the final rating dependent on the arbitrary order in which opponents are evaluated.

## Elo-Based Matchmaking

Training opponents are sampled from:

- fixed baseline agents;
- recent PPO checkpoints.

Historical checkpoints are limited to a sliding window so that the learner primarily trains against reasonably recent versions of itself.

Opponent probabilities depend on Elo distance:

\[
w_i = e^{-\frac{|R_L-R_i|}{T}}
\]

where:

- \(R_L\) is the learner Elo;
- \(R_i\) is the opponent Elo;
- \(T\) is a temperature parameter controlling how strongly matchmaking favors similarly rated opponents.

This creates a simple adaptive curriculum: as the PPO agent becomes stronger, the distribution of opponents it encounters changes with it.

## PPO Checkpoints

Rating updates and model checkpoints use separate schedules.

For example:

```text
10k  → Elo evaluation
20k  → Elo evaluation
25k  → Elo evaluation + checkpoint
30k  → Elo evaluation
40k  → Elo evaluation
50k  → Elo evaluation + checkpoint
...
```

When a checkpoint is created, the current PPO model is frozen and added to the league with the learner's current Elo.

Historical checkpoints can later become opponents for newer versions of the policy.

## Minimax Agent

The project also contains a traditional **Minimax Connect Four agent with alpha-beta pruning**.

Minimax serves two purposes:

1. It provides a strong non-learning baseline for evaluating PPO.
2. It provides stronger opponents in the self-play league.

Different search depths can be used:

```python
MiniMaxAgent(depth=2)
MiniMaxAgent(depth=4)
```

The heuristic considers features such as:

- two-in-a-row opportunities;
- three-in-a-row opportunities;
- opponent threats;
- center-column control.

Terminal positions receive large positive or negative values, with depth taken into account to prefer faster wins and delay unavoidable losses.

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd RL-Connect-Four
```

### 2. Create a Conda environment

The project is developed using **Python 3.10**.

```bash
conda create -n connect4-rl python=3.10
conda activate connect4-rl
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Training

Training uses `MaskablePPO` together with the custom Connect Four environment and self-play manager.

Run the training script from the repository root:

```bash
python -m src.training.train_maskable_ppo
```

The PPO learner is trained against opponents supplied by the self-play league.

A typical configuration uses parameters such as:

```text
Learning rate:     3e-4
Rollout steps:     1024
Batch size:        256
PPO epochs:        10
Gamma:             0.99
GAE lambda:        0.95
Entropy coefficient: 0.01
```

These parameters may change between experiments.

## Monitoring Training

Stable-Baselines3 metrics can be inspected using TensorBoard.

Start TensorBoard with:

```bash
tensorboard --logdir runs/
```

Then open the address shown by TensorBoard in your browser.

Useful PPO metrics include:

- `approx_kl`
- `clip_fraction`
- `entropy_loss`
- `explained_variance`
- `policy_gradient_loss`
- `value_loss`
- training FPS

The self-play callback additionally logs:

- current learner Elo;
- selected evaluation opponents;
- win/draw/loss results;
- proposed Elo changes;
- completed rating periods;
- saved PPO checkpoints;
- league size.

Example:

```text
Rating period | timestep=50000 | learner_elo=1342.6

vs PPO_25000    | W/D/L=72/2/26 | opponent_elo=1250.4 | proposed_delta=+...
vs Tactical     | W/D/L=64/1/35 | opponent_elo=1318.2 | proposed_delta=+...
vs MiniMax2     | W/D/L=18/0/82 | opponent_elo=1450.7 | proposed_delta=-...

Rating period complete | learner_elo=1342.6->...
```

## Evaluation

Agents can be evaluated over many Connect Four games using the project's evaluation utilities.

Evaluation tracks metrics including:

- wins;
- losses;
- draws;
- win rate;
- performance when moving first;
- performance when moving second;
- average reward;
- average game length;
- illegal actions.

For reproducible benchmark evaluation, the trained policy can be run deterministically.

For Elo evaluation during self-play, stochastic PPO actions can be used to generate a more representative distribution of games.

## Testing

The project includes unit tests for the environment, agents, Minimax implementation, and self-play components.

Run the full test suite with:

```bash
pytest
```

For more detailed output:

```bash
pytest -v
```

A specific test file can be run with:

```bash
pytest tests/test_minimax_agent.py -v
```

## Project Structure

A simplified overview of the repository:

```text
RL-Connect-Four/
│
├── src/
│   ├── agents/
│   │   └── agents.py
│   │
│   ├── envs/
│   │   ├── connect_four_env.py
│   │   └── connect_four/
│   │       └── game.py
│   │
│   ├── evaluation/
│   │   └── evaluator.py
│   │
│   ├── self_play/
│   │   └── self_play_manager.py
│   │
│   └── training/
│       └── train_maskable_ppo.py
│
├── tests/
├── models/
├── results/
├── runs/
├── requirements.txt
└── README.md
```

## Agents

### Random Agent

Selects uniformly from the currently legal actions.

### Tactical Agent

Uses simple tactical rules to make stronger decisions than a purely random policy.

### Minimax Agent

Uses depth-limited Minimax search with alpha-beta pruning and a Connect Four-specific heuristic.

### PPO Agent

A neural-network policy trained using Maskable PPO.

Historical versions of the PPO policy can be frozen and added to the self-play league.

## Motivation

Connect Four is a useful environment for experimenting with reinforcement learning because it has:

- simple rules;
- a discrete action space;
- deterministic transitions;
- sparse terminal rewards;
- adversarial gameplay;
- a large enough state space to make naive enumeration impractical;
- strong classical search algorithms that provide useful benchmarks.

This makes it possible to directly compare reinforcement learning against traditional game-playing techniques such as Minimax while experimenting with self-play and adaptive opponent selection.

## Current Research Questions

The project is intended to explore questions such as:

- Can PPO learn a strong Connect Four policy through self-play?
- How does PPO compare with depth-limited Minimax?
- Does Elo-based matchmaking provide a useful training curriculum?
- How quickly does the learner outperform its historical checkpoints?
- How important is opponent diversity during self-play?
- How does stochastic PPO evaluation compare with deterministic evaluation?
- How does policy strength evolve throughout training?

## Future Work

Possible extensions include:

- parallel/vectorized training environments;
- larger PPO architectures;
- deeper Minimax opponents;
- improved opponent sampling strategies;
- prioritized historical self-play;
- Elo visualizations over training time;
- checkpoint-vs-checkpoint tournaments;
- alternative rating systems such as Glicko;
- DQN-based agents;
- Monte Carlo Tree Search;
- AlphaZero-style policy/value learning;

## Technologies

- Python
- NumPy
- Gymnasium
- Stable-Baselines3
- sb3-contrib
- PyTorch
- Loguru
- TensorBoard
- Pytest

## License

This project is intended for educational and research purposes.

Add a license file if you plan to distribute or reuse the project publicly.

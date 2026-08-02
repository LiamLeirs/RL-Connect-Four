from src.agents.agents import RandomAgent

def evaluate_agent(env, agent, opponent=None, num_episodes=5000, seed=0):
    if opponent is None:
        opponent = RandomAgent()

    env.set_opponent(opponent)

    agent_wins = 0
    opponent_wins = 0
    draws = 0

    agent_first = 0
    agent_second = 0
    agent_first_wins = 0
    agent_second_wins = 0

    illegal_moves = 0
    total_reward = 0.0
    total_steps = 0

    for episode in range(num_episodes):
        observation, info = env.reset(
            seed=seed + episode
        )

        agent_player = info["agent_player"]

        if agent_player == 1:
            agent_first += 1
        else:
            agent_second += 1

        terminated = False
        truncated = False
        episode_reward = 0.0
        episode_steps = 0

        while not (terminated or truncated):
            action = agent.select_action(
                observation,
                info["action_mask"],
                env.np_random,
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            episode_reward += reward
            episode_steps += 1

            if info.get("illegal_action", False):
                illegal_moves += 1

        winner = info["winner"]

        if winner == 0:
            draws += 1
        elif winner == agent_player:
            agent_wins += 1

            if agent_player == 1:
                agent_first_wins += 1
            else:
                agent_second_wins += 1
        else:
            opponent_wins += 1

        total_reward += episode_reward
        total_steps += episode_steps

    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    return {
        "seed": seed,
        "num_episodes": num_episodes,
        "agent_wins": agent_wins,
        "opponent_wins": opponent_wins,
        "draws": draws,
        "agent_win_rate": safe_divide(
            agent_wins,
            num_episodes,
        ),
        "opponent_win_rate": safe_divide(
            opponent_wins,
            num_episodes,
        ),
        "draw_rate": safe_divide(
            draws,
            num_episodes,
        ),
        "agent_first_games": agent_first,
        "agent_second_games": agent_second,
        "agent_first_win_rate": safe_divide(
            agent_first_wins,
            agent_first,
        ),
        "agent_second_win_rate": safe_divide(
            agent_second_wins,
            agent_second,
        ),
        "average_reward": safe_divide(
            total_reward,
            num_episodes,
        ),
        "average_game_length": safe_divide(
            total_steps,
            num_episodes,
        ),
        "illegal_moves": illegal_moves,
        "illegal_moves_per_episode": safe_divide(
            illegal_moves,
            num_episodes,
        ),
        "illegal_action_rate": safe_divide(
            illegal_moves,
            total_steps,
        ),
    }

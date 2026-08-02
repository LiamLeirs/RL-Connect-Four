import argparse
import gymnasium as gym
from src.envs.connect_four_env import ConnectFourEnv
from src.agents.opponents import RandomOpponent
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=100)
    return parser.parse_args()

def evaluate_agent(env, args, opponent=RandomOpponent()):
    agent_first = 0
    agent_second = 0

    agent_first_wins = 0
    agent_second_wins = 0
    illegal_moves = 0
    draws = 0
    avg_reward = 0
    total_steps = 0
    env.set_opponent(opponent)

    for episode in range(args.num_episodes):
        observation, info = env.reset(seed=args.seed + episode)
        if info["agent_player"] == 1:
            agent_first += 1
        else:
            agent_second += 1
        terminated = False
        while not terminated:
            legal_actions = np.flatnonzero(info["action_mask"])
            action = int(env.np_random.choice(legal_actions))
            observation, reward, terminated, truncated, info = env.step(action)
            illegal_moves += info.get("illegal_action", False)
            avg_reward += reward
            total_steps += 1
            if terminated:
                if reward == 1.0:
                    if info["agent_player"] == 1:
                        agent_first_wins += 1
                    else:
                        agent_second_wins += 1
                elif reward == 0.0:
                    draws += 1

    agent_first_win_percentage = agent_first_wins / args.num_episodes if agent_first > 0 else 0
    agent_second_win_percentage = agent_second_wins / args.num_episodes if agent_second > 0 else 0
    draw_percentage = draws / args.num_episodes
    avg_reward /= args.num_episodes
    illegal_move_percentage = illegal_moves / args.num_episodes
    avg_game_length = total_steps / args.num_episodes

    return {
        "agent_first_win_percentage": agent_first_win_percentage,
        "agent_second_win_percentage": agent_second_win_percentage,
        "draw_percentage": draw_percentage,
        "average_reward": avg_reward,
        "illegal_move_percentage": illegal_move_percentage,
        "average_game_length": avg_game_length
    }

def main():
    args = parse_args()
    env = ConnectFourEnv(render_mode=None, opponent=RandomOpponent())
    results = evaluate_agent(env, args)
    env.close()

if __name__ == "__main__":
    main()
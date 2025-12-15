import argparse
from metamon.env import get_metamon_teams, BattleAgainstBaseline
from metamon.interface import (
    DefaultObservationSpace,
    DefaultShapedReward,
    BinaryReward,
    DefaultActionSpace,
    MinimalActionSpace,
)
from metamon.baselines import get_baseline


def make_env(
    battle_format: str,
    opponent_name: str,
    reward_name: str,
    action_space_name: str,
):
    team_set = get_metamon_teams(battle_format, "competitive")

    obs_space = DefaultObservationSpace()

    if reward_name == "default":
        reward_fn = DefaultShapedReward()
    elif reward_name == "binary":
        reward_fn = BinaryReward()
    else:
        raise ValueError(f"Unknown reward: {reward_name}")

    if action_space_name == "default":
        action_space = DefaultActionSpace()
    elif action_space_name == "minimal":
        action_space = MinimalActionSpace()
    else:
        raise ValueError(f"Unknown action space: {action_space_name}")

    env = BattleAgainstBaseline(
        battle_format=battle_format,
        observation_space=obs_space,
        action_space=action_space,
        reward_function=reward_fn,
        team_set=team_set,
        opponent_type=get_baseline(opponent_name),
        battle_backend="metamon",
    )

    return env


def run_random_policy(env, n_episodes: int):
    wins = 0
    total_return = 0.0

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            done = terminated or truncated

        total_return += ep_ret
        if info.get("won", False):
            wins += 1

        print(f"Episode {ep}: return={ep_ret:.2f}, won={info.get('won', False)}")

    print("-" * 40)
    print(f"Episodes: {n_episodes}")
    print(f"Win rate: {wins / n_episodes:.3f}")
    print(f"Avg return: {total_return / n_episodes:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--battle_format", type=str, default="gen1ou")
    parser.add_argument("--opponent", type=str, default="RandomBaseline")
    parser.add_argument("--reward", type=str, default="default", choices=["default", "binary"])
    parser.add_argument("--action_space", type=str, default="default", choices=["default", "minimal"])
    parser.add_argument("--episodes", type=int, default=10)

    args = parser.parse_args()

    env = make_env(
        battle_format=args.battle_format,
        opponent_name=args.opponent,
        reward_name=args.reward,
        action_space_name=args.action_space,
    )

    run_random_policy(env, n_episodes=args.episodes)

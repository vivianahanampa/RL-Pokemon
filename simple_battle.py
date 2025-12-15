# simple_battle.py
from metamon.env import get_metamon_teams, BattleAgainstBaseline
from metamon.interface import DefaultObservationSpace, DefaultShapedReward, DefaultActionSpace
from metamon.baselines import get_baseline


def make_env():
    team_set = get_metamon_teams("gen1ou", "competitive")
    obs_space = DefaultObservationSpace()
    reward_fn = DefaultShapedReward()
    action_space = DefaultActionSpace()

    env = BattleAgainstBaseline(
        battle_format="gen1ou",
        observation_space=obs_space,
        action_space=action_space,
        reward_function=reward_fn,
        team_set=team_set,
        opponent_type=get_baseline("RandomBaseline"),  # heuristic opponent
        battle_backend="metamon",                  # explicit, though it's default
    )
    return env


if __name__ == "__main__":
    env = make_env()

    n_episodes = 5
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_return = 0.0

        while not done:
            # random agent for now
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += reward
            done = terminated or truncated

        print(f"Episode {ep}: return={total_return:.2f}")
        print("Final info:", info)
        print("-" * 40)

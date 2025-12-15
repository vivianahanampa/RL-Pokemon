import sys
import random
import collections
from dataclasses import dataclass
from typing import Deque, Tuple, List
from metamon.interface import BinaryReward

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from metamon.env import get_metamon_teams, BattleAgainstBaseline
from metamon.interface import (
    DefaultObservationSpace,
    DefaultShapedReward,
    MinimalActionSpace,
)
from metamon.baselines import get_baseline

# Reward model given by host
from metamon.interface import DefaultShapedReward


# ==========================
#  Utils: obs -> 1D vector
# ==========================

def obs_to_vector(obs) -> np.ndarray:
    """
    Convert metamon observation to a 1D float32 vector.

    DefaultObservationSpace returns a dict with some numeric arrays
    and at least one long text field. Here we:
      - keep only numeric-like entries
      - skip any strings / object arrays
    """
    if isinstance(obs, dict):
        parts = []
        for k, v in obs.items():
            arr = np.asarray(v)

            # Skip pure strings or object/str arrays
            if arr.dtype.kind in ("U", "S", "O"):
                # print(f"Skipping text feature {k} with dtype {arr.dtype}")
                continue

            parts.append(arr.astype(np.float32).ravel())

        if not parts:
            # fallback to something trivial if everything was skipped
            return np.zeros(1, dtype=np.float32)

        return np.concatenate(parts, axis=0).astype(np.float32)

    # Non-dict obs: assume numeric
    arr = np.asarray(obs)
    if arr.dtype.kind in ("U", "S", "O"):
        raise ValueError(f"Observation is non-numeric: {obs}")
    return arr.astype(np.float32).ravel()


# ==========================
#  Replay Buffer
# ==========================

Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[Transition] = collections.deque(maxlen=capacity)

    def push(self, *transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ==========================
#  Q-Network
# ==========================

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==========================
#  DQN Agent
# ==========================

@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_capacity: int = 100_000
    min_buffer_size: int = 1_000
    target_update_interval: int = 1_000  # in steps
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig, device: str = "cpu"):
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.cfg = cfg

        self.q_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_q_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.lr)

        self.replay_buffer = ReplayBuffer(cfg.buffer_capacity)
        self.steps_done = 0

    def epsilon(self) -> float:
        # linear decay
        frac = min(self.steps_done / self.cfg.eps_decay_steps, 1.0)
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    def act(self, state_vec: np.ndarray) -> int:
        eps = self.epsilon()
        self.steps_done += 1
        if random.random() < eps:
            return random.randrange(self.n_actions)

        state_t = torch.from_numpy(
            state_vec).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def push_transition(self, transition: Transition):
        self.replay_buffer.push(*transition)

    def update(self):
        if len(self.replay_buffer) < self.cfg.min_buffer_size:
            return  # not enough data yet

        batch = self.replay_buffer.sample(self.cfg.batch_size)
        states, actions, rewards, next_states, dones = batch

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        # Q(s,a)
        q_values = self.q_net(states).gather(
            1, actions.unsqueeze(1)).squeeze(1)

        # max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states).max(dim=1)[0]
            target = rewards + self.cfg.gamma * next_q_values * (1.0 - dones)

        loss = nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

    def maybe_update_target(self):
        if self.steps_done % self.cfg.target_update_interval == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())


# ==========================
#  Env factory (Metamon)
# ==========================

def make_env():
    team_set = get_metamon_teams("gen1ou", "competitive")
    obs_space = DefaultObservationSpace()
    # reward_fn = BinaryReward()
    # Josie: I changed the reward function to shaped reward
    reward_fn = DefaultShapedReward()
    action_space = MinimalActionSpace()  # smaller action space for first experiments

    env = BattleAgainstBaseline(
        battle_format="gen1ou",
        observation_space=obs_space,
        action_space=action_space,
        reward_function=reward_fn,
        team_set=team_set,
        opponent_type=get_baseline("RandomBaseline"),  # medium difficulty
        battle_backend="metamon",
    )
    return env


# ==========================
#  Training Loop
# ==========================

def train_dqn(
    # 300 Josie: increased from 300 to 2000 for more thorough training
    num_episodes: int = 2000,
    max_steps_per_episode: int = 300,
    device: str = "cpu",
):
    env = make_env()

    # Get obs_dim / n_actions from a sample
    obs, info = env.reset()
    obs_vec = obs_to_vector(obs)
    obs_dim = obs_vec.shape[0]
    n_actions = env.action_space.n

    print(f"obs_dim = {obs_dim}, n_actions = {n_actions}")

    cfg = DQNConfig()
    agent = DQNAgent(obs_dim, n_actions, cfg, device=device)

    all_returns: List[float] = []
    all_wins: List[bool] = []

    global_step = 0

    for episode in range(num_episodes):
        obs, info = env.reset()
        state_vec = obs_to_vector(obs)
        done = False
        ep_return = 0.0
        ep_steps = 0
        won = False

        while not done and ep_steps < max_steps_per_episode:
            action = agent.act(state_vec)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state_vec = obs_to_vector(next_obs)
            done = terminated or truncated
            ep_return += reward
            ep_steps += 1
            global_step += 1

            agent.push_transition(
                (state_vec, action, reward, next_state_vec, done))
            agent.update()
            agent.maybe_update_target()

            state_vec = next_state_vec

        won = bool(info.get("won", False))
        all_returns.append(ep_return)
        all_wins.append(won)

        win_rate_last_20 = np.mean(
            all_wins[-20:]) if len(all_wins) >= 20 else np.mean(all_wins)
        avg_return_last_20 = np.mean(
            all_returns[-20:]) if len(all_returns) >= 20 else np.mean(all_returns)

        print(
            f"Ep {episode:3d} | "
            f"Return {ep_return:7.2f} | "
            f"Won {won} | "
            f"Last20 win_rate {win_rate_last_20:5.2f} | "
            f"Last20 avg_return {avg_return_last_20:7.2f} | "
            f"steps_done {agent.steps_done}"
        )

    print("Training finished.")
    overall_win_rate = np.mean(all_wins)
    overall_avg_return = np.mean(all_returns)
    print(
        f"Overall win rate: {overall_win_rate:.3f}, Overall avg return: {overall_avg_return:.2f}")

    torch.save(agent.q_net.state_dict(), "dqn_model.pth")
    print("Saved model to dqn_model.pth")


if __name__ == "__main__":
    # If you have a GPU and torch sees it, you can switch to "cuda"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dqn(num_episodes=2000, device=device)


# zzx
# ============================================================
#  Metamon Evaluate Wrapper — Required for `--eval_type pokeagent`
# ============================================================

class DQNEvalAgent:
    """
    Wrapper so Metamon evaluate() can use this agent.

    Metamon will:
      - Instantiate this class
      - Call .act(observation) each turn
    """

    def __init__(self, device="cpu"):
        print("[DQNEvalAgent] Initializing evaluation agent...")

        # Build env to infer obs/action structure
        self.env = make_env()
        obs, info = self.env.reset()

        # Convert observation to vector
        state_vec = obs_to_vector(obs)
        obs_dim = state_vec.shape[0]
        n_actions = self.env.action_space.n

        print(f"[DQNEvalAgent] obs_dim = {obs_dim}, n_actions = {n_actions}")

        # Build agent
        cfg = DQNConfig()
        self.agent = DQNAgent(obs_dim, n_actions, cfg, device=device)

        # Load trained weights if exist
        try:
            self.agent.q_net.load_state_dict(
                torch.load("dqn_model.pth", map_location=device)
            )
            print("[DQNEvalAgent] Loaded trained weights from dqn_model.pth")
        except FileNotFoundError:
            print(
                "[DQNEvalAgent] WARNING: dqn_model.pth not found, using untrained network")

        # Evaluation should not use exploration
        self.agent.epsilon = lambda: 0.0  # disable ε-greedy

    def act(self, obs):
        """
        Metamon calls this every step with the current observation.
        """
        state_vec = obs_to_vector(obs)
        action = self.agent.act(state_vec)
        return action

# ============================================================
#  Evaluate DQN Against 6 Baselines
# ============================================================


def evaluate_dqn_against_all_baselines(
    model_path="dqn_model.pth",
    num_battles=100,
    device="cpu"
):
    print("\n========== DQN Evaluation Against All Baselines ==========")

    baselines_to_test = [
        "RandomBaseline",
        "PokeEnvHeuristic",
        "Gen1BossAI",
        "Grunt",
        "GymLeader",
        "EmeraldKaizo",
    ]

    results = {}

    for baseline_name in baselines_to_test:
        print(f"\n--- Evaluating vs {baseline_name} ---")

        # Build environment for this baseline
        team_set = get_metamon_teams("gen1ou", "competitive")

        env = BattleAgainstBaseline(
            battle_format="gen1ou",
            observation_space=DefaultObservationSpace(),
            action_space=MinimalActionSpace(),
            reward_function=BinaryReward(),
            team_set=team_set,
            opponent_type=get_baseline(baseline_name),
            battle_backend="metamon",
        )

        # Load observation / action shape
        obs, info = env.reset()
        obs_dim = obs_to_vector(obs).shape[0]
        n_actions = env.action_space.n

        cfg = DQNConfig()
        agent = DQNAgent(obs_dim, n_actions, cfg, device=device)

        # Load trained model
        agent.q_net.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        agent.epsilon = lambda: 0.0  # no exploration in evaluation

        win_list = []
        return_list = []

        for _ in range(num_battles):
            obs, info = env.reset()
            state = obs_to_vector(obs)
            done = False
            ep_return = 0

            while not done:
                action = agent.act(state)
                next_obs, reward, terminated, truncated, info = env.step(
                    action)
                ep_return += reward
                done = terminated or truncated
                state = obs_to_vector(next_obs)

            win = bool(info.get("won", False))
            win_list.append(win)
            return_list.append(ep_return)

        win_rate = float(np.mean(win_list))
        avg_return = float(np.mean(return_list))

        print(
            f"{baseline_name} → Win Rate: {win_rate:.3f}, Avg Return: {avg_return:.2f}")

        results[baseline_name] = {
            "win_rate": win_rate,
            "avg_return": avg_return,
        }

    print("\n========== Final Evaluation Summary ==========")
    for k, v in results.items():
        print(
            f"{k:15s} | Win Rate = {v['win_rate']:.3f}, Avg Return = {v['avg_return']:.2f}")

    return results

# Run evaluation only
# python -m metamon.rl.my_agents.dqn_agent --eval


if "--eval" in sys.argv:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_dqn_against_all_baselines(
        model_path="dqn_model.pth",
        num_battles=20,
        device=device
    )
    exit()

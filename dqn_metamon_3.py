import random
import collections
from dataclasses import dataclass
from typing import Deque, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from metamon.env import get_metamon_teams, BattleAgainstBaseline
from metamon.interface import (
    DefaultObservationSpace,
    DefaultShapedReward,
    MinimalActionSpace,
    DefaultActionSpace
)
from metamon.baselines import get_baseline


# ==========================
#  Utils: obs -> 1D vector
# ==========================

def obs_to_vector(obs) -> np.ndarray:
    """Convert metamon observation to a 1D float32 vector."""
    if isinstance(obs, dict):
        parts = []
        for k, v in obs.items():
            arr = np.asarray(v)
            if arr.dtype.kind in ("U", "S", "O"):
                continue
            parts.append(arr.astype(np.float32).ravel())
        if not parts:
            return np.zeros(1, dtype=np.float32)
        return np.concatenate(parts, axis=0).astype(np.float32)
    
    arr = np.asarray(obs)
    if arr.dtype.kind in ("U", "S", "O"):
        raise ValueError(f"Observation is non-numeric: {obs}")
    return arr.astype(np.float32).ravel()


# ==========================
#  Prioritized Replay Buffer
# ==========================

Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: Deque[Transition] = collections.deque(maxlen=capacity)
        self.priorities: Deque[float] = collections.deque(maxlen=capacity)
        self.max_priority = 1.0

    def push(self, *transition: Transition):
        self.buffer.append(transition)
        self.priorities.append(self.max_priority)

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
            weights,
            indices
        )
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)


# ==========================
#  Dueling Q-Network
# ==========================

class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


# ==========================
#  Double DQN Agent
# ==========================

@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 5e-4  # Reduced learning rate
    batch_size: int = 128  # Larger batch size
    buffer_capacity: int = 200_000  # Larger buffer
    min_buffer_size: int = 5_000  # More initial exploration
    target_update_interval: int = 500  # More frequent updates
    eps_start: float = 1.0
    eps_end: float = 0.01  # Lower final epsilon
    eps_decay_steps: int = 50_000  # Longer decay
    grad_clip: float = 10.0
    use_prioritized: bool = True
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_end: float = 1.0
    priority_beta_steps: int = 50_000


class DoubleDQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig, device: str = "cpu"):
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.cfg = cfg

        # Use Dueling architecture
        self.q_net = DuelingQNetwork(obs_dim, n_actions).to(self.device)
        self.target_q_net = DuelingQNetwork(obs_dim, n_actions).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        
        # Use prioritized replay if enabled
        if cfg.use_prioritized:
            self.replay_buffer = PrioritizedReplayBuffer(cfg.buffer_capacity, cfg.priority_alpha)
        else:
            from collections import deque
            self.replay_buffer = type('ReplayBuffer', (), {
                'buffer': deque(maxlen=cfg.buffer_capacity),
                'push': lambda self, *t: self.buffer.append(t),
                'sample': lambda self, bs: self._sample(bs),
                '__len__': lambda self: len(self.buffer),
                '_sample': lambda self, bs: self._do_sample(bs)
            })()
            
        self.steps_done = 0
        self.update_count = 0

    def epsilon(self) -> float:
        frac = min(self.steps_done / self.cfg.eps_decay_steps, 1.0)
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)
    
    def priority_beta(self) -> float:
        frac = min(self.update_count / self.cfg.priority_beta_steps, 1.0)
        return self.cfg.priority_beta_start + frac * (self.cfg.priority_beta_end - self.cfg.priority_beta_start)

    def act(self, state_vec: np.ndarray, eval_mode: bool = False) -> int:
        if not eval_mode:
            eps = self.epsilon()
            self.steps_done += 1
            if random.random() < eps:
                return random.randrange(self.n_actions)

        state_t = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def push_transition(self, transition: Transition):
        self.replay_buffer.push(*transition)

    def update(self):
        if len(self.replay_buffer) < self.cfg.min_buffer_size:
            return

        # Sample batch
        if self.cfg.use_prioritized:
            beta = self.priority_beta()
            batch = self.replay_buffer.sample(self.cfg.batch_size, beta)
            states, actions, rewards, next_states, dones, weights, indices = batch
            weights = torch.from_numpy(weights).float().to(self.device)
        else:
            batch = self.replay_buffer.sample(self.cfg.batch_size)
            states, actions, rewards, next_states, dones = batch
            weights = torch.ones(self.cfg.batch_size).to(self.device)

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        # Current Q values
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: use online network to select actions, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1)
            next_q_values = self.target_q_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + self.cfg.gamma * next_q_values * (1.0 - dones)

        # Weighted loss for prioritized replay
        td_errors = q_values - target
        loss = (weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        
        # Update priorities
        if self.cfg.use_prioritized:
            priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
            self.replay_buffer.update_priorities(indices, priorities)
        
        self.update_count += 1

    def maybe_update_target(self):
        if self.steps_done % self.cfg.target_update_interval == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())


# ==========================
#  Env factory
# ==========================

def make_env(action_space_type: str = "minimal"):
    team_set = get_metamon_teams("gen1ou", "competitive")
    obs_space = DefaultObservationSpace()
    reward_fn = DefaultShapedReward()

    if action_space_type == "minimal":
        action_space = MinimalActionSpace()
    elif action_space_type == "default":
        action_space = DefaultActionSpace()
    else:
        raise ValueError(f"Unknown action space type: {action_space_type}")

    env = BattleAgainstBaseline(
        battle_format="gen1ou",
        observation_space=obs_space,
        action_space=action_space,
        reward_function=reward_fn,
        team_set=team_set,
        opponent_type=get_baseline("Gen1BossAI"),
        battle_backend="metamon",
    )
    return env


# ==========================
#  Training Loop
# ==========================

def train_dqn(
    num_episodes: int = 1000,
    max_steps_per_episode: int = 300,
    device: str = "cpu",
    action_space_type: str = "minimal",
    eval_interval: int = 50,
    eval_episodes: int = 10,
):
    env = make_env(action_space_type=action_space_type)

    obs, info = env.reset()
    obs_vec = obs_to_vector(obs)
    obs_dim = obs_vec.shape[0]
    n_actions = env.action_space.n

    print(f"obs_dim = {obs_dim}, n_actions = {n_actions}")

    cfg = DQNConfig()
    agent = DoubleDQNAgent(obs_dim, n_actions, cfg, device=device)

    all_returns: List[float] = []
    all_wins: List[bool] = []
    eval_win_rates: List[float] = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        state_vec = obs_to_vector(obs)
        done = False
        ep_return = 0.0
        ep_steps = 0

        while not done and ep_steps < max_steps_per_episode:
            action = agent.act(state_vec)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state_vec = obs_to_vector(next_obs)
            done = terminated or truncated
            ep_return += reward
            ep_steps += 1

            agent.push_transition((state_vec, action, reward, next_state_vec, done))
            agent.update()
            agent.maybe_update_target()

            state_vec = next_state_vec

        won = bool(info.get("won", False))
        all_returns.append(ep_return)
        all_wins.append(won)

        # Evaluation
        if (episode + 1) % eval_interval == 0:
            eval_wins = 0
            for _ in range(eval_episodes):
                obs, info = env.reset()
                state_vec = obs_to_vector(obs)
                done = False
                steps = 0
                while not done and steps < max_steps_per_episode:
                    action = agent.act(state_vec, eval_mode=True)
                    next_obs, _, terminated, truncated, info = env.step(action)
                    state_vec = obs_to_vector(next_obs)
                    done = terminated or truncated
                    steps += 1
                if info.get("won", False):
                    eval_wins += 1
            eval_win_rate = eval_wins / eval_episodes
            eval_win_rates.append(eval_win_rate)
            print(f"\n*** EVAL at ep {episode+1}: Win rate = {eval_win_rate:.2%} ***\n")

        win_rate_last_50 = np.mean(all_wins[-50:]) if len(all_wins) >= 50 else np.mean(all_wins)
        avg_return_last_50 = np.mean(all_returns[-50:]) if len(all_returns) >= 50 else np.mean(all_returns)

        print(
            f"Ep {episode:4d} | "
            f"Return {ep_return:7.2f} | "
            f"Won {won} | "
            f"L50 WR {win_rate_last_50:5.2%} | "
            f"L50 Ret {avg_return_last_50:7.2f} | "
            f"ε {agent.epsilon():.3f} | "
            f"Buf {len(agent.replay_buffer)}"
        )

    print("\nTraining finished.")
    overall_win_rate = np.mean(all_wins)
    overall_avg_return = np.mean(all_returns)
    print(f"Overall win rate: {overall_win_rate:.2%}")
    print(f"Overall avg return: {overall_avg_return:.2f}")
    if eval_win_rates:
        print(f"Best eval win rate: {max(eval_win_rates):.2%}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action_space",
        type=str,
        choices=["minimal", "default"],
        default="minimal",
        help="Which action space to use for DQN training.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of training episodes.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, action_space={args.action_space}")

    train_dqn(
        num_episodes=args.episodes,
        device=device,
        action_space_type=args.action_space,
    )
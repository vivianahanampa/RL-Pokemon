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
#  Enhanced obs -> vector with normalization
# ==========================

class ObservationProcessor:
    def __init__(self):
        self.mean = None
        self.std = None
        self.count = 0
        
    def process(self, obs) -> np.ndarray:
        """Convert and normalize observation."""
        if isinstance(obs, dict):
            parts = []
            for k, v in obs.items():
                arr = np.asarray(v)
                if arr.dtype.kind in ("U", "S", "O"):
                    continue
                parts.append(arr.astype(np.float32).ravel())
            if not parts:
                return np.zeros(1, dtype=np.float32)
            vec = np.concatenate(parts, axis=0).astype(np.float32)
        else:
            arr = np.asarray(obs)
            if arr.dtype.kind in ("U", "S", "O"):
                raise ValueError(f"Observation is non-numeric: {obs}")
            vec = arr.astype(np.float32).ravel()
        
        # Online normalization
        if self.mean is None:
            self.mean = vec.copy()
            self.std = np.ones_like(vec)
        else:
            self.count += 1
            alpha = min(1.0 / self.count, 0.01)  # Decay factor
            self.mean = (1 - alpha) * self.mean + alpha * vec
            self.std = (1 - alpha) * self.std + alpha * np.abs(vec - self.mean)
            
        # Normalize
        return (vec - self.mean) / (self.std + 1e-8)


# ==========================
#  Prioritized Replay Buffer with n-step returns
# ==========================

Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, n_step: int = 3, gamma: float = 0.99):
        self.capacity = capacity
        self.alpha = alpha
        self.n_step = n_step
        self.gamma = gamma
        self.buffer: Deque[Transition] = collections.deque(maxlen=capacity)
        self.priorities: Deque[float] = collections.deque(maxlen=capacity)
        self.n_step_buffer: Deque[Transition] = collections.deque(maxlen=n_step)
        self.max_priority = 1.0

    def push(self, *transition: Transition):
        self.n_step_buffer.append(transition)
        
        if len(self.n_step_buffer) < self.n_step:
            return
            
        # Compute n-step return
        state, action = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
        n_step_reward = 0.0
        done = False
        
        for i, (s, a, r, ns, d) in enumerate(self.n_step_buffer):
            n_step_reward += (self.gamma ** i) * r
            if d:
                done = True
                next_state = ns
                break
        
        if not done:
            next_state = self.n_step_buffer[-1][3]
            
        self.buffer.append((state, action, n_step_reward, next_state, done))
        self.priorities.append(self.max_priority)

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        
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
#  Noisy Dueling Q-Network
# ==========================

class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration."""
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)


class NoisyDuelingQNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 512):
        super().__init__()
        # Larger network
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Value stream with noisy layers
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            NoisyLinear(hidden_dim // 2, 1)
        )
        
        # Advantage stream with noisy layers
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            NoisyLinear(hidden_dim // 2, n_actions)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values
    
    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# ==========================
#  Rainbow DQN Agent
# ==========================

@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 128
    buffer_capacity: int = 300_000
    min_buffer_size: int = 10_000
    target_update_interval: int = 1000
    n_step: int = 3
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_end: float = 1.0
    priority_beta_steps: int = 100_000
    grad_clip: float = 10.0
    weight_decay: float = 1e-5


class RainbowDQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig, device: str = "cpu"):
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.cfg = cfg

        self.q_net = NoisyDuelingQNetwork(obs_dim, n_actions).to(self.device)
        self.target_q_net = NoisyDuelingQNetwork(obs_dim, n_actions).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = optim.AdamW(
            self.q_net.parameters(), 
            lr=cfg.lr, 
            weight_decay=cfg.weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100_000, eta_min=1e-5
        )
        
        self.replay_buffer = PrioritizedReplayBuffer(
            cfg.buffer_capacity, 
            cfg.priority_alpha,
            cfg.n_step,
            cfg.gamma
        )
            
        self.steps_done = 0
        self.update_count = 0

    def priority_beta(self) -> float:
        frac = min(self.update_count / self.cfg.priority_beta_steps, 1.0)
        return self.cfg.priority_beta_start + frac * (self.cfg.priority_beta_end - self.cfg.priority_beta_start)

    def act(self, state_vec: np.ndarray, eval_mode: bool = False) -> int:
        if not eval_mode:
            self.steps_done += 1
            # Reset noise for exploration
            self.q_net.reset_noise()

        state_t = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def push_transition(self, transition: Transition):
        self.replay_buffer.push(*transition)

    def update(self):
        if len(self.replay_buffer) < self.cfg.min_buffer_size:
            return

        beta = self.priority_beta()
        batch = self.replay_buffer.sample(self.cfg.batch_size, beta)
        states, actions, rewards, next_states, dones, weights, indices = batch
        
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)
        weights = torch.from_numpy(weights).float().to(self.device)

        # Current Q values
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            self.target_q_net.reset_noise()
            next_actions = self.q_net(next_states).argmax(dim=1)
            next_q_values = self.target_q_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # n-step gamma already applied in buffer
            target = rewards + (self.cfg.gamma ** self.cfg.n_step) * next_q_values * (1.0 - dones)

        # Huber loss (more robust than MSE)
        td_errors = q_values - target
        huber_loss = nn.functional.smooth_l1_loss(q_values, target, reduction='none')
        loss = (weights * huber_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        
        # Update priorities
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
#  Training Loop with curriculum
# ==========================

def train_dqn(
    num_episodes: int = 1000,
    max_steps_per_episode: int = 300,
    device: str = "cpu",
    action_space_type: str = "minimal",
    eval_interval: int = 50,
    eval_episodes: int = 20,
    save_path: str = "best_model.pt",
):
    env = make_env(action_space_type=action_space_type)
    obs_processor = ObservationProcessor()

    obs, info = env.reset()
    obs_vec = obs_processor.process(obs)
    obs_dim = obs_vec.shape[0]
    n_actions = env.action_space.n

    print(f"obs_dim = {obs_dim}, n_actions = {n_actions}")

    cfg = DQNConfig()
    agent = RainbowDQNAgent(obs_dim, n_actions, cfg, device=device)

    all_returns: List[float] = []
    all_wins: List[bool] = []
    eval_win_rates: List[float] = []
    best_eval_wr = 0.0

    for episode in range(num_episodes):
        obs, info = env.reset()
        state_vec = obs_processor.process(obs)
        done = False
        ep_return = 0.0
        ep_steps = 0

        while not done and ep_steps < max_steps_per_episode:
            action = agent.act(state_vec)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state_vec = obs_processor.process(next_obs)
            done = terminated or truncated
            ep_return += reward
            ep_steps += 1

            agent.push_transition((state_vec, action, reward, next_state_vec, done))
            
            # Multiple updates per step for better sample efficiency
            if len(agent.replay_buffer) >= cfg.min_buffer_size:
                for _ in range(2):
                    agent.update()
            
            agent.maybe_update_target()
            state_vec = next_state_vec

        won = bool(info.get("won", False))
        all_returns.append(ep_return)
        all_wins.append(won)

        # Evaluation
        if (episode + 1) % eval_interval == 0:
            eval_wins = 0
            eval_returns = []
            for _ in range(eval_episodes):
                obs, info = env.reset()
                state_vec = obs_processor.process(obs)
                done = False
                steps = 0
                ep_ret = 0
                while not done and steps < max_steps_per_episode:
                    action = agent.act(state_vec, eval_mode=True)
                    next_obs, r, terminated, truncated, info = env.step(action)
                    state_vec = obs_processor.process(next_obs)
                    done = terminated or truncated
                    ep_ret += r
                    steps += 1
                eval_returns.append(ep_ret)
                if info.get("won", False):
                    eval_wins += 1
                    
            eval_win_rate = eval_wins / eval_episodes
            eval_avg_return = np.mean(eval_returns)
            eval_win_rates.append(eval_win_rate)
            
            print(f"\n{'='*80}")
            print(f"EVAL at episode {episode+1}:")
            print(f"  Win Rate: {eval_win_rate:.1%} ({eval_wins}/{eval_episodes})")
            print(f"  Avg Return: {eval_avg_return:.2f}")
            print(f"  Buffer Size: {len(agent.replay_buffer)}")
            print(f"  Learning Rate: {agent.optimizer.param_groups[0]['lr']:.2e}")
            print(f"{'='*80}\n")
            
            # Save best model
            if eval_win_rate > best_eval_wr:
                best_eval_wr = eval_win_rate
                torch.save({
                    'episode': episode,
                    'q_net': agent.q_net.state_dict(),
                    'optimizer': agent.optimizer.state_dict(),
                    'eval_win_rate': eval_win_rate,
                }, save_path)
                print(f"💾 New best model saved! Win rate: {eval_win_rate:.1%}\n")

        # Logging
        if (episode + 1) % 10 == 0:
            win_rate_last_100 = np.mean(all_wins[-100:]) if len(all_wins) >= 100 else np.mean(all_wins)
            avg_return_last_100 = np.mean(all_returns[-100:]) if len(all_returns) >= 100 else np.mean(all_returns)
            
            print(
                f"Ep {episode+1:4d} | "
                f"Ret {ep_return:7.2f} | "
                f"Won {int(won)} | "
                f"L100 WR {win_rate_last_100:5.1%} | "
                f"L100 Ret {avg_return_last_100:7.2f} | "
                f"Buf {len(agent.replay_buffer):6d} | "
                f"Updates {agent.update_count:6d}"
            )

    print("\n" + "="*80)
    print("Training finished!")
    print(f"Overall win rate: {np.mean(all_wins):.1%}")
    print(f"Overall avg return: {np.mean(all_returns):.2f}")
    if eval_win_rates:
        print(f"Best eval win rate: {max(eval_win_rates):.1%}")
    print("="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--action_space", type=str, choices=["minimal", "default"], 
                       default="minimal", help="Action space type")
    parser.add_argument("--episodes", type=int, default=1000, help="Training episodes")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--save", type=str, default="best_metamon_agent.pt", 
                       help="Path to save best model")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, action_space={args.action_space}")

    train_dqn(
        num_episodes=args.episodes,
        device=device,
        action_space_type=args.action_space,
        save_path=args.save,
    )
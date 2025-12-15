import sys
import random
import collections
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from metamon.env import get_metamon_teams, BattleAgainstBaseline
from metamon.interface import (
    DefaultObservationSpace,
    DefaultShapedReward,
    MinimalActionSpace,
    BinaryReward,
)
from metamon.baselines import get_baseline

# ==========================
#  Enhanced Utils: obs -> 1D vector
# ==========================
def obs_to_vector(obs) -> np.ndarray:
    """
    Convert metamon observation to a 1D float32 vector with better feature engineering.
    """
    if isinstance(obs, dict):
        numeric_features = []
        # Look for common numeric fields in metamon
        for k, v in obs.items():
            arr = np.asarray(v)
            # Skip text/object arrays
            if arr.dtype.kind in ("U", "S", "O"):
                continue
            # Flatten and add
            flat_arr = arr.ravel()
            # Normalize if it's HP (usually 0-100)
            if 'hp' in k.lower():
                flat_arr = flat_arr / 100.0
            # Normalize if it's stat (typical range 0-255)
            elif any(stat in k.lower() for stat in ['atk', 'def', 'spa', 'spd', 'spe']):
                flat_arr = flat_arr / 255.0
            numeric_features.append(flat_arr)
        
        if numeric_features:
            vector = np.concatenate(numeric_features, axis=0).astype(np.float32)
        else:
            vector = np.zeros(1, dtype=np.float32)
        
        # Add some engineered features
        engineered = []
        if 'active' in obs:
            try:
                active_arr = np.asarray(obs['active']).ravel()
                if active_arr.dtype.kind not in ("U", "S", "O"):
                    engineered.append(active_arr.astype(np.float32))
            except:
                pass
        
        if engineered:
            vector = np.concatenate([vector] + engineered, axis=0)
        return vector
    
    # Non-dict obs
    arr = np.asarray(obs)
    if arr.dtype.kind in ("U", "S", "O"):
        return np.zeros(1, dtype=np.float32)
    return arr.astype(np.float32).ravel()


# ==========================
#  Prioritized Experience Replay with N-step
# ==========================
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, 
                 n_step: int = 3, gamma: float = 0.99):
        self.capacity = capacity
        self.alpha = alpha  # prioritization exponent
        self.beta = beta    # importance sampling exponent
        self.beta_increment = 0.001
        self.n_step = n_step
        self.gamma = gamma
        
        self.buffer = collections.deque(maxlen=capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # N-step buffer
        self.n_step_buffer = collections.deque(maxlen=n_step)
    
    def _get_n_step_info(self):
        """Calculate n-step return and final state"""
        reward = sum([self.gamma ** i * transition[2] 
                     for i, transition in enumerate(self.n_step_buffer)])
        final_state = self.n_step_buffer[-1][3]
        done = self.n_step_buffer[-1][4]
        return reward, final_state, done
    
    def push(self, *transition):
        self.n_step_buffer.append(transition)
        
        if len(self.n_step_buffer) < self.n_step:
            return
        
        # Get n-step transition
        n_reward, n_next_state, n_done = self._get_n_step_info()
        state, action = self.n_step_buffer[0][:2]
        
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append((state, action, n_reward, n_next_state, n_done))
        else:
            self.buffer[self.position] = (state, action, n_reward, n_next_state, n_done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        if self.size == 0:
            return None
        
        priorities = self.priorities[:self.size] ** self.alpha
        probs = priorities / priorities.sum()
        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        states, actions, rewards, next_states, dones = zip(*samples)
        
        # Importance sampling weights
        total = self.size
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        weights = torch.FloatTensor(weights)
        
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            weights,
        )
    
    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + 1e-6)
    
    def __len__(self):
        return self.size


# ==========================
#  Noisy Linear Layer
# ==========================
class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration"""
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
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
        return x.sign().mul(x.abs().sqrt())
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# ==========================
#  Rainbow DQN Network (Dueling + Noisy)
# ==========================
class RainbowQNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dims: List[int] = [1024, 512, 256]):
        super().__init__()
        
        # Shared feature extractor with LayerNorm
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        self.feature = nn.Sequential(*layers)
        
        # Dueling architecture with Noisy layers
        # Value stream
        self.value_hidden = NoisyLinear(prev_dim, 256)
        self.value = NoisyLinear(256, 1)
        
        # Advantage stream
        self.advantage_hidden = NoisyLinear(prev_dim, 256)
        self.advantage = NoisyLinear(256, n_actions)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        
        # Value stream
        value = F.relu(self.value_hidden(features))
        value = self.value(value)
        
        # Advantage stream
        advantage = F.relu(self.advantage_hidden(features))
        advantage = self.advantage(advantage)
        
        # Combine using dueling architecture
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
    
    def reset_noise(self):
        """Reset noise for all noisy layers"""
        self.value_hidden.reset_noise()
        self.value.reset_noise()
        self.advantage_hidden.reset_noise()
        self.advantage.reset_noise()


# ==========================
#  Rainbow DQN Configuration
# ==========================
@dataclass
class RainbowConfig:
    # Hyperparameters
    gamma: float = 0.99
    lr: float = 1e-4
    lr_decay: float = 0.999
    batch_size: int = 64
    buffer_capacity: int = 100_000
    min_buffer_size: int = 5_000
    target_update_interval: int = 1000
    
    # Rainbow components
    use_double_dqn: bool = True
    use_dueling: bool = True
    use_noisy: bool = True
    use_per: bool = True
    use_n_step: bool = True
    
    # PER parameters
    per_alpha: float = 0.6
    per_beta: float = 0.4
    
    # N-step parameters
    n_step: int = 3
    
    # Network
    hidden_dims: List[int] = field(default_factory=lambda: [1024, 512, 256])
    
    # Training
    gradient_clip: float = 10.0
    loss_fn: str = 'huber'  # 'mse' or 'huber'


# ==========================
#  Rainbow DQN Agent
# ==========================
class RainbowDQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: RainbowConfig, device: str = "cpu"):
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.cfg = cfg
        
        # Create Q-networks
        self.q_net = RainbowQNetwork(obs_dim, n_actions, cfg.hidden_dims).to(self.device)
        self.target_q_net = RainbowQNetwork(obs_dim, n_actions, cfg.hidden_dims).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # Optimizer with learning rate scheduler
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=cfg.lr_decay)
        
        # Replay buffer
        if cfg.use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                cfg.buffer_capacity, 
                alpha=cfg.per_alpha, 
                beta=cfg.per_beta,
                n_step=cfg.n_step if cfg.use_n_step else 1,
                gamma=cfg.gamma
            )
        else:
            self.replay_buffer = collections.deque(maxlen=cfg.buffer_capacity)
        
        self.steps_done = 0
        self.training_losses = []
        
        # Action tracking for Figure 3
        self.action_counts = np.zeros(n_actions, dtype=np.int64)
        self.eval_action_counts = np.zeros(n_actions, dtype=np.int64)
    
    def act(self, state_vec: np.ndarray, eval_mode: bool = False) -> int:
        """Select action using noisy networks (no epsilon needed)"""
        self.steps_done += 1
        
        state_t = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
        
        if eval_mode:
            self.q_net.eval()
        else:
            self.q_net.train()
        
        with torch.no_grad():
            q_values = self.q_net(state_t)
        
        action = int(q_values.argmax(dim=1).item())
        
        # Track actions
        if eval_mode:
            self.eval_action_counts[action] += 1
        else:
            self.action_counts[action] += 1
        
        return action
    
    def push_transition(self, transition):
        if self.cfg.use_per:
            self.replay_buffer.push(*transition)
        else:
            self.replay_buffer.append(transition)
    
    def update(self) -> Optional[float]:
        if self.cfg.use_per:
            if len(self.replay_buffer) < self.cfg.min_buffer_size:
                return None
            batch = self.replay_buffer.sample(self.cfg.batch_size)
            if batch is None:
                return None
            states, actions, rewards, next_states, dones, indices, weights = batch
            weights = weights.to(self.device)
        else:
            if len(self.replay_buffer) < self.cfg.min_buffer_size:
                return None
            batch = random.sample(self.replay_buffer, self.cfg.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = np.stack(states)
            actions = np.array(actions, dtype=np.int64)
            rewards = np.array(rewards, dtype=np.float32)
            next_states = np.stack(next_states)
            dones = np.array(dones, dtype=np.float32)
            weights = torch.ones(self.cfg.batch_size, device=self.device)
            indices = None
        
        # Convert to tensors
        states_t = torch.from_numpy(states).float().to(self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        next_states_t = torch.from_numpy(next_states).float().to(self.device)
        dones_t = torch.from_numpy(dones).float().to(self.device)
        
        # Current Q values
        current_q = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        # Double DQN target
        with torch.no_grad():
            if self.cfg.use_double_dqn:
                # Use online network to select actions
                next_actions = self.q_net(next_states_t).argmax(dim=1)
                # Use target network to evaluate actions
                next_q = self.target_q_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q = self.target_q_net(next_states_t).max(dim=1)[0]
            
            # N-step return
            n_step_factor = self.cfg.gamma ** (self.cfg.n_step if self.cfg.use_n_step else 1)
            target = rewards_t + n_step_factor * next_q * (1 - dones_t)
        
        # Compute loss
        if self.cfg.loss_fn == 'huber':
            loss = F.smooth_l1_loss(current_q, target, reduction='none')
        else:  # mse
            loss = F.mse_loss(current_q, target, reduction='none')
        
        # Weighted loss for PER
        weighted_loss = (loss * weights).mean()
        
        # Update priorities if using PER
        if self.cfg.use_per and indices is not None:
            errors = loss.detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, errors)
        
        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.gradient_clip)
        self.optimizer.step()
        
        # Reset noise
        if self.cfg.use_noisy:
            self.q_net.reset_noise()
            self.target_q_net.reset_noise()
        
        self.training_losses.append(weighted_loss.item())
        return weighted_loss.item()
    
    def soft_update_target(self):
        """Hard update target network (Rainbow uses hard updates)"""
        self.target_q_net.load_state_dict(self.q_net.state_dict())
    
    def hard_update_target(self):
        """Hard update target network parameters"""
        self.target_q_net.load_state_dict(self.q_net.state_dict())
    
    def save(self, path: str):
        """Save agent state"""
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_q_net_state_dict': self.target_q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'steps_done': self.steps_done,
            'cfg': self.cfg,
            'action_counts': self.action_counts,
            'eval_action_counts': self.eval_action_counts,
        }, path)
    
    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_q_net.load_state_dict(checkpoint['target_q_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.steps_done = checkpoint['steps_done']
        if 'action_counts' in checkpoint:
            self.action_counts = checkpoint['action_counts']
            self.eval_action_counts = checkpoint['eval_action_counts']
        # Set networks to eval mode
        self.q_net.eval()
        self.target_q_net.eval()


# ==========================
#  Environment Wrapper
# ==========================
class BattleEnvWrapper:
    def __init__(self, baseline_name: str = "Gen1BossAI", use_shaped_reward: bool = True):
        self.baseline_name = baseline_name
        self.use_shaped_reward = use_shaped_reward
        team_set = get_metamon_teams("gen1ou", "competitive")
        self.env = BattleAgainstBaseline(
            battle_format="gen1ou",
            observation_space=DefaultObservationSpace(),
            action_space=MinimalActionSpace(),
            reward_function=DefaultShapedReward() if use_shaped_reward else BinaryReward(),
            team_set=team_set,
            opponent_type=get_baseline(baseline_name),
            battle_backend="local",  # Use local backend to avoid authentication
        )
    
    def reset(self):
        obs, info = self.env.reset()
        return obs_to_vector(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs_to_vector(obs), reward, terminated, truncated, info
    
    @property
    def action_space(self):
        return self.env.action_space


# ==========================
#  Training Function
# ==========================
def train_rainbow_with_eval(
    num_episodes: int = 1000,
    eval_interval: int = 50,
    eval_episodes: int = 20,
    device: str = "cpu",
    save_dir: str = "rainbow_checkpoints",
):
    """Train Rainbow DQN with periodic evaluation"""
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training environment
    train_env = BattleEnvWrapper("Gen1BossAI", use_shaped_reward=True)
    
    # Get dimensions
    obs, _ = train_env.reset()
    obs_dim = len(obs)
    n_actions = train_env.action_space.n
    
    print(f"obs_dim = {obs_dim}, n_actions = {n_actions}")
    
    # Configuration
    cfg = RainbowConfig(
        hidden_dims=[1024, 512, 256],
        batch_size=64,
        buffer_capacity=100_000,
        min_buffer_size=5_000,
        lr=1e-4,
        use_double_dqn=True,
        use_dueling=True,
        use_noisy=True,
        use_per=True,
        use_n_step=True,
        n_step=3,
    )
    
    # Create agent
    agent = RainbowDQNAgent(obs_dim, n_actions, cfg, device=device)
    
    # Track metrics
    metrics = {
        'episode': [],
        'return': [],
        'length': [],
        'won': [],
        'loss': [],
        'eval_results': {},
    }
    
    # Training loop
    global_step = 0
    best_win_rate = 0
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        # Reset environment
        state, info = train_env.reset()
        done = False
        ep_return = 0
        ep_length = 0
        won = False
        
        while not done:
            # Act
            action = agent.act(state, eval_mode=False)
            next_state, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.push_transition((state, action, reward, next_state, done))
            
            # Update
            loss = agent.update()
            if loss is not None:
                metrics['loss'].append(loss)
            
            # Update target network
            if global_step % agent.cfg.target_update_interval == 0:
                agent.hard_update_target()
            
            # Update state
            state = next_state
            ep_return += reward
            ep_length += 1
            global_step += 1
        
        if done:
            won = bool(info.get("won", False))
        
        # Update learning rate
        if agent.scheduler:
            agent.scheduler.step()
        
        # Store episode metrics
        metrics['episode'].append(episode)
        metrics['return'].append(ep_return)
        metrics['length'].append(ep_length)
        metrics['won'].append(won)
        
        # Periodic evaluation
        if episode % eval_interval == 0 or episode == num_episodes - 1:
            print(f"\nEpisode {episode}: Evaluating...")
            eval_results = evaluate_agent_properly(agent, eval_episodes=eval_episodes, device=device)
            metrics['eval_results'][episode] = eval_results
            
            # Save model if improved
            current_win_rate = eval_results['Gen1BossAI']['win_rate']
            if current_win_rate > best_win_rate:
                best_win_rate = current_win_rate
                agent.save(f"{save_dir}/rainbow_best.pth")
                print(f"Saved new best model with win rate: {best_win_rate:.3f}")
            
            # Print progress
            print(f"  Return: {ep_return:.2f}, Won: {won}")
            print(f"  Eval vs Gen1BossAI: {current_win_rate:.3f}")
        
        # Save checkpoint
        if episode % 500 == 0:
            agent.save(f"{save_dir}/rainbow_episode_{episode}.pth")
            print(f"Saved checkpoint at episode {episode}")
    
    # Final save
    agent.save(f"{save_dir}/rainbow_final.pth")
    
    # Save metrics
    save_metrics(metrics, save_dir)
    
    # Plot training curves
    plot_training_curves(metrics, save_dir)
    
    return agent, metrics


def evaluate_agent_properly(agent: RainbowDQNAgent, eval_episodes: int = 20, device: str = "cpu") -> Dict:
    """Evaluate agent against all baselines with proper setup"""
    baselines = [
        "RandomBaseline",
        "PokeEnvHeuristic",
        "Gen1BossAI",
        "Grunt",
        "GymLeader",
        "EmeraldKaizo",
    ]
    
    results = {}
    
    for baseline in baselines:
        env_wrapper = BattleEnvWrapper(baseline, use_shaped_reward=False)
        wins = 0
        total_reward = 0
        battle_lengths = []
        
        for _ in range(eval_episodes):
            state, info = env_wrapper.reset()
            done = False
            ep_reward = 0
            steps = 0
            
            while not done:
                action = agent.act(state, eval_mode=True)
                next_state, reward, terminated, truncated, info = env_wrapper.step(action)
                done = terminated or truncated
                state = next_state
                ep_reward += reward
                steps += 1
            
            if info.get("won", False):
                wins += 1
            total_reward += ep_reward
            battle_lengths.append(steps)
        
        win_rate = wins / eval_episodes
        avg_reward = total_reward / eval_episodes
        avg_length = np.mean(battle_lengths)
        
        results[baseline] = {
            'win_rate': win_rate,
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'wins': wins,
            'total_battles': eval_episodes,
        }
    
    return results


def evaluate_rainbow_with_exact_format(
    model_path: str = "rainbow_checkpoints/rainbow_best.pth",
    num_battles: int = 100,
    device: str = "cpu"
):
    """Evaluate Rainbow DQN against all baselines with EXACT output format"""
    print("\n" + "="*60)
    print("RAINBOW DQN EVALUATION - EXACT FORMAT")
    print("="*60)
    
    baselines = [
        "RandomBaseline",
        "PokeEnvHeuristic",
        "Gen1BossAI",
        "Grunt",
        "GymLeader",
        "EmeraldKaizo",
    ]
    
    # Get environment dimensions
    env_wrapper = BattleEnvWrapper("RandomBaseline", use_shaped_reward=False)
    obs, _ = env_wrapper.reset()
    obs_dim = len(obs)
    n_actions = env_wrapper.action_space.n
    
    # Load agent
    try:
        checkpoint = torch.load(model_path, map_location=device)
        cfg = checkpoint['cfg']
    except:
        cfg = RainbowConfig()
    
    agent = RainbowDQNAgent(obs_dim, n_actions, cfg, device=device)
    agent.load(model_path)
    
    results = {}
    
    for baseline in baselines:
        print(f"\nEvaluating vs {baseline}...")
        env_wrapper = BattleEnvWrapper(baseline, use_shaped_reward=False)
        wins = 0
        returns = []
        
        for battle_idx in range(num_battles):
            state, info = env_wrapper.reset()
            done = False
            ep_return = 0
            
            while not done:
                action = agent.act(state, eval_mode=True)
                next_state, reward, terminated, truncated, info = env_wrapper.step(action)
                done = terminated or truncated
                state = next_state
                ep_return += reward
            
            if info.get("won", False):
                wins += 1
            returns.append(ep_return)
            
            # Progress update
            if (battle_idx + 1) % 20 == 0:
                current_win_rate = wins / (battle_idx + 1)
                print(f"  Battle {battle_idx + 1}/{num_battles} - Win rate: {current_win_rate:.3f}")
        
        win_rate = wins / num_battles
        avg_return = np.mean(returns)
        
        results[baseline] = {
            "win_rate": float(win_rate),
            "avg_return": float(avg_return),
            "wins": int(wins),
            "total_battles": num_battles,
        }
    
    # Print EXACT format as requested
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for baseline in baselines:
        if baseline in results:
            stats = results[baseline]
            win_rate_str = f"{stats['win_rate']:.3f}"
            return_str = f"{stats['avg_return']:7.2f}"
            print(f"{baseline:20s} | Win Rate = {win_rate_str}, Avg Return = {return_str}")
    
    print("="*60)
    
    # Overall statistics
    total_wins = sum(stats['wins'] for stats in results.values())
    total_battles = sum(stats['total_battles'] for stats in results.values())
    if total_battles > 0:
        overall_win_rate = total_wins / total_battles
        print(f"\nOverall Performance:")
        print(f"Total Battles: {total_battles}")
        print(f"Total Wins: {total_wins}")
        print(f"Overall Win Rate: {overall_win_rate:.3f}")
    
    # Save results
    with open("rainbow_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to rainbow_evaluation_results.json")
    
    # Save action distribution
    print(f"\nAction Distribution:")
    action_labels = ['Move 1', 'Move 2', 'Move 3', 'Move 4',
                     'Switch 1', 'Switch 2', 'Switch 3', 'Switch 4', 'Switch 5', 'Switch 6']
    total_actions = agent.eval_action_counts.sum()
    if total_actions > 0:
        for i, (label, count) in enumerate(zip(action_labels, agent.eval_action_counts)):
            freq = count / total_actions
            print(f"  {label:12s}: {freq:.4f} ({count} times)")
        
        # Save for Figure 3
        action_data = {
            'frequencies': (agent.eval_action_counts / total_actions).tolist(),
            'counts': agent.eval_action_counts.tolist(),
            'total_actions': int(total_actions),
        }
        with open("rainbow_action_distribution.json", "w") as f:
            json.dump(action_data, f, indent=2)
        print(f"\nAction distribution saved to rainbow_action_distribution.json")
    
    return results


def save_metrics(metrics: Dict, save_dir: str):
    """Save training metrics to files"""
    # Save episode metrics
    episode_df = pd.DataFrame({
        'episode': metrics['episode'],
        'return': metrics['return'],
        'length': metrics['length'],
        'won': metrics['won'],
    })
    episode_df.to_csv(f"{save_dir}/training_metrics.csv", index=False)
    
    # Save evaluation results
    eval_data = []
    for episode, results in metrics['eval_results'].items():
        for baseline, baseline_results in results.items():
            eval_data.append({
                'episode': episode,
                'baseline': baseline,
                'win_rate': baseline_results['win_rate'],
                'avg_reward': baseline_results['avg_reward'],
                'avg_length': baseline_results['avg_length'],
                'wins': baseline_results['wins'],
                'total_battles': baseline_results['total_battles'],
            })
    
    if eval_data:
        eval_df = pd.DataFrame(eval_data)
        eval_df.to_csv(f"{save_dir}/evaluation_metrics.csv", index=False)
    
    # Save loss history
    if metrics['loss']:
        loss_df = pd.DataFrame({'loss': metrics['loss']})
        loss_df.to_csv(f"{save_dir}/loss_history.csv", index=False)
    
    print(f"Metrics saved to {save_dir}")


def plot_training_curves(metrics: Dict, save_dir: str):
    """Plot training and evaluation metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Return over episodes
    axes[0, 0].plot(metrics['episode'], metrics['return'], alpha=0.6)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].set_title('Returns over Episodes')
    
    # Plot 2: Win rate (moving average)
    window = 20
    if len(metrics['won']) >= window:
        wins_moving_avg = pd.Series(metrics['won']).rolling(window=window).mean()
        axes[0, 1].plot(metrics['episode'][:len(wins_moving_avg)], wins_moving_avg)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Win Rate')
        axes[0, 1].set_title(f'Win Rate (MA{window})')
    
    # Plot 3: Loss
    if metrics['loss']:
        axes[0, 2].plot(metrics['loss'])
        axes[0, 2].set_xlabel('Update Step')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].set_title('Training Loss')
    
    # Plot 4: Evaluation win rates
    if metrics['eval_results']:
        episodes = sorted(metrics['eval_results'].keys())
        baselines = list(next(iter(metrics['eval_results'].values())).keys())
        for baseline in baselines[:3]:
            win_rates = [metrics['eval_results'][ep][baseline]['win_rate'] for ep in episodes]
            axes[1, 0].plot(episodes, win_rates, label=baseline)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Win Rate')
        axes[1, 0].set_title('Evaluation Win Rates')
        axes[1, 0].legend()
    
    # Plot 5: Episode lengths
    axes[1, 1].plot(metrics['episode'], metrics['length'], alpha=0.6)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Length')
    axes[1, 1].set_title('Episode Lengths')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_dir}/training_curves.png")


# ==========================
#  Main Execution
# ==========================
if __name__ == "__main__":
    # Check for --eval flag
    if "--eval" in sys.argv:
        # Determine device
        if torch.backends.mps.is_available():
            device = "mps"
            print("Using MPS (Apple Silicon) acceleration")
        elif torch.cuda.is_available():
            device = "cuda"
            print("Using CUDA acceleration")
        else:
            device = "cpu"
            print("Using CPU")
        
        # Default model path
        model_path = "rainbow_checkpoints/rainbow_best.pth"
        
        # Check if custom model path is provided
        if len(sys.argv) > 2 and sys.argv[2].endswith(".pth"):
            model_path = sys.argv[2]
            print(f"Using custom model: {model_path}")
        
        # Run evaluation with exact format
        evaluate_rainbow_with_exact_format(
            model_path=model_path,
            num_battles=100,
            device=device
        )
        sys.exit(0)
    
    # Parse arguments for training
    import argparse
    parser = argparse.ArgumentParser(description="Train or evaluate Rainbow DQN agent")
    parser.add_argument("--mode", type=str, default="train", 
                        choices=["train", "evaluate"],
                        help="Mode: train or evaluate")
    parser.add_argument("--model_path", type=str, default="rainbow_checkpoints/rainbow_best.pth",
                        help="Path to model checkpoint (for evaluate mode)")
    parser.add_argument("--num_episodes", type=int, default=1000,
                        help="Number of training episodes")
    parser.add_argument("--eval_episodes", type=int, default=20,
                        help="Number of evaluation episodes per baseline")
    parser.add_argument("--device", type=str, 
                        default="mps" if torch.backends.mps.is_available() else "cpu",
                        help="Device to use (mps, cuda, or cpu)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print("Starting Rainbow DQN training...")
        agent, metrics = train_rainbow_with_eval(
            num_episodes=args.num_episodes,
            device=args.device,
            save_dir="rainbow_checkpoints"
        )
    elif args.mode == "evaluate":
        print(f"Evaluating model from {args.model_path}...")
        evaluate_rainbow_with_exact_format(
            model_path=args.model_path,
            num_battles=args.eval_episodes,
            device=args.device
        )
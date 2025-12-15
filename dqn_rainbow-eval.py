"""
Script to extract action distribution data from trained Rainbow DQN
and generate Figure 3 for the paper.
"""

import json
import numpy as np
import torch
from rainbow_dqn import RainbowAgent, RainbowConfig, BattleEnvWrapper, obs_to_vector

def extract_action_distribution(
    model_path: str = "rainbow_checkpoints/rainbow_best.pth",
    num_battles: int = 100,
    baseline: str = "Gen1BossAI",
    device: str = "cpu"
):
    """
    Extract action distribution from a trained Rainbow DQN model.
    
    Args:
        model_path: Path to trained model
        num_battles: Number of battles to run for statistics
        baseline: Opponent to evaluate against
        device: Device to run on
    
    Returns:
        action_frequencies: Array of 10 frequencies (4 moves + 6 switches)
    """
    
    # Setup environment to get dimensions
    env = BattleEnvWrapper(baseline)
    obs, _ = env.reset()
    obs_dim = len(obs)
    n_actions = env.action_space.n
    
    print(f"Loading model from {model_path}")
    print(f"obs_dim={obs_dim}, n_actions={n_actions}")
    
    # Load agent
    cfg = RainbowConfig()
    agent = RainbowAgent(obs_dim, n_actions, cfg, device=device)
    agent.load(model_path)
    
    # Reset action tracking
    agent.reset_action_counts()
    
    print(f"\nRunning {num_battles} battles against {baseline}...")
    wins = 0
    
    for battle in range(num_battles):
        state, info = env.reset()
        done = False
        
        while not done:
            action = agent.act(state, eval_mode=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
        
        if info.get("won", False):
            wins += 1
        
        if (battle + 1) % 20 == 0:
            print(f"  Completed {battle + 1}/{num_battles} battles, Win rate: {wins/(battle+1):.3f}")
    
    # Get action distribution
    action_counts = agent.eval_action_counts
    total_actions = action_counts.sum()
    action_frequencies = action_counts / total_actions
    
    print(f"\n{'='*60}")
    print(f"RESULTS vs {baseline}")
    print(f"{'='*60}")
    print(f"Total battles: {num_battles}")
    print(f"Wins: {wins}")
    print(f"Win rate: {wins/num_battles:.3f}")
    print(f"Total actions taken: {total_actions}")
    
    print(f"\n{'='*60}")
    print(f"ACTION DISTRIBUTION")
    print(f"{'='*60}")
    
    action_labels = ['Move 1', 'Move 2', 'Move 3', 'Move 4',
                     'Switch 1', 'Switch 2', 'Switch 3', 'Switch 4', 'Switch 5', 'Switch 6']
    
    for i, (label, count, freq) in enumerate(zip(action_labels, action_counts, action_frequencies)):
        action_type = "Move" if i < 4 else "Switch"
        print(f"{label:12s} | Count: {count:6d} | Frequency: {freq:.4f} | Type: {action_type}")
    
    # Save to file
    output_data = {
        'action_counts': action_counts.tolist(),
        'action_frequencies': action_frequencies.tolist(),
        'action_labels': action_labels,
        'total_actions': int(total_actions),
        'num_battles': num_battles,
        'wins': wins,
        'win_rate': wins / num_battles,
        'baseline': baseline,
    }
    
    output_file = "figure3_action_data.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Data saved to {output_file}")
    print(f"{'='*60}")
    
    # Print Python code to use in figure generation
    print(f"\n{'='*60}")
    print(f"COPY THIS INTO YOUR FIGURE GENERATION CODE:")
    print(f"{'='*60}")
    print(f"frequencies = np.array({action_frequencies.tolist()})")
    print(f"{'='*60}")
    
    return action_frequencies


def generate_figure3_with_real_data(action_frequencies: np.ndarray, save_path: str = "figure3_real.png"):
    """Generate Figure 3 with real action distribution data"""
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    
    # Set publication-quality defaults
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 10
    
    fig, ax = plt.subplots(figsize=(3.5, 3))
    
    action_labels = ['Move 1', 'Move 2', 'Move 3', 'Move 4',
                     'Switch 1', 'Switch 2', 'Switch 3', 'Switch 4', 'Switch 5', 'Switch 6']
    
    colors = ['#d62728' if 'Move' in a else '#9467bd' for a in action_labels]
    
    bars = ax.barh(action_labels, action_frequencies, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Selection Frequency')
    ax.set_title('Action Distribution for Rainbow DQN')
    ax.set_xlim(0, max(action_frequencies) * 1.15)
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    
    # Add value labels
    for i, (bar, freq) in enumerate(zip(bars, action_frequencies)):
        ax.text(freq + 0.005, i, f'{freq:.3f}', va='center', fontsize=8)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', alpha=0.8, label='Moves'),
        Patch(facecolor='#9467bd', alpha=0.8, label='Switches')
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {save_path} and PDF version")
    plt.close()


if __name__ == "__main__":
    import sys
    
    # Determine device
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Default parameters
    model_path = "rainbow_checkpoints/rainbow_best.pth"
    num_battles = 100
    baseline = "Gen1BossAI"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    if len(sys.argv) > 2:
        num_battles = int(sys.argv[2])
    if len(sys.argv) > 3:
        baseline = sys.argv[3]
    
    # Extract action distribution
    action_frequencies = extract_action_distribution(
        model_path=model_path,
        num_battles=num_battles,
        baseline=baseline,
        device=device
    )
    
    # Generate figure
    generate_figure3_with_real_data(action_frequencies)
    
    print("\n Done! Use the frequencies array printed above in your figure generation code.")
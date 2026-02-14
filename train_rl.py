"""
TRAINING SCRIPT FOR RL WORDLE SOLVER
=====================================
Train the DQN agent and track performance over time.
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from game_engine import load_word_lists
from pattern_matrix import load_pattern_matrix
from rl_solver import RLSolver


def train_rl_agent(
    solver: RLSolver,
    training_words: list,
    n_episodes: int = 1000,
    eval_interval: int = 100,
    eval_size: int = 50,
    save_path: str = None
):
    """
    Train the RL agent.
    
    Args:
        solver: RLSolver instance
        training_words: List of words for training
        n_episodes: Number of training episodes
        eval_interval: Evaluate every N episodes
        eval_size: Number of words to use for evaluation
        save_path: Path to save model checkpoints
    
    Returns:
        Training history dictionary
    """
    
    history = {
        'episode': [],
        'reward': [],
        'guesses': [],
        'win_rate': [],
        'loss': [],
        'epsilon': [],
        'eval_avg_guesses': [],
        'eval_win_rate': []
    }
    
    print("="*70)
    print("TRAINING RL WORDLE SOLVER")
    print("="*70)
    print(f"Episodes: {n_episodes}")
    print(f"Training words: {len(training_words)}")
    print(f"Eval interval: {eval_interval}")
    print(f"Device: {solver.device}")
    print("="*70)
    
    # Evaluation words (separate from training)
    eval_words = random.sample(training_words, eval_size)
    
    # Training loop
    for episode in tqdm(range(n_episodes), desc="Training"):
        # Sample random word
        target_word = random.choice(training_words)
        
        # Train on this episode
        metrics = solver.train_episode(target_word)
        
        # Record metrics
        history['episode'].append(episode)
        history['reward'].append(metrics['reward'])
        history['guesses'].append(metrics['guesses'])
        history['win_rate'].append(1.0 if metrics['won'] else 0.0)
        history['loss'].append(metrics['loss'])
        history['epsilon'].append(metrics['epsilon'])
        
        # Evaluate periodically
        if (episode + 1) % eval_interval == 0:
            eval_results = evaluate_agent(solver, eval_words, verbose=False)
            history['eval_avg_guesses'].append(eval_results['avg_guesses'])
            history['eval_win_rate'].append(eval_results['win_rate'])
            
            # Print progress
            recent_rewards = np.mean(history['reward'][-eval_interval:])
            recent_guesses = np.mean(history['guesses'][-eval_interval:])
            recent_wins = np.mean(history['win_rate'][-eval_interval:])
            
            print(f"\n{'='*70}")
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"Recent (last {eval_interval}):")
            print(f"  Avg Reward: {recent_rewards:.2f}")
            print(f"  Avg Guesses: {recent_guesses:.2f}")
            print(f"  Win Rate: {recent_wins:.1%}")
            print(f"  Epsilon: {metrics['epsilon']:.3f}")
            print(f"Evaluation:")
            print(f"  Avg Guesses: {eval_results['avg_guesses']:.2f}")
            print(f"  Win Rate: {eval_results['win_rate']:.1%}")
            print(f"  Median Guesses: {eval_results['median_guesses']:.0f}")
            print(f"{'='*70}")
            
            # Save checkpoint
            if save_path:
                checkpoint_path = f"{save_path}_ep{episode+1}.pt"
                solver.save_model(checkpoint_path)
        else:
            # Pad evaluation history for non-eval episodes
            if len(history['eval_avg_guesses']) > 0:
                history['eval_avg_guesses'].append(history['eval_avg_guesses'][-1])
                history['eval_win_rate'].append(history['eval_win_rate'][-1])
            else:
                history['eval_avg_guesses'].append(0)
                history['eval_win_rate'].append(0)
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    final_eval = evaluate_agent(solver, eval_words, verbose=True)
    
    # Save final model
    if save_path:
        import os
        save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
        os.makedirs(save_dir, exist_ok=True)
        final_path = f"{save_path}_final.pt"
        solver.save_model(final_path)
    return history


def evaluate_agent(solver: RLSolver, test_words: list, verbose: bool = False):
    """
    Evaluate the RL agent on a set of test words.
    
    Args:
        solver: RLSolver instance
        test_words: List of words to test on
        verbose: If True, print detailed results
    
    Returns:
        Dictionary with evaluation metrics
    """
    original_training_mode = solver.training_mode
    solver.training_mode = False  # Disable exploration
    
    results = {
        'wins': 0,
        'losses': 0,
        'guess_counts': [],
        'failed_words': []
    }
    
    iterator = tqdm(test_words, desc="Evaluating") if verbose else test_words
    
    for word in iterator:
        guesses, won = solver.solve(word, verbose=False)
        
        if won:
            results['wins'] += 1
            results['guess_counts'].append(len(guesses))
        else:
            results['losses'] += 1
            results['guess_counts'].append(7)  # Penalty
            results['failed_words'].append(word)
    
    # Calculate metrics
    total = len(test_words)
    avg_guesses = np.mean(results['guess_counts'])
    median_guesses = np.median(results['guess_counts'])
    win_rate = results['wins'] / total
    
    solver.training_mode = original_training_mode
    
    eval_results = {
        'win_rate': win_rate,
        'avg_guesses': avg_guesses,
        'median_guesses': median_guesses,
        'wins': results['wins'],
        'losses': results['losses'],
        'failed_words': results['failed_words']
    }
    
    if verbose:
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Avg Guesses: {avg_guesses:.2f}")
        print(f"Median Guesses: {median_guesses:.0f}")
        print(f"Wins: {results['wins']}/{total}")
        print(f"Losses: {results['losses']}")
        if results['failed_words']:
            print(f"Failed words: {', '.join(results['failed_words'][:10])}")
    
    return eval_results


def plot_training_history(history: dict, save_path: str = None):
    """
    Plot training metrics over time.
    
    Args:
        history: Training history dictionary
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('RL Agent Training Progress', fontsize=16, fontweight='bold')
    
    episodes = history['episode']
    
    # Plot 1: Reward over time
    ax = axes[0, 0]
    ax.plot(episodes, history['reward'], alpha=0.3, label='Episode Reward')
    # Moving average
    window = 50
    if len(history['reward']) >= window:
        ma_reward = pd.Series(history['reward']).rolling(window).mean()
        ax.plot(episodes, ma_reward, linewidth=2, label=f'{window}-Episode MA')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Guesses over time
    ax = axes[0, 1]
    ax.plot(episodes, history['guesses'], alpha=0.3, label='Episode Guesses')
    if len(history['guesses']) >= window:
        ma_guesses = pd.Series(history['guesses']).rolling(window).mean()
        ax.plot(episodes, ma_guesses, linewidth=2, label=f'{window}-Episode MA')
    ax.axhline(y=3.6, color='r', linestyle='--', label='Entropy Baseline (3.6)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Number of Guesses')
    ax.set_title('Guesses per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Win rate over time
    ax = axes[1, 0]
    if len(history['win_rate']) >= window:
        ma_winrate = pd.Series(history['win_rate']).rolling(window).mean()
        ax.plot(episodes, ma_winrate, linewidth=2, label=f'{window}-Episode MA Win Rate')
    ax.axhline(y=0.995, color='r', linestyle='--', label='Entropy Baseline (99.5%)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Win Rate')
    ax.set_title('Win Rate (Moving Average)')
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Evaluation metrics
    ax = axes[1, 1]
    # Filter out zero values (non-eval episodes)
    eval_episodes = [e for e, v in zip(episodes, history['eval_avg_guesses']) if v > 0]
    eval_guesses = [v for v in history['eval_avg_guesses'] if v > 0]
    eval_winrate = [v for v in history['eval_win_rate'] if v > 0]
    
    if eval_episodes:
        ax.plot(eval_episodes, eval_guesses, 'o-', linewidth=2, label='Eval Avg Guesses')
        ax2 = ax.twinx()
        ax2.plot(eval_episodes, eval_winrate, 's-', color='green', 
                linewidth=2, label='Eval Win Rate')
        ax2.set_ylabel('Win Rate', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_ylim([0, 1.05])
    
    ax.axhline(y=3.6, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Guesses')
    ax.set_title('Evaluation Performance')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RL Wordle Solver')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--eval-interval', type=int, default=100, help='Evaluation interval')
    parser.add_argument('--eval-size', type=int, default=50, help='Evaluation set size')
    parser.add_argument('--save-path', type=str, default='models/rl_wordle', help='Model save path')
    parser.add_argument('--plot', action='store_true', help='Plot training history')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load data
    print("Loading word lists and pattern matrix...")
    list_a, list_b = load_word_lists()
    answers = list_a if len(list_a) < len(list_b) else list_b
    guesses = list_a if len(list_a) > len(list_b) else list_b
    patterns = load_pattern_matrix("pattern_matrix.npy")
    
    print(f"Loaded {len(answers)} answer words, {len(guesses)} guess words")
    
    # Create solver
    print("Initializing RL solver...")
    solver = RLSolver(
        answers, 
        guesses, 
        pattern_matrix=patterns,
        use_answer_list=False
    )
    
    # Train
    history = train_rl_agent(
        solver,
        training_words=answers,  # Train on valid answers
        n_episodes=args.episodes,
        eval_interval=args.eval_interval,
        eval_size=args.eval_size,
        save_path=args.save_path
    )
    
    # Save history
    history_df = pd.DataFrame(history)
    history_path = f"{args.save_path}_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"\nTraining history saved to {history_path}")
    
    # Plot if requested
    if args.plot:
        plot_path = f"{args.save_path}_training.png"
        plot_training_history(history, save_path=plot_path)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)

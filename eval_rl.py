"""
COMPREHENSIVE EVALUATION SCRIPT
================================
Compare all solvers: Naive, Entropy, and RL-DQN

Enhanced version of eval.py with RL support
"""

import time
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
from game_engine import load_word_lists
from pattern_matrix import load_pattern_matrix 
from solvers import NaiveSolver, EntropySolver
from rl_solver import RLSolver
from enum import Enum


class EvalMode(Enum):
    BLIND = "blind"
    STANDARD = "standard"
    RESTRICTED = "restricted"


def resolve_mode(mode: EvalMode, list_a, list_b):
    """Determine answer and guess lists based on evaluation mode."""
    guesses = list_a if len(list_a) > len(list_b) else list_b
    real_answers = list_a if len(list_a) < len(list_b) else list_b
    
    if mode == EvalMode.BLIND:
        return (
            guesses,
            guesses,
            False,
            "Blind Mode - Bigger List Used"
        )
    
    if mode == EvalMode.STANDARD:
        return (
            real_answers,
            guesses,
            False,
            "Standard Mode"
        )
    
    if mode == EvalMode.RESTRICTED:
        return (
            real_answers,
            real_answers,
            True,
            "Restricted Mode"
        )
    
    raise ValueError(f"Unknown mode: {mode}")


def evaluate_solver(solver, test_words: List[str], verbose: bool = False) -> Dict[str, Any]:
    """
    Standardized benchmark loop for fair comparison.
    
    Args:
        solver: Any solver instance (Naive, Entropy, or RL)
        test_words: List of words to test on
        verbose: Show progress bar
    
    Returns:
        Dictionary with evaluation metrics
    """
    results = {
        'solver_name': solver.name,
        'total_words': len(test_words),
        'wins': 0,
        'losses': 0,
        'guess_counts': [],
        'solve_times': [],
        'failed_words': []
    }
    
    iterator = tqdm(test_words, desc=f"Eval {solver.name}") if verbose else test_words
    
    for word in iterator:
        start_time = time.time()
        
        try:
            guesses, won = solver.solve(word)
        except Exception as e:
            print(f"\n❌ Error solving {word}: {e}")
            won = False
            guesses = []
        
        elapsed = time.time() - start_time
        results['solve_times'].append(elapsed)
        
        if won:
            results['wins'] += 1
            results['guess_counts'].append(len(guesses))
        else:
            results['losses'] += 1
            results['guess_counts'].append(7)  # Penalty
            results['failed_words'].append(word)
    
    # Calculate aggregate metrics
    avg_guesses = sum(results['guess_counts']) / len(results['guess_counts']) if results['guess_counts'] else 0
    median_guesses = np.median(results['guess_counts']) if results['guess_counts'] else 0
    avg_time = sum(results['solve_times']) / len(results['solve_times']) if results['solve_times'] else 0
    
    return {
        'solver_name': solver.name,
        'win_rate': results['wins'] / len(test_words),
        'avg_guesses': avg_guesses,
        'median_guesses': median_guesses,
        'avg_time': avg_time,
        'losses': results['losses'],
        'failed_words': results['failed_words']
    }


def run_benchmark(
    mode: EvalMode = EvalMode.BLIND,
    n_tests: int = 100,
    seed: int = 42,
    verbose: bool = True,
    include_rl: bool = True,
    rl_model_path: str = 'models\rl_wordle_final.pt'
):
    """
    Run comprehensive benchmark comparing all solvers.
    
    Args:
        mode: Evaluation mode (blind, standard, restricted)
        n_tests: Number of test words
        seed: Random seed
        verbose: Show progress
        include_rl: Include RL solver in comparison
        rl_model_path: Path to trained RL model (if None, uses untrained)
    
    Returns:
        DataFrame with results
    """
    random.seed(seed)
    
    print("=" * 70)
    print("WORDLE SOLVER BENCHMARK")
    print("=" * 70)
    
    # Load word lists
    list_a, list_b = load_word_lists()
    answers, guesses, use_answer_list, mode_desc = resolve_mode(mode, list_a, list_b)
    
    effective_guesses = answers if use_answer_list else guesses
    
    print(f"Mode: {mode_desc}")
    print(f"Answers: {len(answers)} | Guesses: {len(effective_guesses)}")
    print(f"use_answer_list = {use_answer_list}")
    
    # Load pattern matrix
    patterns = load_pattern_matrix("pattern_matrix.npy")
    if patterns is None:
        raise RuntimeError("pattern_matrix.npy not found")
    
    # Initialize solvers
    solvers = []
    
    # 1. Naive solver
    naive = NaiveSolver(
        answer_list=answers,
        guess_list=effective_guesses,
        pattern_matrix=patterns,
        use_answer_list=use_answer_list
    )
    solvers.append(naive)
    
    # 2. Entropy solver
    entropy = EntropySolver(
        answer_list=answers,
        guess_list=effective_guesses,
        pattern_matrix=patterns,
        word_freq_path="word_frequency.csv",
        use_answer_list=use_answer_list
    )
    solvers.append(entropy)
    
    # 3. RL solver (if enabled)
    if include_rl:
        print("\nInitializing RL solver...")
        rl_solver = RLSolver(
            answer_list=answers,
            guess_list=effective_guesses,
            pattern_matrix=patterns,
            use_answer_list=use_answer_list
        )
        
        # Load trained model if provided
        if rl_model_path:
            try:
                rl_solver.load_model(rl_model_path)
                print(f"✅ Loaded trained RL model from {rl_model_path}")
            except Exception as e:
                print(f"⚠️  Could not load RL model: {e}")
                print("Using untrained RL model")
        else:
            print("⚠️  No RL model path provided - using untrained model")
        
        solvers.append(rl_solver)
    
    # Test set (always use real solutions)
    real_solutions, _ = load_word_lists("valid_solutions.csv", "all_words.csv")
    test_words = random.sample(real_solutions, min(n_tests, len(real_solutions)))
    
    print(f"\nTest set: {len(test_words)} words")
    print("=" * 70)
    
    # Run evaluation
    rows = []
    for solver in solvers:
        print(f"\nEvaluating {solver.name}...")
        metrics = evaluate_solver(solver, test_words, verbose=verbose)
        
        rows.append({
            "Solver": solver.name,
            "Mode": mode.value,
            "Win Rate": f"{metrics['win_rate']:.1%}",
            "Avg Guesses": f"{metrics['avg_guesses']:.2f}",
            "Median Guesses": f"{metrics['median_guesses']:.0f}",
            "Avg Time (s)": f"{metrics['avg_time']:.4f}",
            "Losses": metrics['losses']
        })
        
        # Print detailed results
        print(f"\n{solver.name} Results:")
        print(f"  Win Rate: {metrics['win_rate']:.1%}")
        print(f"  Avg Guesses: {metrics['avg_guesses']:.2f}")
        print(f"  Median Guesses: {metrics['median_guesses']:.0f}")
        print(f"  Avg Time: {metrics['avg_time']:.4f}s")
        print(f"  Losses: {metrics['losses']}")
        if metrics['failed_words']:
            print(f"  Failed words: {', '.join(metrics['failed_words'][:5])}")
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    
    return pd.DataFrame(rows)


def detailed_comparison(
    test_words: List[str],
    patterns: np.ndarray,
    answers: List[str],
    guesses: List[str],
    rl_model_path: str = None
):
    """
    Detailed word-by-word comparison of all solvers.
    
    Returns:
        DataFrame with per-word results
    """
    print("Running detailed comparison...")
    
    # Initialize solvers
    naive = NaiveSolver(answers, guesses, patterns, use_answer_list=False)
    entropy = EntropySolver(answers, guesses, patterns, 
                           word_freq_path="word_frequency.csv", 
                           use_answer_list=False)
    rl_solver = RLSolver(answers, guesses, patterns, use_answer_list=False)
    
    if rl_model_path:
        try:
            rl_solver.load_model(rl_model_path)
        except:
            print("⚠️  Using untrained RL model")
    
    results = []
    
    for word in tqdm(test_words, desc="Comparing"):
        row = {'word': word}
        
        # Test each solver
        for solver, name in [(naive, 'naive'), (entropy, 'entropy'), (rl_solver, 'rl')]:
            guesses, won = solver.solve(word)
            row[f'{name}_guesses'] = len(guesses) if won else 7
            row[f'{name}_won'] = won
        
        results.append(row)
    
    df = pd.DataFrame(results)
    
    # Add comparison columns
    df['entropy_vs_naive'] = df['naive_guesses'] - df['entropy_guesses']
    df['rl_vs_entropy'] = df['entropy_guesses'] - df['rl_guesses']
    df['rl_vs_naive'] = df['naive_guesses'] - df['rl_guesses']
    
    return df


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Wordle Solvers')
    parser.add_argument('--mode', type=str, default='blind', 
                       choices=['blind', 'standard', 'restricted'],
                       help='Evaluation mode')
    parser.add_argument('--n-tests', type=int, default=100, 
                       help='Number of test words')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--include-rl', action='store_true', 
                       help='Include RL solver')
    parser.add_argument('--rl-model', type=str, default=None,
                       help='Path to trained RL model')
    parser.add_argument('--detailed', action='store_true',
                       help='Run detailed word-by-word comparison')
    parser.add_argument('--save', type=str, default=None,
                       help='Save results to CSV')
    
    args = parser.parse_args()
    
    # Convert mode string to enum
    mode_map = {
        'blind': EvalMode.BLIND,
        'standard': EvalMode.STANDARD,
        'restricted': EvalMode.RESTRICTED
    }
    mode = mode_map[args.mode]
    
    # Run benchmark
    df = run_benchmark(
        mode=mode,
        n_tests=args.n_tests,
        seed=args.seed,
        verbose=True,
        include_rl=args.include_rl,
        rl_model_path=args.rl_model
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))
    
    # Save if requested
    if args.save:
        df.to_csv(args.save, index=False)
        print(f"\nResults saved to {args.save}")
    
    # Detailed comparison if requested
    if args.detailed:
        print("\nRunning detailed comparison...")
        list_a, list_b = load_word_lists()
        answers = list_a if len(list_a) < len(list_b) else list_b
        guesses = list_a if len(list_a) > len(list_b) else list_b
        patterns = load_pattern_matrix("pattern_matrix.npy")
        
        real_solutions, _ = load_word_lists("valid_solutions.csv", "all_words.csv")
        test_words = random.sample(real_solutions, min(args.n_tests, len(real_solutions)))
        
        detailed_df = detailed_comparison(test_words, patterns, answers, guesses, args.rl_model)
        
        print("\nDetailed Statistics:")
        print(f"  Entropy beats Naive: {(detailed_df['entropy_vs_naive'] > 0).sum()} times")
        print(f"  RL beats Entropy: {(detailed_df['rl_vs_entropy'] > 0).sum()} times")
        print(f"  RL beats Naive: {(detailed_df['rl_vs_naive'] > 0).sum()} times")
        
        if args.save:
            detailed_path = args.save.replace('.csv', '_detailed.csv')
            detailed_df.to_csv(detailed_path, index=False)
            print(f"  Detailed results saved to {detailed_path}")

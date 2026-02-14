import time
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
from game_engine import load_word_lists
from pattern_matrix import load_pattern_matrix 
from solvers import NaiveSolver, EntropySolver
from enum import Enum

class EvalMode(Enum):
    BLIND = "blind"
    STANDARD= "standard"
    RESTRICTED= "restricted"

def resolve_mode(mode: EvalMode, list_a, list_b):
    guesses = list_a if len(list_a) > len(list_b) else list_b
    real_answers = list_a if len(list_a) < len(list_b) else list_b
    if mode== EvalMode.BLIND:
        return(
            guesses,
            guesses,
            False,
            "Blind Mode, Bigger List Used"
        )
    
    if mode== EvalMode.STANDARD:
        return(
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
            "RESTRICTED MODE"
    )
    raise ValueError(f"Unknown mode: {mode}")

def evaluate_solver(solver, test_words: List[str], verbose: bool = False) -> Dict[str, Any]:
    """
    Standardized benchmark loop to ensure fair comparison between solvers.
    Captures time, guess count, and failure rates.
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

    # Helper for the console output so I can track progress during long runs
    iterator = tqdm(test_words, desc=f"Eval {solver.name}") if verbose else test_words

    for word in iterator:
        start_time = time.time()
        
        try:
            # The core solve method returns the path taken
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
            # Penalize failures with 7 guesses so they affect the average score
            results['guess_counts'].append(7) 
            results['failed_words'].append(word)

    # Calculate aggregate metrics for the report
    avg_guesses = sum(results['guess_counts']) / len(results['guess_counts']) if results['guess_counts'] else 0
    avg_time = sum(results['solve_times']) / len(results['solve_times']) if results['solve_times'] else 0
    
    return {
        'win_rate': results['wins'] / len(test_words),
        'avg_guesses': avg_guesses,
        'avg_time': avg_time,
        'losses': results['losses']
    }

def run_benchmark(
mode: EvalMode = EvalMode.BLIND,
n_tests: int = 100,
seed: int = 42,
verbose: bool = True
):
    random.seed(seed)
    print("=" * 60)
    print("WORDLE BOT BENCHMARK")
    print("=" * 60)
    # Load word lists
    list_a, list_b = load_word_lists()
    answers, guesses, use_answer_list, mode_desc = resolve_mode(mode, list_a, list_b)
    effective_guesses = len(answers) if use_answer_list else len(guesses)
    print(f"Mode: {mode_desc}")
    print(f"Answers: {len(answers)} | Guesses: {len(effective_guesses)}")
    print(f"use_answer_list = {use_answer_list}")
    # Load pattern matrix
    patterns = load_pattern_matrix("pattern_matrix.npy")
    if patterns is None:
        raise RuntimeError("pattern_matrix.npy not found")
    # Initialize solvers
    naive = NaiveSolver(
        answer_list=answers,
        guess_list=effective_guesses,
        pattern_matrix=patterns,
        use_answer_list=use_answer_list
    )
    entropy = EntropySolver(
        answer_list=answers,
        guess_list=effective_guesses,
        pattern_matrix=patterns,
        word_freq_path="word_frequency.csv",
        use_answer_list=use_answer_list
    )   
    # Test set (always real solutions)
    real_solutions, _ = load_word_lists("valid_solutions.csv", "all_words.csv")
    test_words = random.sample(real_solutions, n_tests)
    # Run evaluation
    rows = []
    for solver in [naive, entropy]:
        metrics = evaluate_solver(solver, test_words, verbose=verbose)
        rows.append({
            "Solver": solver.name,
            "Mode": mode.value,
            "Win Rate": metrics["win_rate"],
            "Avg Guesses": metrics["avg_guesses"],
            "Avg Time (s)": metrics["avg_time"],
            "Losses": metrics["losses"]
        })
    return pd.DataFrame(rows)

df=run_benchmark()
print(df)
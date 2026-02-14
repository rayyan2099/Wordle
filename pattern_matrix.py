import random
import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import Counter
from game_engine import get_pattern, filter_words, load_word_lists

"""
PRECOMPUTATION MODULE
=====================
Precompute all possible patterns to massively speed up entropy calculation.
This creates a matrix where:
  patterns[i, j] = pattern when guessing word_i against answer_j

This is computed ONCE and reused by all solvers.
"""

def precompute_pattern_matrix(guess_list: List[str], answer_list: List[str], 
                              verbose: bool = True) -> np.ndarray:
    """

    Returns:
        patterns[i, j] = integer pattern for guess i against answer j
        Pattern encoding: convert "01202" to integer using base-3
    """
    n_guesses = len(guess_list)
    n_answers = len(answer_list)
    
    # Use int16 to save memory (patterns are 0-242, well within int16 range)
    patterns = np.zeros((n_guesses, n_answers), dtype=np.int16)
    
    if verbose:
        print(f"Precomputing {n_guesses} × {n_answers} = {n_guesses * n_answers:,} patterns...")
        print("This will take ~5-10 minutes but only needs to run once!")
    
    # Progress tracking
    progress_interval = max(1, n_guesses // 20)  # Update 20 times
    
    for i, guess in enumerate(guess_list):
        if verbose and i % progress_interval == 0:
            pct = (i / n_guesses) * 100
            print(f"Progress: {pct:.1f}% ({i}/{n_guesses})", end='\r')
        
        for j, answer in enumerate(answer_list):
            pattern_str = get_pattern(guess, answer)
            # Convert "01202" to integer: 0×3^4 + 1×3^3 + 2×3^2 + 0×3^1 + 2×3^0
            pattern_int = int(pattern_str, 3)  # Base-3 to integer
            patterns[i, j] = pattern_int
    
    if verbose:
        print(f"\n✅ Precomputation complete!")
        print(f"   Matrix size: {patterns.nbytes / 1024 / 1024:.1f} MB")
    
    return patterns


def save_pattern_matrix(patterns: np.ndarray, filename: str = "pattern_matrix.npy"):
    """Save precomputed matrix to disk."""
    np.save(filename, patterns)
    print(f"✅ Saved pattern matrix to {filename}")


def load_pattern_matrix(filename: str = "pattern_matrix.npy") -> Optional[np.ndarray]:
    """Load precomputed matrix from disk."""
    try:
        patterns = np.load(filename)
        print(f"✅ Loaded pattern matrix from {filename}")
        return patterns
    except FileNotFoundError:
        print(f"⚠️  Pattern matrix not found: {filename}")
        return None


def pattern_int_to_str(pattern_int: int) -> str:
    """Convert integer pattern back to string."""
    # Convert base-10 integer to base-3 string
    if pattern_int == 0:
        return "00000"
    
    result = []
    temp = pattern_int
    for _ in range(5):
        result.append(str(temp % 3))
        temp //= 3
    
    return ''.join(reversed(result))


if __name__ == "__main__":
    # Load lists
    list_a, list_b = load_word_lists()
    if len(list_a) < len(list_b):
        answers, guesses = list_a, list_b
    else:
        answers, guesses = list_b, list_a

    # Precompute (takes 5-10 minutes)
    print("Starting precomputation...")
    patterns = precompute_pattern_matrix(guesses, guesses, verbose=True)

    # Save to disk
    save_pattern_matrix(patterns, "pattern_matrix.npy")
    print("Done!")
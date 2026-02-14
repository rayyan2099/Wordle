import random
import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import Counter
from game_engine import get_pattern, filter_words, load_word_lists
from pattern_matrix import pattern_int_to_str  # Import helper function

"""
WORDLE SOLVERS
==============
All solvers use the precomputed pattern matrix from pattern_matrix.py for:
1. Fair comparison (same data access)
2. Massive speed improvement (60x faster!)
"""

"""BASE SOLVER CLASS"""

class BaseSolver:
    def __init__(self, answer_list: List[str], guess_list: List[str],
                 pattern_matrix: Optional[np.ndarray] = None):
        """
        Initialize solver.
        
        Args:
            answer_list: List of possible answers (2,315 words)
            guess_list: List of valid guesses (12,972 words)
            pattern_matrix: Precomputed pattern matrix from pattern_matrix.py
        """
        self.answer_list = answer_list
        self.guess_list = guess_list
        self.name = "Base"
        
        # Store pattern matrix for fast lookup
        self.pattern_matrix = pattern_matrix
        self.use_matrix = pattern_matrix is not None
        
        # Create word-to-index mappings for O(1) lookup
        if self.use_matrix:
            self.guess_to_idx = {word: i for i, word in enumerate(guess_list)}
            self.answer_to_idx = {word: i for i, word in enumerate(answer_list)}
    
    def get_pattern_fast(self, guess: str, answer: str) -> str:
        """
        Get pattern using precomputed matrix (if available).
        Falls back to regular get_pattern if matrix not available.
        
        This is ~100x faster than calculating pattern on the fly!
        """
        if self.use_matrix:
            guess_idx = self.guess_to_idx.get(guess)
            answer_idx = self.answer_to_idx.get(answer)
            
            if guess_idx is not None and answer_idx is not None:
                pattern_int = self.pattern_matrix[guess_idx, answer_idx]
                return pattern_int_to_str(pattern_int)
        
        # Fallback to regular calculation
        return get_pattern(guess, answer)
    
    def solve(self, answer: str, max_guesses: int = 6, verbose: bool = False) -> Tuple[List[str], bool]:
        raise NotImplementedError("Subclasses must implement solve()")
    
    def get_next_guess(self, possible_answers: List[str], guess_history: List[str]) -> str:
        raise NotImplementedError("Subclasses must implement get_next_guess()")


"""**NAIVE SOLVER CLASS**
Random selection with pattern-based filtering.
Uses precomputed matrix for fair comparison with entropy solver.
"""

class NaiveSolver(BaseSolver):
    def __init__(self, answer_list: List[str], guess_list: List[str],
                 pattern_matrix: Optional[np.ndarray] = None,
                 use_answer_list: bool = False):
        """
        Initialize Naive Solver.
        
        Args:
            answer_list: List of possible answers (2,315 words)
            guess_list: List of valid guesses (12,972 words)
            pattern_matrix: Precomputed patterns (for fairness & speed)
            use_answer_list: If True, only search answer_list (unfair but fast)
                           If False, search full guess_list (fair, realistic)
        """
        super().__init__(answer_list, guess_list, pattern_matrix)
        self.name = "Naive"
        self.use_answer_list = use_answer_list
        
        # Set search space based on configuration
        self.search_space = answer_list if use_answer_list else guess_list
    
    def get_next_guess(self, possible_answers: List[str],
                      guess_history: List[str]) -> str:
        """Pick random word from remaining candidates."""
        return random.choice(possible_answers)
    
    def solve(self, answer: str, max_guesses: int = 6, verbose: bool = False) -> Tuple[List[str], bool]:
        """
        Solve using random selection strategy.
        """
        answer = answer.upper()
        possible_answers = self.search_space.copy()
        guesses = []

        for attempt in range(max_guesses):
            guess = self.get_next_guess(possible_answers, guesses)
            guesses.append(guess)

            if verbose:
                print(f"Attempt {attempt + 1}: {guess} "
                      f"({len(possible_answers)} words remaining)")
            
            if guess == answer:
                if verbose:
                    print(f"✅ Solved in {len(guesses)} guesses!")
                return guesses, True
            
            # Use fast pattern lookup if matrix available
            pattern = self.get_pattern_fast(guess, answer)
            possible_answers = filter_words(possible_answers, guess, pattern)

            if verbose:
                from game_engine import format_pattern_emoji
                print(f"   Pattern: {format_pattern_emoji(pattern)}")
                print(f"   Remaining: {len(possible_answers)} words")
            
            if len(possible_answers) == 0:
                if verbose:
                    print("❌ No valid words remaining!")
                return guesses, False
        
        if verbose:
            print(f"❌ Failed after {max_guesses} guesses")
        return guesses, False


"""**ENTROPY SOLVER CLASS**
Information theory + word frequency.
Uses precomputed matrix for 60x speedup!
"""

class EntropySolver(BaseSolver):
    def __init__(self, answer_list: List[str], guess_list: List[str],
                 pattern_matrix: Optional[np.ndarray] = None,
                 word_freq_path: str = None,
                 use_answer_list: bool = False,
                 search_threshold: int = 50,
                 freq_weight: float = 0.01):
        """
        Initialize Entropy Solver.
        
        Args:
            answer_list: List of possible answers (2,315 words)
            guess_list: List of valid guesses (12,972 words)
            pattern_matrix: Precomputed patterns (HIGHLY RECOMMENDED!)
            word_freq_path: Path to word frequency file (optional)
            use_answer_list: If True, only search answer_list (unfair)
                           If False, search full guess_list (fair)
            search_threshold: When remaining words > threshold, use full search space
            freq_weight: Weight for frequency (0 = pure entropy, 1 = equal weight)
        """
        super().__init__(answer_list, guess_list, pattern_matrix)
        self.name = "Entropy"
        self.use_answer_list = use_answer_list
        self.search_threshold = search_threshold
        self.freq_weight = freq_weight
        
        # Set search space
        self.search_space = answer_list if use_answer_list else guess_list
        
        # Load word frequencies
        self.word_frequencies = self._load_word_frequencies(word_freq_path)
        
        # Precompute optimal first guess
        self.first_guess = "SOARE"  # Research-backed optimal
        # Alternatives: "RAISE", "ARISE", "IRATE", "SLATE"
    
    def _load_word_frequencies(self, freq_path: Optional[str]) -> Dict[str, float]:
        """Load word frequency database."""
        word_freq = {}
        
        if freq_path is None:
            # No frequency file - use uniform distribution
            for word in self.answer_list:
                word_freq[word] = 1.0
            for word in self.guess_list:
                if word not in word_freq:
                    word_freq[word] = 0.1
        else:
            # Load from file (format: word,frequency)
            try:
                with open(freq_path, 'r') as f:
                    next(f)
                    for line in f:
                        line = line.strip()
                        if line and ',' in line:
                            word, freq = line.split(',', 1)
                            word_freq[word.upper()] = float(freq)
            except Exception as e:
                print(f"Warning: Could not load frequency file: {e}")
                print("Using uniform distribution instead.")
                for word in self.answer_list:
                    word_freq[word] = 1.0
        
        # Normalize frequencies to 0-1 range
        if word_freq:
            max_freq = max(word_freq.values())
            if max_freq > 0:
                word_freq = {word: freq / max_freq for word, freq in word_freq.items()}
        
        return word_freq
    
    def calculate_entropy_fast(self, guess: str, possible_answer_indices: np.ndarray) -> float:
        """
        Calculate entropy using precomputed matrix (FAST!).
        
        This is ~60x faster than calculating patterns on the fly!
        
        Args:
            guess: Word to evaluate
            possible_answer_indices: NumPy array of indices in answer_list
        
        Returns:
            Entropy in bits
        """
        if len(possible_answer_indices) == 0:
            return 0.0
        if len(possible_answer_indices) == 1:
            return 0.0
        
        # Get guess index
        guess_idx = self.guess_to_idx.get(guess)
        if guess_idx is None:
            # Guess not in precomputed matrix, fall back to slow method
            return self.calculate_entropy_slow(guess, 
                [self.answer_list[i] for i in possible_answer_indices])
        
        # Get all patterns for this guess against possible answers
        # This is a SINGLE array slice - incredibly fast!
        patterns = self.pattern_matrix[guess_idx, possible_answer_indices]
        
        # Count occurrences of each pattern (vectorized!)
        unique, counts = np.unique(patterns, return_counts=True)
        
        # Calculate entropy using vectorized operations
        total = len(possible_answer_indices)
        probabilities = counts / total
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        return entropy
    
    def calculate_entropy_slow(self, guess: str, possible_answers: List[str]) -> float:
        """
        Fallback slow entropy calculation (without matrix).
        Only used if guess is not in precomputed matrix.
        """
        if len(possible_answers) <= 1:
            return 0.0
        
        patterns = []
        for answer in possible_answers:
            pattern = get_pattern(guess, answer)
            patterns.append(pattern)
        
        pattern_counts = Counter(patterns)
        total = len(possible_answers)
        entropy = 0.0
        
        for count in pattern_counts.values():
            if count > 0:
                probability = count / total
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def get_frequency_score(self, word: str) -> float:
        """Get normalized frequency score for a word (0-1)."""
        return self.word_frequencies.get(word, 0.1)
    
    def calculate_combined_score(self, guess: str, 
                                possible_answer_indices: np.ndarray) -> float:
        """
        Combine entropy and word frequency.
        
        Score = Entropy × (1 + freq_weight × Frequency)
        
        This gives entropy primary importance while using frequency
        as a tie-breaker for similar entropy values.
        """
        entropy = self.calculate_entropy_fast(guess, possible_answer_indices)
        frequency = self.get_frequency_score(guess)
        combined = entropy * (1.0 + self.freq_weight * frequency)
        return combined
    
    def find_best_guess(self, possible_answers: List[str]) -> str:
        """
        Find the guess with maximum combined score.
        
        Uses vectorized operations with precomputed matrix for speed.
        """
        # Edge cases
        if len(possible_answers) == 0:
            return random.choice(self.search_space)
        if len(possible_answers) == 1:
            return possible_answers[0]
        if len(possible_answers) == 2:
            # Just pick the more frequent one
            scores = [(w, self.get_frequency_score(w)) for w in possible_answers]
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[0][0]
        
        # Convert possible answers to indices for fast matrix lookup
        possible_indices = np.array([self.answer_to_idx[w] 
                                     for w in possible_answers 
                                     if w in self.answer_to_idx])
        
        # Determine search space based on remaining words
        if len(possible_answers) > self.search_threshold:
            search_space = self.search_space
        else:
            # Few words left - only search among candidates
            search_space = possible_answers
        
        # Find best guess using vectorized operations (FAST!)
        best_guess = None
        best_score = -1
        
        for guess in search_space:
            score = self.calculate_combined_score(guess, possible_indices)
            
            if score > best_score:
                best_score = score
                best_guess = guess
        
        return best_guess
    
    def get_next_guess(self, possible_answers: List[str],
                      guess_history: List[str]) -> str:
        """Choose next guess using entropy + frequency."""
        # Use precomputed first guess
        if len(guess_history) == 0:
            return self.first_guess
        
        return self.find_best_guess(possible_answers)
    
    def solve(self, answer: str, max_guesses: int = 6, verbose: bool = False) -> Tuple[List[str], bool]:
        """
        Solve using entropy maximization + frequency weighting.
        """
        answer = answer.upper()
        possible_answers = self.search_space.copy()
        guesses = []
        
        for attempt in range(max_guesses):
            guess = self.get_next_guess(possible_answers, guesses)
            guesses.append(guess)
            
            if verbose:
                if len(possible_answers) > 1:
                    # Calculate entropy for display
                    possible_indices = np.array([self.answer_to_idx[w] 
                                               for w in possible_answers 
                                               if w in self.answer_to_idx])
                    entropy = self.calculate_entropy_fast(guess, possible_indices)
                    freq = self.get_frequency_score(guess)
                    print(f"Attempt {attempt + 1}: {guess}")
                    print(f"   Entropy: {entropy:.3f} bits")
                    print(f"   Frequency: {freq:.3f}")
                    print(f"   Remaining: {len(possible_answers)} words")
                else:
                    print(f"Attempt {attempt + 1}: {guess} (only option)")
            
            if guess == answer:
                if verbose:
                    print(f"✅ Solved in {len(guesses)} guesses!")
                return guesses, True
            
            # Use fast pattern lookup
            pattern = self.get_pattern_fast(guess, answer)
            possible_answers = filter_words(possible_answers, guess, pattern)
            
            if verbose:
                from game_engine import format_pattern_emoji
                print(f"   Pattern: {format_pattern_emoji(pattern)}")
                print(f"   After filter: {len(possible_answers)} words")
            
            if len(possible_answers) == 0:
                if verbose:
                    print("❌ No valid words remaining!")
                return guesses, False
        
        if verbose:
            print(f"❌ Failed after {max_guesses} guesses")
        return guesses, False


# ============================================================================
# DEMO & QUICK TESTS
# ============================================================================

def demo_solvers():
    """Demo showing both solvers with precomputed matrix."""
    print("="*70)
    print("WORDLE SOLVER DEMO")
    print("="*70)
    print()
    
    # Load word lists
    try:
        from pattern_matrix import load_pattern_matrix
        
        list_a, list_b = load_word_lists()
        
        if len(list_a) < len(list_b):
            answers = list_a
            guesses = list_b
        else:
            answers = list_b
            guesses = list_a
        
        print(f"Loaded {len(answers)} answers, {len(guesses)} guesses")
    except Exception as e:
        print(f"Error loading word lists: {e}")
        return
    
    # Try to load precomputed matrix
    patterns = load_pattern_matrix("pattern_matrix.npy")
    
    if patterns is None:
        print("\n⚠️  No precomputed matrix found!")
        print("   Run this to create it:")
        print("   >>> from pattern_matrix import precompute_pattern_matrix, save_pattern_matrix")
        print("   >>> patterns = precompute_pattern_matrix(guesses, answers)")
        print("   >>> save_pattern_matrix(patterns)")
        print("\n   Continuing without matrix (will be slower)...")
    else:
        print(f"✅ Using precomputed matrix ({patterns.nbytes / 1024 / 1024:.1f} MB)")
    
    print()
    
    # Create solvers
    print("Creating solvers...")
    naive = NaiveSolver(guesses, guesses, pattern_matrix=patterns, 
                       use_answer_list=False)
    entropy = EntropySolver(guesses, guesses, pattern_matrix=patterns,
                           use_answer_list=False)
    
    # Test words
    test_words = ["CRANE", "STARE", "BOXER"]
    
    for word in test_words:
        print(f"\n{'='*70}")
        print(f"Target: {word}")
        print('='*70)
        
        print("\n--- Naive Solver ---")
        guesses_n, won_n = naive.solve(word, verbose=True)
        
        print("\n--- Entropy Solver ---")
        guesses_e, won_e = entropy.solve(word, verbose=True)
        
        print(f"\nComparison:")
        print(f"  Naive:   {len(guesses_n)} guesses")
        print(f"  Entropy: {len(guesses_e)} guesses")


def quick_performance_test():
    """Quick test showing speedup with matrix."""
    import time
    from pattern_matrix import load_pattern_matrix
    
    print("="*70)
    print("PERFORMANCE TEST: With vs Without Matrix")
    print("="*70)
    print()
    
    # Load data
    try:
        list_a, list_b = load_word_lists()
        if len(list_a) < len(list_b):
            answers, guesses = list_a, list_b
        else:
            answers, guesses = list_b, list_a
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Load matrix
    patterns = load_pattern_matrix("pattern_matrix.npy")
    
    if patterns is None:
        print("No matrix found - cannot run performance test")
        return
    
    # Test on 10 words
    test_words = random.sample(answers, 10)
    
    # With matrix
    print("Testing WITH matrix...")
    solver_fast = EntropySolver(answers, guesses, pattern_matrix=patterns,
                               use_answer_list=False)
    
    start = time.time()
    for word in test_words:
        solver_fast.solve(word)
    time_with = time.time() - start
    
    # Without matrix
    print("Testing WITHOUT matrix...")
    solver_slow = EntropySolver(answers, guesses, pattern_matrix=None,
                                use_answer_list=False)
    
    start = time.time()
    for word in test_words:
        solver_slow.solve(word)
    time_without = time.time() - start
    
    # Results
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"  Without matrix: {time_without:.2f} seconds")
    print(f"  With matrix:    {time_with:.2f} seconds")
    print(f"  Speedup:        {time_without / time_with:.1f}x faster! 🚀")
    print(f"\n  Extrapolated time for 2,315 words:")
    print(f"    Without matrix: {(time_without / 10) * 2315 / 60:.1f} minutes")
    print(f"    With matrix:    {(time_with / 10) * 2315 / 60:.1f} minutes")


if __name__ == "__main__":
  import sys
    
  if len(sys.argv) > 1 and sys.argv[1] == "perf":
    # Run: python solvers.py perf
    quick_performance_test()
  else:
    # Run: python solvers.py
    demo_solvers()
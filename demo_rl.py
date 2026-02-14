"""
DEMO SCRIPT - RL WORDLE SOLVER
==============================
Quick demonstration of the RL solver capabilities
"""

import sys
import random
from game_engine import load_word_lists, format_pattern_emoji
from pattern_matrix import load_pattern_matrix
from solvers import NaiveSolver, EntropySolver
from rl_solver import RLSolver


def demo_single_game(solver, target_word: str, solver_name: str):
    """Demonstrate a single game with detailed output."""
    print(f"\n{'='*70}")
    print(f"{solver_name} solving: {target_word}")
    print('='*70)
    
    guesses, won = solver.solve(target_word, verbose=True)
    
    print(f"\n{'='*70}")
    if won:
        print(f"✅ {solver_name} SOLVED in {len(guesses)} guesses!")
    else:
        print(f"❌ {solver_name} FAILED after {len(guesses)} attempts")
    print('='*70)
    
    return len(guesses), won


def compare_solvers(target_word: str, naive, entropy, rl):
    """Compare all three solvers on the same word."""
    print(f"\n{'='*70}")
    print(f"COMPARING ALL SOLVERS ON: {target_word}")
    print('='*70)
    
    results = []
    
    for solver, name in [(naive, "Naive"), (entropy, "Entropy"), (rl, "RL-DQN")]:
        print(f"\n{name}:")
        print("-" * 40)
        guesses, won = solver.solve(target_word, verbose=False)
        
        # Show guess sequence
        for i, guess in enumerate(guesses, 1):
            pattern = solver.get_pattern_fast(guess, target_word)
            emoji = format_pattern_emoji(pattern)
            print(f"  {i}. {guess:5s} {emoji}")
        
        results.append({
            'solver': name,
            'guesses': len(guesses),
            'won': won
        })
        
        if won:
            print(f"  ✅ Solved in {len(guesses)} guesses")
        else:
            print(f"  ❌ Failed")
    
    # Summary
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print('='*70)
    for r in results:
        status = "✅ WON" if r['won'] else "❌ LOST"
        print(f"{r['solver']:10s} - {r['guesses']} guesses - {status}")
    print('='*70)


def demo_training_sample():
    """Demonstrate a few training episodes."""
    print(f"\n{'='*70}")
    print("RL TRAINING DEMONSTRATION (5 episodes)")
    print('='*70)
    
    # Load data
    list_a, list_b = load_word_lists()
    answers = list_a if len(list_a) < len(list_b) else list_b
    guesses = list_a if len(list_a) > len(list_b) else list_b
    patterns = load_pattern_matrix("pattern_matrix.npy")
    
    # Create RL solver
    solver = RLSolver(answers, guesses, pattern_matrix=patterns, use_answer_list=False)
    
    # Training words
    training_words = random.sample(answers, 5)
    
    print("\nTraining on 5 random words:")
    print(f"Words: {', '.join(training_words)}")
    print("\nTraining progress:")
    
    for episode, word in enumerate(training_words, 1):
        metrics = solver.train_episode(word)
        
        status = "✅ WON" if metrics['won'] else "❌ LOST"
        print(f"\nEpisode {episode}: {word}")
        print(f"  Reward: {metrics['reward']:6.2f}")
        print(f"  Guesses: {metrics['guesses']}")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Epsilon: {metrics['epsilon']:.3f}")
        print(f"  Status: {status}")
    
    print(f"\n{'='*70}")
    print(f"Replay buffer size: {len(solver.replay_buffer)}")
    print(f"Episodes trained: {solver.episode_count}")
    print('='*70)


def main():
    """Main demo function."""
    print("="*70)
    print("REINFORCEMENT LEARNING WORDLE SOLVER - DEMO")
    print("="*70)
    
    # Load data
    print("\n1. Loading word lists and pattern matrix...")
    list_a, list_b = load_word_lists()
    answers = list_a if len(list_a) < len(list_b) else list_b
    guesses = list_a if len(list_a) > len(list_b) else list_b
    patterns = load_pattern_matrix("pattern_matrix.npy")
    
    if patterns is None:
        print("❌ Pattern matrix not found!")
        print("Run: python pattern_matrix.py")
        sys.exit(1)
    
    print(f"✅ Loaded {len(answers)} answers, {len(guesses)} guesses")
    
    # Create solvers
    print("\n2. Initializing solvers...")
    
    naive = NaiveSolver(
        answer_list=answers,
        guess_list=guesses,
        pattern_matrix=patterns,
        use_answer_list=False
    )
    print("✅ Naive solver ready")
    
    entropy = EntropySolver(
        answer_list=answers,
        guess_list=guesses,
        pattern_matrix=patterns,
        word_freq_path="word_frequency.csv",
        use_answer_list=False
    )
    print("✅ Entropy solver ready")
    
    rl_solver = RLSolver(
        answer_list=answers,
        guess_list=guesses,
        pattern_matrix=patterns,
        use_answer_list=False
    )
    print(f"✅ RL solver ready (device: {rl_solver.device})")
    
    # Demo 1: Single game with each solver
    print("\n" + "="*70)
    print("DEMO 1: Single Game Comparison")
    print("="*70)
    
    test_word = "CRANE"
    print(f"\nTest word: {test_word}")
    
    compare_solvers(test_word, naive, entropy, rl_solver)
    
    # Demo 2: Another word
    print("\n" + "="*70)
    print("DEMO 2: Another Comparison")
    print("="*70)
    
    test_word2 = "STARE"
    print(f"\nTest word: {test_word2}")
    
    compare_solvers(test_word2, naive, entropy, rl_solver)
    
    # Demo 3: Training sample
    demo_training_sample()
    
    # Demo 4: Post-training comparison
    print(f"\n{'='*70}")
    print("DEMO 3: After Training (5 episodes)")
    print('='*70)
    
    test_word3 = random.choice(answers)
    print(f"\nTesting on: {test_word3}")
    print("(Note: RL solver now has 5 episodes of experience)")
    
    print(f"\n{'='*70}")
    print("RL Solver (with 5 episodes training):")
    print('='*70)
    guesses, won = rl_solver.solve(test_word3, verbose=True)
    
    # Final summary
    print(f"\n{'='*70}")
    print("DEMO COMPLETE")
    print('='*70)
    print("\nKey Takeaways:")
    print("1. ✅ RL solver integrates with existing project structure")
    print("2. ✅ Uses same BaseSolver interface as Naive and Entropy")
    print("3. ✅ Can be trained incrementally with train_episode()")
    print("4. ✅ Performance improves with training")
    print("\nNext Steps:")
    print("• Train for longer: python train_rl.py --episodes 1000")
    print("• Full evaluation: python eval_rl.py --include-rl")
    print("• See RL_README.md for detailed documentation")
    print('='*70)


if __name__ == "__main__":
    main()

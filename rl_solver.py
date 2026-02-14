"""
REINFORCEMENT LEARNING WORDLE SOLVER
====================================
Deep Q-Network (DQN) implementation for Wordle solving.

Integrates with existing project structure:
- Uses BaseSolver interface from solvers.py
- Uses precomputed pattern_matrix for speed
- Compatible with eval.py benchmark framework
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, Counter
from typing import List, Tuple, Dict, Optional
from game_engine import get_pattern, filter_words
from solvers import BaseSolver
from pattern_matrix import pattern_int_to_str


# ============================================================================
# STATE ENCODER
# ============================================================================

class StateEncoder:
    """
    Encodes the current Wordle game state into a fixed-size feature vector.
    
    Features (total: 156 dimensions):
    - Game progress features (6 dims)
    - Letter position frequencies (26 x 5 = 130 dims)
    - Constraint features (20 dims)
    """
    
    def __init__(self, answer_list: List[str], guess_list: List[str]):
        self.answer_list = answer_list
        self.guess_list = guess_list
        self.alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.feature_dim = 156
        
    def encode_state(self, 
                     possible_answers: List[str],
                     guess_history: List[Tuple[str, str]]) -> np.ndarray:
        """
        Convert game state to feature vector.
        
        Args:
            possible_answers: List of remaining candidate words
            guess_history: [(guess, pattern), ...] history of guesses
            
        Returns:
            numpy array of shape (feature_dim,)
        """
        features = []
        
        # 1. Game progress features (6 dims)
        n_candidates = len(possible_answers)
        n_guesses = len(guess_history)
        
        features.extend([
            n_candidates / len(self.answer_list),   # Normalized count
            np.log10(max(1, n_candidates)) / 4,     # Log scale
            1.0 if n_candidates <= 2 else 0.0,      # Endgame flag
            1.0 if n_candidates <= 10 else 0.0,     # Near-end flag  
            n_guesses / 6,                          # Turn number
            1.0 if n_guesses == 0 else 0.0,         # First guess flag
        ])
        
        # 2. Letter position frequencies (26 x 5 = 130 dims)
        position_freqs = self._compute_position_frequencies(possible_answers)
        features.extend(position_freqs.flatten())
        
        # 3. Constraint features from history (20 dims)
        constraint_features = self._encode_constraints(guess_history, n_candidates)
        features.extend(constraint_features)
        
        return np.array(features, dtype=np.float32)
    
    def _compute_position_frequencies(self, words: List[str]) -> np.ndarray:
        """
        Compute frequency of each letter at each position.
        
        Returns: (26, 5) array where [i, j] = freq of letter i at position j
        """
        freqs = np.zeros((26, 5), dtype=np.float32)
        
        if len(words) == 0:
            return freqs
        
        for word in words:
            for pos, char in enumerate(word):
                if char in self.alphabet:
                    letter_idx = ord(char) - ord('A')
                    freqs[letter_idx, pos] += 1
        
        # Normalize by number of words
        freqs /= max(1, len(words))
        
        return freqs
    
    def _encode_constraints(self, 
                           guess_history: List[Tuple[str, str]],
                           n_candidates: int) -> List[float]:
        """
        Encode known constraints from guess history.
        """
        features = [0.0] * 20
        
        if not guess_history:
            return features
        
        idx = 0
        
        # Last 3 guesses: greens, yellows, grays (9 dims)
        for i in range(3):
            if i < len(guess_history):
                guess, pattern = guess_history[-(i+1)]
                n_greens = pattern.count('2')
                n_yellows = pattern.count('1')
                n_grays = pattern.count('0')
                features[idx:idx+3] = [n_greens/5, n_yellows/5, n_grays/5]
            idx += 3
        
        # Cumulative counts (3 dims)
        all_patterns = [p for _, p in guess_history]
        total_greens = sum(p.count('2') for p in all_patterns)
        total_yellows = sum(p.count('1') for p in all_patterns)
        total_grays = sum(p.count('0') for p in all_patterns)
        features[idx:idx+3] = [total_greens/25, total_yellows/25, total_grays/130]
        idx += 3
        
        # Information gain metrics (8 dims)
        if len(guess_history) >= 1:
            initial_size = len(self.answer_list)
            reduction_rate = 1 - (n_candidates / initial_size)
            features[idx] = reduction_rate
            idx += 1
            
            avg_reduction = reduction_rate / len(guess_history)
            features[idx] = avg_reduction
            idx += 1
        else:
            idx += 2
        
        unique_patterns = len(set(p for _, p in guess_history[-3:]))
        features[idx] = unique_patterns / min(3, max(1, len(guess_history)))
        
        return features


# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a random batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# DQN NETWORK
# ============================================================================

class DQNetwork(nn.Module):
    """Deep Q-Network for Wordle."""
    
    def __init__(self, state_dim: int, action_dim: int = 130, hidden_dims: List[int] = [256, 256, 128]):
        super(DQNetwork, self).__init__()
        
        # State encoder
        state_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            state_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.state_encoder = nn.Sequential(*state_layers)
        self.state_dim = prev_dim
        
        # Action encoder (word encoding to features)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, prev_dim)
        )
        
        # Combined Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(prev_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: (batch, state_dim)
            action: (batch, action_dim) optional action encoding
            
        Returns:
            If action provided: Q-values (batch, 1)
            If no action: state features (batch, state_dim)
        """
        state_features = self.state_encoder(state)
        
        if action is None:
            return state_features
        
        action_features = self.action_encoder(action)
        combined = torch.cat([state_features, action_features], dim=-1)
        q_value = self.q_head(combined)
        return q_value.squeeze(-1)


# ============================================================================
# RL SOLVER
# ============================================================================

class RLSolver(BaseSolver):
    """
    Reinforcement Learning Wordle Solver using DQN.
    """
    
    def __init__(self, 
                 answer_list: List[str], 
                 guess_list: List[str],
                 pattern_matrix: Optional[np.ndarray] = None,
                 use_answer_list: bool = False,
                 device: str = None):
        """Initialize RL Solver."""
        super().__init__(answer_list, guess_list, pattern_matrix)
        self.name = "RL-DQN"
        self.use_answer_list = use_answer_list
        self.search_space = answer_list if use_answer_list else guess_list
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # State encoder
        self.encoder = StateEncoder(answer_list, guess_list)
        
        # Neural networks
        self.policy_net = DQNetwork(self.encoder.feature_dim).to(self.device)
        self.target_net = DQNetwork(self.encoder.feature_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Training components
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
        # Hyperparameters
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.target_update_freq = 10
        
        # Training state
        self.episode_count = 0
        self.training_mode = False
        
        # Word mappings
        self.word_to_idx = {word: i for i, word in enumerate(self.search_space)}
        self.idx_to_word = {i: word for word, i in self.word_to_idx.items()}
        
        # Optimal first guess
        self.first_guess = "SOARE"
    
    def encode_word_simple(self, word: str) -> np.ndarray:
        """Simple word encoding: letter frequencies by position."""
        encoding = np.zeros((26, 5), dtype=np.float32)
        for pos, char in enumerate(word):
            if char in self.encoder.alphabet:
                letter_idx = ord(char) - ord('A')
                encoding[letter_idx, pos] = 1.0
        return encoding.flatten()
    
    def select_action(self, 
                     state: np.ndarray,
                     possible_answers: List[str],
                     epsilon: float = None) -> str:
        """Select action using epsilon-greedy strategy."""
        if epsilon is None:
            epsilon = self.epsilon
        
        if len(possible_answers) == 0:
            return random.choice(self.search_space)
        if len(possible_answers) == 1:
            return possible_answers[0]
        
        # Epsilon-greedy
        if random.random() < epsilon and self.training_mode:
            return random.choice(self.search_space)
        
        return self._get_best_action(state, possible_answers)
    
    def _get_best_action(self, state: np.ndarray, candidates: List[str]) -> str:
        """Get best action from neural network using proper Q-values."""
        self.policy_net.eval()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            best_word = None
            best_q_value = float('-inf')
            
            # Limit search space for efficiency
            search_words = candidates if len(candidates) <= 100 else self.search_space[:500]
            
            # Batch process for efficiency
            batch_size = 32
            for i in range(0, len(search_words), batch_size):
                batch_words = search_words[i:i+batch_size]
                
                # Encode all words in batch
                action_batch = []
                for word in batch_words:
                    word_encoding = self.encode_word_simple(word)
                    action_batch.append(word_encoding)
                
                action_tensor = torch.FloatTensor(action_batch).to(self.device)
                state_batch = state_tensor.repeat(len(batch_words), 1)
                
                # Get Q-values
                q_values = self.policy_net(state_batch, action_tensor)
                
                # Find best in this batch
                max_idx = q_values.argmax().item()
                max_q = q_values[max_idx].item()
                
                if max_q > best_q_value:
                    best_q_value = max_q
                    best_word = batch_words[max_idx]
            
            if best_word is None:
                best_word = random.choice(candidates)
        
        self.policy_net.train()
        return best_word
    
    def get_next_guess(self, 
                       possible_answers: List[str],
                       guess_history: List[str]) -> str:
        """Get next guess (BaseSolver interface)."""
        if len(guess_history) == 0:
            return self.first_guess
        
        history_with_patterns = []
        state = self.encoder.encode_state(possible_answers, history_with_patterns)
        return self.select_action(state, possible_answers, epsilon=0.0)
    
    def solve(self, answer: str, max_guesses: int = 6, 
              verbose: bool = False) -> Tuple[List[str], bool]:
        """Solve a Wordle puzzle (BaseSolver interface)."""
        answer = answer.upper()
        possible_answers = self.search_space.copy()
        guesses = []
        guess_history = []
        
        for attempt in range(max_guesses):
            state = self.encoder.encode_state(possible_answers, guess_history)
            guess = self.select_action(state, possible_answers, epsilon=0.0)
            guesses.append(guess)
            
            if verbose:
                print(f"Attempt {attempt + 1}: {guess} "
                      f"({len(possible_answers)} words remaining)")
            
            if guess == answer:
                if verbose:
                    print(f"âœ… Solved in {len(guesses)} guesses!")
                return guesses, True
            
            pattern = self.get_pattern_fast(guess, answer)
            guess_history.append((guess, pattern))
            possible_answers = filter_words(possible_answers, guess, pattern)
            
            if verbose:
                from game_engine import format_pattern_emoji
                print(f"   Pattern: {format_pattern_emoji(pattern)}")
                print(f"   Remaining: {len(possible_answers)} words")
            
            if len(possible_answers) == 0:
                if verbose:
                    print("âŒ No valid words remaining!")
                return guesses, False
        
        if verbose:
            print(f"âŒ Failed after {max_guesses} guesses")
        return guesses, False
    
    # ========================================================================
    # TRAINING METHODS
    # ========================================================================
    
    def train_episode(self, target_word: str) -> Dict[str, float]:
        """Train on a single episode."""
        self.training_mode = True
        target_word = target_word.upper()
        
        possible_answers = self.search_space.copy()
        guess_history = []
        episode_reward = 0
        transitions = []
        
        for attempt in range(6):
            state = self.encoder.encode_state(possible_answers, guess_history)
            guess = self.select_action(state, possible_answers)
            pattern = self.get_pattern_fast(guess, target_word)
            guess_history.append((guess, pattern))
            
            reward = self._calculate_reward(guess, target_word, pattern, attempt)
            episode_reward += reward
            done = (guess == target_word)
            
            if not done:
                possible_answers = filter_words(possible_answers, guess, pattern)
                next_state = self.encoder.encode_state(possible_answers, guess_history)
            else:
                next_state = state
            
            action_idx = self.word_to_idx.get(guess, 0)
            transitions.append((state, action_idx, reward, next_state, done))
            
            if done:
                break
            
            if len(possible_answers) == 0:
                transitions[-1] = (state, action_idx, -10.0, next_state, True)
                break
        
        # Add to replay buffer
        for transition in transitions:
            self.replay_buffer.push(*transition)
        
        # Train
        if len(self.replay_buffer) >= self.batch_size:
            loss = self.train_step()
        else:
            loss = 0.0
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network
        self.episode_count += 1
        if self.episode_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.training_mode = False
        
        return {
            'reward': episode_reward,
            'guesses': len(transitions),
            'won': guess == target_word,
            'loss': loss,
            'epsilon': self.epsilon
        }
    
    def _calculate_reward(self, guess: str, answer: str, 
                         pattern: str, attempt: int) -> float:
        """Calculate reward for a guess."""
        if guess == answer:
            return 10.0 + (6 - attempt) * 2.0
        
        reward = 0.0
        reward += pattern.count('2') * 0.5  # Green
        reward += pattern.count('1') * 0.2  # Yellow
        reward -= 0.5  # Penalty per guess
        
        return reward
    
    def train_step(self) -> float:
        """Perform one training step using experience replay."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get action encodings
        action_encodings = []
        for action_idx in actions:
            if action_idx < len(self.search_space):
                word = self.search_space[action_idx]
                action_encodings.append(self.encode_word_simple(word))
            else:
                action_encodings.append(np.zeros(130, dtype=np.float32))
        
        action_tensor = torch.FloatTensor(action_encodings).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states, action_tensor)
        
        # Target Q values
        with torch.no_grad():
            # Sample actions for next state
            next_q_values = []
            for i in range(len(next_states)):
                sample_actions = random.sample(range(min(100, len(self.search_space))), 
                                             min(10, len(self.search_space)))
                max_next_q = float('-inf')
                
                for act_idx in sample_actions:
                    word = self.search_space[act_idx]
                    act_enc = self.encode_word_simple(word)
                    act_tensor = torch.FloatTensor(act_enc).unsqueeze(0).to(self.device)
                    state_tensor = next_states[i:i+1]
                    q_val = self.target_net(state_tensor, act_tensor).item()
                    max_next_q = max(max_next_q, q_val)
                
                next_q_values.append(max_next_q)
            
            next_q_values = torch.FloatTensor(next_q_values).to(self.device)
            target_q = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Loss and optimize
        loss = F.smooth_l1_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def save_model(self, filepath: str):
        """Save model weights."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
        }, filepath)
        print(f"âœ… Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.episode_count = checkpoint.get('episode_count', 0)
        print(f"âœ… Model loaded from {filepath}")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    from game_engine import load_word_lists
    from pattern_matrix import load_pattern_matrix
    
    print("="*70)
    print("RL SOLVER TEST")
    print("="*70)
    
    # Load data
    list_a, list_b = load_word_lists()
    answers = list_a if len(list_a) < len(list_b) else list_b
    guesses = list_a if len(list_a) > len(list_b) else list_b
    patterns = load_pattern_matrix("pattern_matrix.npy")
    
    print(f"Answers: {len(answers)}, Guesses: {len(guesses)}")
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Create solver
    solver = RLSolver(answers, guesses, pattern_matrix=patterns, use_answer_list=False)
    
    # Test on a few words
    test_words = ["CRANE", "STARE", "BOXER"]
    
    print("\n" + "="*70)
    print("TESTING UNTRAINED MODEL")
    print("="*70)
    
    for word in test_words:
        print(f"\nTarget: {word}")
        guesses, won = solver.solve(word, verbose=True)
        print(f"Result: {'WON' if won else 'LOST'} in {len(guesses)} guesses")
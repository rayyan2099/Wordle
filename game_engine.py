
def get_pattern(guess:str, answer:str)->str:
  guess=guess.upper()
  answer=answer.upper()
  pattern=['0']*5
  answer_chars=list(answer)
  guess_chars=list(guess)
  """
  0 -> Gray
  1 -> Yellow
  2 -> Green
  """
  for i in range(5):
    if guess_chars[i]== answer_chars[i]:
        pattern[i]='2'
        answer_chars[i] = None
        guess_chars[i] = None
  for i in range(5):
    if guess_chars[i] is not None:
      if guess_chars[i] in answer_chars:
        pattern[i]='1'
        idx = answer_chars.index(guess_chars[i])
        answer_chars[idx]=None

  return ''.join(pattern)

def matches_pattern(word: str , guess: str , pattern: str)->bool:
  return get_pattern(guess , word) == pattern

def filter_words(word_list: list[str], guess: str, pattern: str)->list[str]:
  return [word for word in word_list if matches_pattern(word, guess , pattern)]

from typing import Tuple

def load_word_lists(answers_path: str = None, guesses_path: str = None) -> Tuple[list[str],list[str]]:
  if answers_path is None:
    answers_path = "valid_solutions.csv"

  if guesses_path is None:
    guesses_path = "all_words.csv"

  with open(answers_path, 'r') as f:
    answers = [line.strip().upper() for line in f if line.strip()]

  with open(guesses_path, 'r') as f:
    guesses = [line.strip().upper() for line in f if line.strip()]

  return answers[1:], guesses[1:]

def format_pattern_emoji(pattern:str)->str:
  colors = {
    '0': '⬛', # Gray
    '1': '🟨', # Yellow
    '2': '🟩' # Green
   }
  return ''.join(colors[c] for c in pattern)

def test_pattern_matching():
 """Test pattern matching with various cases."""

 # Basic test
 assert get_pattern("STARE", "CRANE") == "00212"
 assert get_pattern("RAISE", "CRANE") == "11002"

 # Duplicate letters
 assert get_pattern("ALLAY", "LLAMA") == "12110"
 assert get_pattern("SPEED", "ERASE") == "10110"
 assert get_pattern("ROBOT", "FLOOR") == "11020"

 # All green
 assert get_pattern("CRANE", "CRANE") == "22222"

 # All gray
 assert get_pattern("QUICK", "BEANS") == "00000"

 print("■ All pattern tests passed.")

def test_filtering():
 """Test word filtering."""
 words = ["CRANE", "CRATE", "CRAZE", "GRAZE", "TRAIN"]
 filtered = filter_words(words, "STARE", "00212")
 assert "CRANE" in filtered
 assert "CRATE" not in filtered

 print("■ All filtering tests passed.")

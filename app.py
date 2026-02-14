import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from game_engine import load_word_lists, get_pattern
from pattern_matrix import load_pattern_matrix, pattern_int_to_str
from solvers import EntropySolver


# ============================================================
# 1. CACHED ASSET LOADING
# ============================================================

@st.cache_resource
def load_assets():
    list_a, list_b = load_word_lists()
    guesses = list_a if len(list_a) > len(list_b) else list_b
    # Blind mode: solver sees only guess_list
    answers = guesses
    matrix = load_pattern_matrix("pattern_matrix.npy")
    return answers, guesses, matrix


answers, guesses, matrix = load_assets()

if "solver" not in st.session_state:
    st.session_state.solver = EntropySolver(
        answers, guesses, matrix,
        word_freq_path="word_frequency.csv",
        use_answer_list=False
    )


# ============================================================
# 2. RANKING ENGINE
#    Runs once per game-state, cached in session_state.
#    Computes entropy + frequency + combined score for every
#    word in guess_list against the current possible_indices.
# ============================================================

def rank_all_guesses(solver, possible_indices: np.ndarray) -> pd.DataFrame:
    """
    Score every word in guess_list against the current candidate set.

    Returns a DataFrame sorted by combined score descending, with columns:
        word        – the guess
        entropy     – bits of information this guess provides
        frequency   – normalised word frequency (0–1)
        score       – combined score (entropy × (1 + freq_weight × freq))
        is_candidate– True if the word is still in possible_indices
                      (i.e. it could actually BE the answer)
    """
    n = len(solver.guess_list)
    candidate_set = set(possible_indices.tolist())

    words = []
    entropies = np.zeros(n)
    frequencies = np.zeros(n)

    for i, word in enumerate(solver.guess_list):
        words.append(word)
        entropies[i] = solver.calculate_entropy_fast(word, possible_indices)
        frequencies[i] = solver.get_frequency_score(word)

    scores = entropies * (1.0 + solver.freq_weight * frequencies)
    is_candidate = np.array([
        solver.guess_to_idx[w] in candidate_set for w in words
    ])

    df = pd.DataFrame({
        "word": words,
        "entropy": entropies,
        "frequency": frequencies,
        "score": scores,
        "is_candidate": is_candidate,
    })
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # 1-based rank
    df.index.name = "rank"
    return df


def get_partition_sizes(solver, guess: str, possible_indices: np.ndarray) -> pd.DataFrame:
    """
    For a single guess, show how it partitions the candidate set by pattern.
    Each row = one pattern outcome, with count and probability.
    Sorted by count descending.
    """
    guess_idx = solver.guess_to_idx[guess]
    patterns = matrix[guess_idx, possible_indices]
    unique, counts = np.unique(patterns, return_counts=True)

    total = len(possible_indices)
    rows = []
    for pat_int, cnt in zip(unique, counts):
        pat_str = pattern_int_to_str(int(pat_int))
        emoji = pat_str.replace("2", "🟩").replace("1", "🟨").replace("0", "⬛")
        rows.append({
            "pattern": pat_str,
            "emoji": emoji,
            "count": int(cnt),
            "probability": cnt / total,
        })

    df = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
    return df


# ============================================================
# 3. SESSION STATE HELPERS
# ============================================================

def init_state():
    if "history" not in st.session_state:
        st.session_state.history = []
        st.session_state.possible_indices = np.arange(len(guesses))
        st.session_state.rankings = None
        st.session_state.rankings_version = -1  # tracks which history length the rankings match


def reset_game():
    st.session_state.history = []
    st.session_state.possible_indices = np.arange(len(guesses))
    st.session_state.rankings = None
    st.session_state.rankings_version = -1


def rankings_are_stale():
    """True if rankings haven't been computed for the current game state."""
    return st.session_state.rankings_version != len(st.session_state.history)


init_state()


# ============================================================
# 4. SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Controls")
target_word = st.sidebar.text_input(
    "Target Word (hidden)", type="password"
).upper().strip()

top_n = st.sidebar.slider("Words to show", min_value=5, max_value=50, value=20, step=5)

if st.sidebar.button("🔄 Reset Game"):
    reset_game()

st.sidebar.divider()
st.sidebar.markdown(
    "**How it works**\n\n"
    "Enter a target word, then either guess yourself or click "
    "**Rank All Guesses** to see the engine's full scored list.\n\n"
    "🟩 = correct position · 🟨 = wrong position · ⬛ = not in word"
)


# ============================================================
# 5. MAIN LAYOUT
# ============================================================

st.title("🤖 The Entropy Engine")
st.caption("Wordle solver — information-theoretic ranking of every possible guess")

col_game, col_rank = st.columns([2, 3], gap="large")


# ─────────────────────────────────────────────
# LEFT: game board
# ─────────────────────────────────────────────
with col_game:
    st.subheader("Game Board")

    # Guess input
    user_guess = st.text_input("Your guess", placeholder="5-letter word", key="guess_input").upper().strip()

    submit_clicked = st.button("Submit Guess", use_container_width=True)

    if submit_clicked:
        if not target_word:
            st.error("Set a target word in the sidebar first.")
        elif len(user_guess) != 5:
            st.warning("Guess must be exactly 5 letters.")
        elif user_guess not in st.session_state.solver.guess_to_idx:
            st.warning(f"'{user_guess}' is not in the word list.")
        else:
            pat = get_pattern(user_guess, target_word)
            st.session_state.history.append((user_guess, pat))

            # Filter possible_indices using the matrix (fast vectorised path)
            idx = st.session_state.solver.guess_to_idx[user_guess]
            p_int = int(pat, 3)
            row = matrix[idx, st.session_state.possible_indices]
            st.session_state.possible_indices = st.session_state.possible_indices[row == p_int]

            # Invalidate cached rankings so they recompute for new state
            st.session_state.rankings = None

    # History board
    if st.session_state.history:
        st.markdown("---")
        for i, (g, p) in enumerate(st.session_state.history, 1):
            emoji = p.replace("2", "🟩").replace("1", "🟨").replace("0", "⬛")
            st.markdown(f"**{i}.** `{g}`  {emoji}")
    else:
        st.markdown("*No guesses yet.*")

    # Remaining count
    remaining = len(st.session_state.possible_indices)
    st.markdown("---")
    st.metric("Candidates remaining", remaining)

    # Win detection
    if remaining == 1:
        answer_word = guesses[st.session_state.possible_indices[0]]
        st.balloons()
        st.success(f"🎉 Solved! The word is **{answer_word}**", icon="✅")
    elif remaining == 0:
        st.error("Something went wrong — no candidates left. Reset and try again.")


# ─────────────────────────────────────────────
# RIGHT: ranked guesses + charts
# ─────────────────────────────────────────────
with col_rank:
    st.subheader("Ranked Guesses")
    remaining = len(st.session_state.possible_indices)

    if remaining <= 1:
        st.info("Game is over — reset to play again.")

    else:
        # ── Compute rankings (cached until history changes) ──
        need_compute = rankings_are_stale()

        if need_compute:
            with st.spinner(f"Scoring all {len(guesses):,} words…"):
                st.session_state.rankings = rank_all_guesses(
                    st.session_state.solver,
                    st.session_state.possible_indices
                )
                st.session_state.rankings_version = len(st.session_state.history)

        df = st.session_state.rankings

        if df is not None:
            # ── Table: top N ──
            display = df.head(top_n).copy()
            display["entropy"] = display["entropy"].round(3)
            display["frequency"] = display["frequency"].round(3)
            display["score"] = display["score"].round(3)
            display["candidate"] = display["is_candidate"].map({True: "✅ yes", False: ""})
            display = display[["word", "entropy", "score", "frequency", "candidate"]]
            display.columns = ["Word", "Entropy (bits)", "Score", "Freq", "Candidate?"]

            st.dataframe(display, use_container_width=True)

            # ── Bar chart: entropy of top N ──
            top = df.head(top_n).copy()
            top["label"] = top.apply(
                lambda r: r["word"] + (" ⭐" if r["is_candidate"] else ""), axis=1
            )
            top["color"] = top["is_candidate"].map({True: "#34d399", False: "#60a5fa"})

            fig_bar = go.Figure(
                data=[go.Bar(
                    x=top["label"],
                    y=top["entropy"],
                    marker_color=top["color"].tolist(),
                    text=[f"{e:.3f}" for e in top["entropy"]],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Entropy: %{y:.3f} bits<extra></extra>",
                )]
            )
            fig_bar.update_layout(
                title="Entropy by Word",
                xaxis_title=None,
                yaxis_title="Bits",
                xaxis_tickangle=-45,
                height=340,
                margin=dict(t=40, b=80, l=40, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=None,
            )
            fig_bar.add_annotation(
                x=0.5, y=-0.18, xref="paper", yref="paper",
                text="🟢 ⭐ = still a candidate &nbsp;&nbsp; 🔵 = information-only guess",
                showarrow=False, font=dict(size=11, color="#888")
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # ── Partition chart for the #1 word ──
            best_word = df.iloc[0]["word"]
            st.markdown(f"**Pattern partitions for `{best_word}`** (the top-ranked guess)")
            st.caption(
                "Each bar = one possible pattern outcome. "
                "More, smaller bars = more information gained."
            )

            partitions = get_partition_sizes(
                st.session_state.solver, best_word, st.session_state.possible_indices
            )

            prob_labels = partitions["probability"].apply(lambda p: f"{p:.1%}").tolist()
            fig_part = go.Figure(
                data=[go.Bar(
                    x=partitions["emoji"],
                    y=partitions["count"],
                    marker_color="#a78bfa",
                    text=[f"{c}" for c in partitions["count"]],
                    textposition="outside",
                    customdata=prob_labels,
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Count: %{y}<br>"
                        "P = %{customdata}<extra></extra>"
                    ),
                )]
            )
            fig_part.update_layout(
                title=f"How '{best_word}' splits {remaining} candidates",
                xaxis_title="Pattern",
                yaxis_title="Words in bucket",
                height=300,
                margin=dict(t=40, b=60, l=40, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_part, use_container_width=True)
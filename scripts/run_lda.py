"""Fit LDA for one or more numbers of topics K, with a fixed seed, and save each run.

Command:
    python scripts/run_lda.py 10                      one run, K = 10
    python scripts/run_lda.py 5 10 15 --n-iter 500    several runs, one after the other

Input:
    data/song_bow.parquet             (from scripts/build_song_bow.py)

Output, one folder per run:  data/lda_grid_search/k{K}_seed{SEED}/
    model.npz            fitted model: settings, phi, theta, log-likelihood, vocab, songs.
                         Reload with:  LDA.load("data/lda_grid_search/k10_seed0/model.npz")

    song_topics.parquet  one row per song: song, topic_0 ... topic_{K-1} (the song's topic mix).
                         song is the same key as in song_bow.parquet, so the two can be joined on it.

    top_words.csv        one row per topic: its share of all songs and its top words. Read these to pick K.

    convergence.png      log P(w|z) after each sweep. It should level off; if it is still rising, use more sweeps.

A run whose folder already has model.npz is skipped, so a killed job can be restarted with the same command.
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no screen on the HPC: draw straight to files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# 1. Config
ROOT = Path(__file__).resolve().parent.parent      # repo root (this file is in scripts/)
sys.path.insert(0, str(ROOT))                       # so lda_gibbs.py in the repo root can be imported
from lda_gibbs import LDA

DATA = ROOT / "data"
OUT = DATA / "lda_grid_search"

# 2. Define stopwords and filtering threshold
FILLERS = {"oh", "ooh", "yeah", "yea", "la", "na", "da", "hey", "uh", "ah", "whoa", "mmm", "ha",
           "doo", "ya", "yo", "huh", "woah", "ohh", "ay", "eh", "im", "dont", "aint", "caus", "gonna",
           "wanna", "got", "get", "let", "just", "like"}
STOP = set(ENGLISH_STOP_WORDS) | FILLERS
MAX_DF = 0.4      # drop words found in more than 40% of songs
MIN_DF = 20       # drop words found in fewer than 20 songs

def load_counts():
    """
        Song x word count matrix X, with uninformative words removed.
        Returns X, the song index of each row, and the word of each column.
    """
    # 1. Load data
    bow = pd.read_parquet(DATA / "song_bow.parquet")
    songs = np.sort(bow["song"].unique())
    vocab = np.sort(bow["word"].unique())

    # 2. Create index arrays
    row = np.searchsorted(songs, bow["song"].to_numpy())    # index of song 
    col = np.searchsorted(vocab, bow["word"].to_numpy())    # index of word 

    # 3. Build sparse matrix 
    X = sp.csr_matrix((bow["count"].to_numpy(dtype=float), (row, col)), shape=(len(songs), len(vocab)))

    # 4. Filter out uninformative words
    doc_freq = np.asarray((X > 0).sum(axis=0)).ravel()
    keep_word = (~np.isin(vocab, list(STOP))) & (doc_freq / X.shape[0] <= MAX_DF) & (doc_freq >= MIN_DF)
    
    X, vocab = X[:, keep_word], vocab[keep_word]
    return X, songs, vocab

def save_run(model, folder, songs, vocab, n_top):
    """
    Write the three output files of one run into output folder.
    """
    folder.mkdir(parents=True, exist_ok=True)

    K = model.n_topics
    # 1. Model
    model.save(folder / "model.npz", vocab=vocab, songs=songs)

    # 2. Topic mix of each song, keyed like song_bow.parquet
    mix = pd.DataFrame(model.doc_topic, columns=[f"topic_{j}" for j in range(K)])
    mix.insert(0, "song", songs) # Insert the song index as the first column, so it can be joined with song_bow.parquet
    mix.to_parquet(folder / "song_topics.parquet", index=False)

    # 3. Top words of each topic
    top = pd.DataFrame({
        "topic": np.arange(K),
        "share": model.doc_topic.mean(axis=0).round(4),
        "top_words": [" ".join(words) for words in model.top_words(vocab, n=n_top)],
    })
    top.to_csv(folder / "top_words.csv", index=False)

    # 4. Convergence plot: log P(w|z) should rise and then level off
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(model.log_likelihood) + 1), model.log_likelihood)
    ax.set_xlabel("sweep")
    ax.set_ylabel("log P(w | z)")
    ax.set_title(f"K={K}: {late_rise(model.log_likelihood):.1%} of the rise in the last quarter")
    fig.tight_layout()
    fig.savefig(folder / "convergence.png", dpi=120)
    plt.close(fig)

def late_rise(log_likelihood):
    """Share of the total rise in log P(w|z) that happened in the last quarter of the sweeps.
    Close to 0 means the sampler had settled; a few percent or more means more sweeps are needed."""
    ll = np.asarray(log_likelihood)
    total = ll[-1] - ll[0] # Get fibnal minus initial log-likelihood

    if total <= 0:
        return 0.0
    
    return (ll[-1] - ll[3 * len(ll) // 4]) / total # Difference between final and 3/4 log-likelihood divided by total rise

def main():
    # 1. Define argparse stuff
    parser = argparse.ArgumentParser()
    parser.add_argument("ks", type=int, nargs="+", help="numbers of topics to try")
    parser.add_argument("--n-iter", type=int, default=200, help="Gibbs sweeps per run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-top", type=int, default=15, help="words listed per topic")
    args = parser.parse_args()

    # 2. Load data
    X, songs, vocab = load_counts()
    print(f"{X.shape[0]:,} songs, {len(vocab):,} words, {int(X.sum()):,} tokens", flush=True)

    # 3. Fit LDA for each K
    for k in args.ks:
        folder = OUT / f"k{k}_seed{args.seed}"
        if (folder / "model.npz").exists():
            print(f"K={k}: {folder} already done, skipping", flush=True)
            continue
        
        start = time.time()
        model = LDA(n_topics=k, n_iter=args.n_iter, random_state=args.seed).fit(X)
        save_run(model, folder, songs, vocab, args.n_top)
        rise = late_rise(model.log_likelihood)

        print(f"K={k}: done in {(time.time() - start) / 60:.1f} min, ", flush=True)
        print(f"K={k}: rise in last quarter of sweeps = {rise:.1%}", flush=True)

if __name__ == "__main__":
    main()

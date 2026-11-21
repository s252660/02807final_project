"""Compute global topic and word importance for Gibbs K=7 across seeds."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "lda_grid_search"
OUT = ROOT / "data" / "lda_feature_importance"
K = 7
SEEDS = range(1, 6)


def main():
    topic_importances = []
    word_importances = None
    vocab = None

    for seed in SEEDS:
        model_path = RUNS / f"k{K}_seed{seed}" / "model.npz"
        if not model_path.exists():
            continue

        with np.load(model_path) as model:
            phi = model["topic_word"]
            theta = model["doc_topic"]
            seed_vocab = model["vocab"]

        if vocab is None:
            vocab = seed_vocab
        elif not np.array_equal(vocab, seed_vocab):
            raise ValueError(f"Vocabulary differs in seed {seed}")

        mean_topic_share = theta.mean(axis=0)
        topic_importances.append(mean_topic_share)

        seed_word_importance = mean_topic_share @ phi
        seed_word_importance /= seed_word_importance.sum()
        if word_importances is None:
            word_importances = seed_word_importance
        else:
            word_importances += seed_word_importance

    if word_importances is None:
        raise FileNotFoundError(f"No K={K} Gibbs seed runs found.")

    n_runs = len(topic_importances)
    mean_topic_share = np.mean(topic_importances, axis=0)
    mean_word_importance = word_importances / n_runs

    topic_table = pd.DataFrame({
        "topic": np.arange(K),
        "importance": mean_topic_share,
        "importance_percent": mean_topic_share * 100,
    }).sort_values("importance", ascending=False)

    word_table = pd.DataFrame({
        "word": vocab,
        "importance": mean_word_importance,
        "importance_percent": mean_word_importance * 100,
    }).sort_values("importance", ascending=False)

    OUT.mkdir(parents=True, exist_ok=True)
    topic_table.to_csv(OUT / "k7_topic_importance_across_songs.csv", index=False)
    word_table.to_csv(OUT / "k7_word_importance_across_songs.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ordered_topics = topic_table.sort_values("topic")
    ax.bar(
        [f"Topic {topic}" for topic in ordered_topics["topic"]],
        ordered_topics["importance_percent"],
        color="#4C78A8",
    )
    ax.set_ylabel("Average share of songs (%)")
    ax.set_xlabel("Topic")
    ax.set_title(f"Gibbs K={K}: overall topic importance across songs\n(mean of seeds 1-{n_runs})")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "k7_topic_importance_across_songs.png", dpi=180)
    plt.close(fig)

    top_words = word_table.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_words["word"][::-1], top_words["importance_percent"][::-1], color="#F58518")
    ax.set_xlabel("Estimated share of all word probability (%)")
    ax.set_title(f"Gibbs K={K}: overall word importance across songs\n(mean of seeds 1-{n_runs})")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "k7_word_importance_across_songs.png", dpi=180)
    plt.close(fig)

    print(topic_table.to_string(index=False))
    print("\nTop 20 words:")
    print(top_words[["word", "importance_percent"]].to_string(index=False))


if __name__ == "__main__":
    main()

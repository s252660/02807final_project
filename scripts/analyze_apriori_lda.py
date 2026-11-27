"""Assign LDA K=7 topics to the top Apriori recommendations for user 0.

Run from the repository root:
    python scripts/analyze_apriori_lda.py

Optionally write the result table:
    python scripts/analyze_apriori_lda.py --output data/apriori_recommendation_topics.csv
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODEL_PATH = (
    DATA / "lda_comparison" / "gibbs_k7_a0.143_b0.1_seed0" / "arrays.npz"
)

# Top-10 output from the Apriori notebook's example user (user 0).
# Replace these rows with another Apriori recommendation output to analyze it.
RECOMMENDATIONS = [
    (2294, "Fleet Foxes", "Sun It Rises", 1.960),
    (618, "Fleet Foxes", "White Winter Hymnal", 1.836),
    (2300, "Fleet Foxes", "Tiger Mountain Peasant Song", 1.709),
    (2303, "Fleet Foxes", "Your Protector", 1.616),
    (2307, "Fleet Foxes", "Quiet Houses", 1.570),
    (2298, "Fleet Foxes", "Meadowlarks", 1.517),
    (2308, "Fleet Foxes", "Oliver James", 1.459),
    (86, "John Mayer", "Heartbreak Warfare", 1.279),
    (569, "Jack Johnson", "Do You Remember", 1.235),
    (848, "Jack Johnson", "Bubble Toes", 1.209),
]

# Topic IDs correspond to the rows of phi/theta in the saved model.
TOPIC_NAMES = {
    0: "Religion/Death",
    1: "Eros Love",
    2: "Rap/profanity",
    3: "Nostalgia home",
    4: "Dance love",
    5: "Aphrodite love",
    6: "Spanish (language)",
}


def load_song_mappings():
    """Map MSD song IDs to the original integer song indices used by LDA."""
    with (DATA / "id_maps.pkl").open("rb") as stream:
        _, all_song_ids = pickle.load(stream)
    return {str(song_id): index for index, song_id in enumerate(all_song_ids)}


def find_recommended_song_ids():
    """Match recommendation artist/title pairs to MSD song IDs."""
    wanted = {(artist, title) for _, artist, title, _ in RECOMMENDATIONS}
    matches = {pair: set() for pair in wanted}

    with (DATA / "unique_tracks.txt").open(
        encoding="utf-8", errors="replace"
    ) as stream:
        for line in stream:
            fields = line.rstrip("\r\n").split("<SEP>", 3)
            if len(fields) != 4:
                continue
            _, song_id, artist, title = fields
            pair = (artist, title)
            if pair in matches:
                matches[pair].add(song_id)

    return matches


def analyze():
    """Return recommendation rows with their dominant K=7 topic mixtures."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Saved LDA model not found: {MODEL_PATH}")

    song_index_by_id = load_song_mappings()
    song_ids_by_name = find_recommended_song_ids()

    with np.load(MODEL_PATH) as model:
        phi = model["phi"]
        theta = model["theta"]
        songs = model["songs"]

    if phi.shape[0] != len(TOPIC_NAMES) or theta.shape[1] != len(TOPIC_NAMES):
        raise ValueError(
            "The saved model dimensions do not match the seven configured topic names."
        )

    model_row_by_song = {int(song): row for row, song in enumerate(songs)}
    rows = []

    for rank, artist, title, score in RECOMMENDATIONS:
        pair = (artist, title)
        song_ids = song_ids_by_name[pair]
        if len(song_ids) != 1:
            raise ValueError(
                f"Expected exactly one MSD song ID for {artist} - {title}; "
                f"found {sorted(song_ids)}."
            )

        song_id = next(iter(song_ids))
        if song_id not in song_index_by_id:
            raise ValueError(f"Song ID {song_id} is missing from data/id_maps.pkl.")

        song_index = song_index_by_id[song_id]
        if song_index not in model_row_by_song:
            raise ValueError(
                f"{artist} - {title} (song index {song_index}) is not in the saved LDA model."
            )

        mixture = theta[model_row_by_song[song_index]]
        topic_id = int(np.argmax(mixture))
        other_topics = np.argsort(mixture)[::-1]
        runner_up_id = int(other_topics[1])

        row = {
            "rank": rank,
            "artist": artist,
            "title": title,
            "apriori_score": score,
            "song_id": song_id,
            "topic_id": topic_id,
            "topic": TOPIC_NAMES[topic_id],
            "topic_share": float(mixture[topic_id]),
            "runner_up_topic": TOPIC_NAMES[runner_up_id],
            "runner_up_share": float(mixture[runner_up_id]),
        }
        row.update(
            {f"topic_{i}_share": float(value) for i, value in enumerate(mixture)}
        )
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional CSV output path (defaults to printing the results only)",
    )
    args = parser.parse_args()

    results = analyze()
    display_columns = [
        "rank",
        "artist",
        "title",
        "apriori_score",
        "topic_id",
        "topic",
        "topic_share",
        "runner_up_topic",
        "runner_up_share",
    ]
    print(results[display_columns].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if args.output:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"\nSaved full topic mixtures to {output_path}")


if __name__ == "__main__":
    main()

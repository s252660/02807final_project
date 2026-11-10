"""Build song-level bags of words for filtered_100_10.npz.

Run from anywhere:
    python scripts/build_song_bow.py

Inputs:
    data/filtered_100_10.npz
    data/id_maps.pkl
    data/unique_tracks.txt
    data/mxm_dataset.db

Outputs:
    data/song_bow.parquet
    data/song_bow_coverage.parquet
    data/filtered_100_10_track_song_mapping.csv
"""

import csv
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


DATA = Path(__file__).resolve().parent.parent / "data"


def load_songs():
    """Return the filtered song IDs and their integer song indexes."""
    filtered = np.load(DATA / "filtered_100_10.npz")
    with (DATA / "id_maps.pkl").open("rb") as stream:
        _, all_song_ids = pickle.load(stream)

    song_indexes = np.unique(filtered["old_song"])
    return pd.DataFrame({
        "song": song_indexes,
        "song_id": np.asarray(all_song_ids, dtype=object)[song_indexes],
    })


def load_tracks(song_ids):
    """Read track IDs for the requested song IDs from unique_tracks.txt."""
    tracks = []
    wanted = set(song_ids)
    with (DATA / "unique_tracks.txt").open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            fields = line.rstrip("\r\n").split("<SEP>", 2)
            if len(fields) >= 2 and fields[1] in wanted:
                tracks.append((fields[1], fields[0]))

    return pd.DataFrame(tracks, columns=["song_id", "track_id"]).drop_duplicates()


def query_bags(songs, tracks):
    """Aggregate sparse lyric counts across all tracks belonging to each song."""
    track_songs = tracks.merge(songs, on="song_id", how="inner")
    with sqlite3.connect(f"file:{DATA / 'mxm_dataset.db'}?mode=ro", uri=True) as con:
        con.execute("CREATE TEMP TABLE wanted_tracks (track_id TEXT PRIMARY KEY, song INTEGER)")
        con.executemany(
            "INSERT INTO wanted_tracks (track_id, song) VALUES (?, ?)",
            track_songs[["track_id", "song"]].itertuples(index=False, name=None),
        )
        bags = pd.read_sql_query(
            """
            SELECT wanted_tracks.song, lyrics.word,
                   SUM(lyrics.count) AS count,
                   COUNT(DISTINCT lyrics.track_id) AS n_tracks
            FROM lyrics
            JOIN wanted_tracks ON wanted_tracks.track_id = lyrics.track_id
            GROUP BY wanted_tracks.song, lyrics.word
            ORDER BY wanted_tracks.song, lyrics.word
            """,
            con,
        )
        track_coverage = pd.read_sql_query(
            """
            SELECT wanted_tracks.song, COUNT(DISTINCT lyrics.track_id) AS n_bow_tracks
            FROM lyrics
            JOIN wanted_tracks ON wanted_tracks.track_id = lyrics.track_id
            GROUP BY wanted_tracks.song
            """,
            con,
        )
        bow_tracks = {
            row[0]
            for row in con.execute(
                "SELECT DISTINCT track_id FROM lyrics "
                "WHERE track_id IN (SELECT track_id FROM wanted_tracks)"
            )
        }

    return bags, track_songs, track_coverage, bow_tracks


def write_track_mapping(track_songs, bow_tracks):
    """Write every filtered song-track pair and its bag-of-words availability."""
    output = DATA / "filtered_100_10_track_song_mapping.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("track_id", "song_id", "has_bow"))
        for row in track_songs.itertuples(index=False):
            writer.writerow((row.track_id, row.song_id, row.track_id in bow_tracks))
    return output


def main():
    songs = load_songs()
    tracks = load_tracks(songs["song_id"])
    bags, track_songs, track_coverage, bow_tracks = query_bags(songs, tracks)

    coverage = songs.merge(
        track_songs.groupby("song").size().rename("n_tracks"),
        on="song",
        how="left",
    ).merge(track_coverage, on="song", how="left")
    coverage["n_tracks"] = coverage["n_tracks"].fillna(0).astype("int64")
    coverage["n_bow_tracks"] = coverage["n_bow_tracks"].fillna(0).astype("int64")
    coverage["has_bow"] = coverage["n_bow_tracks"] > 0

    bags.to_parquet(DATA / "song_bow.parquet", index=False)
    coverage.to_parquet(DATA / "song_bow_coverage.parquet", index=False)
    track_mapping = write_track_mapping(track_songs, bow_tracks)

    print(f"Songs: {len(songs):,}")
    print(f"Song-track mappings: {len(track_songs):,}")
    print(f"Songs with bags of words: {coverage['has_bow'].sum():,}")
    print(f"Bag-of-words rows: {len(bags):,}")
    print(f"Wrote {DATA / 'song_bow.parquet'}")
    print(f"Wrote {DATA / 'song_bow_coverage.parquet'}")
    print(f"Wrote {track_mapping}")


if __name__ == "__main__":
    main()

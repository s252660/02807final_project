"""Keep only the songs with lyrics from filtered_100_10.npz.

Run from anywhere:  python scripts/build_lyrics_npz.py

Inputs:
    data/filtered_100_10.npz                     made by testing_data.ipynb
    data/id_maps.pkl                             made by testing_data.ipynb
    data/filtered_100_10_track_song_mapping.csv  made by scripts/build_song_bow.py

Output:
    data/lyrics_100_10.npz, with the same keys as filtered_100_10.npz:
        user, song, plays   one row per interaction; user and song are numbered again from 0
        old_user            old_user[u] = index of user u in id_maps.pkl (same as in filtered_100_10.npz)
        old_song            old_song[s] = index of song s in id_maps.pkl (same as in filtered_100_10.npz)
        song_id             song_id[s]  = MSD song ID (SO...) of song s

A song is kept when at least one of its tracks has a bag of words (has_bow = True).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def main():
    d = np.load(DATA / "filtered_100_10.npz")
    with open(DATA / "id_maps.pkl", "rb") as f:
        _, all_song_ids = pickle.load(f)
    song_ids = np.asarray(all_song_ids, dtype=object)[d["old_song"]]

    mapping = pd.read_csv(DATA / "filtered_100_10_track_song_mapping.csv")
    lyric_ids = set(mapping.loc[mapping["has_bow"], "song_id"])
    keep_song = np.fromiter((s in lyric_ids for s in song_ids), dtype=bool, count=len(song_ids))

    mask = keep_song[d["song"]]
    kept_users, user = np.unique(d["user"][mask], return_inverse=True)
    kept_songs, song = np.unique(d["song"][mask], return_inverse=True)

    out = DATA / "lyrics_100_10.npz"
    np.savez(out,
             user=user.astype(np.int32), song=song.astype(np.int32), plays=d["plays"][mask],
             old_user=d["old_user"][kept_users], old_song=d["old_song"][kept_songs],
             song_id=song_ids[kept_songs].astype(str))

    print(f"songs: {len(kept_songs):,} of {len(song_ids):,} have lyrics")
    print(f"users: {len(kept_users):,} | interactions: {mask.sum():,} of {len(mask):,}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""Attach metadata to every song in data/filtered_100_10.npz.

Run from anywhere:  python scripts/build_song_metadata.py

Inputs (see README for downloads):
    data/filtered_100_10.npz, data/id_maps.pkl          made by testing_data.ipynb
    data/external/unique_tracks/unique_tracks.txt       track ID (TR...) <-> song ID (SO...)
    data/external/sid_mismatches/sid_mismatches.txt     known wrong song-track pairs, removed
    data/external/track_metadata/track_metadata.db      title, artist, year, duration, ...
    data/external/lastfm_tags/lastfm_tags.db            Last.fm tags per track
    data/external/tagtraum/msd_tagtraum_cd2.cls         one genre per track

Outputs:
    data/song_meta.parquet   one row per song; row i (column `song`) = song i in filtered_100_10.npz
    data/song_tags.parquet   Last.fm tags in long format: song, tag, weight (0-100)

One song ID can have several track IDs (the same song on different releases). Descriptive fields
come from the song's first valid track; tags and genres are combined over all of its valid tracks.
"""

import pickle
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
EXT = DATA / "external"


def connect_ro(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def load_songs():
    """Songs in the filtered dataset, with popularity counts."""
    d = np.load(DATA / "filtered_100_10.npz")
    with open(DATA / "id_maps.pkl", "rb") as f:
        _, all_song_ids = pickle.load(f)
    songs = pd.DataFrame({"song_id": np.asarray(all_song_ids, dtype=object)[d["old_song"]]})
    songs.index.name = "song"
    songs["n_listeners"] = np.bincount(d["song"], minlength=len(songs))
    songs["total_plays"] = np.bincount(d["song"], weights=d["plays"], minlength=len(songs)).astype(np.int64)
    return songs


def load_tracks(songs):
    """(song, track_id) pairs for our songs, without the known mismatches."""
    tracks = pd.read_csv(EXT / "unique_tracks/unique_tracks.txt", sep="<SEP>", engine="python",
                         header=None, names=["track_id", "song_id", "artist", "title"], usecols=[0, 1])
    tracks = tracks[tracks["song_id"].isin(songs["song_id"])]

    pat = re.compile(r"<(SO[A-Z0-9]+) (TR[A-Z0-9]+)>")
    with open(EXT / "sid_mismatches/sid_mismatches.txt") as f:
        bad = pd.DataFrame([m.groups() for line in f if (m := pat.search(line))],
                           columns=["song_id", "track_id"])

    n_before = len(tracks)
    tracks = (tracks.merge(bad, how="left", indicator=True)
                    .query("_merge == 'left_only'").drop(columns="_merge")
                    .merge(songs.reset_index()[["song", "song_id"]], on="song_id"))

    songs["n_tracks"] = tracks.groupby("song").size().reindex(songs.index, fill_value=0)
    songs["mismatched"] = songs["song_id"].isin(bad["song_id"]) & (songs["n_tracks"] == 0)
    print(f"tracks: removed {n_before - len(tracks):,} mismatched pairs; "
          f"{(songs.n_tracks == 0).sum():,} songs left without a valid track")
    return tracks


def add_descriptive(songs, tracks):
    with connect_ro(EXT / "track_metadata/track_metadata.db") as con:
        tm = pd.read_sql_query(
            "SELECT track_id, title, artist_name AS artist, artist_id, release, year, duration, "
            "artist_familiarity, artist_hotttnesss FROM songs", con)
    first = tracks.sort_values(["song", "track_id"]).drop_duplicates("song")[["song", "track_id"]]
    desc = first.merge(tm, on="track_id", how="left").set_index("song")
    desc["year"] = desc["year"].replace(0, np.nan).astype("Int16")    # 0 means unknown
    return songs.join(desc)


def add_genre(songs, tracks):
    genre = pd.read_csv(EXT / "tagtraum/msd_tagtraum_cd2.cls", sep="\t", comment="#", header=None,
                        names=["track_id", "genre", "genre_minority"], usecols=[0, 1])
    per_song = tracks.merge(genre, on="track_id").groupby("song")["genre"].agg(lambda g: g.mode().iloc[0])
    songs["genre"] = per_song
    return songs


def load_tags(songs, tracks):
    """Last.fm tags per song; the highest weight is kept when a song has several tracks."""
    with connect_ro(EXT / "lastfm_tags/lastfm_tags.db") as con:
        tids = pd.read_sql_query("SELECT rowid AS tid_row, tid AS track_id FROM tids", con)
        tids = tids.merge(tracks[["track_id", "song"]], on="track_id")
        con.execute("CREATE TEMP TABLE want (tid_row INTEGER PRIMARY KEY)")
        con.executemany("INSERT INTO want VALUES (?)", ((int(r),) for r in tids["tid_row"]))
        tt = pd.read_sql_query(
            "SELECT tid_tag.tid AS tid_row, tags.tag AS tag, tid_tag.val AS weight FROM tid_tag "
            "JOIN want ON want.tid_row = tid_tag.tid JOIN tags ON tags.rowid = tid_tag.tag", con)

    song_tags = (tt.merge(tids[["tid_row", "song"]], on="tid_row")
                   .groupby(["song", "tag"], as_index=False)["weight"].max()
                   .sort_values(["song", "weight"], ascending=[True, False], ignore_index=True))
    songs["n_tags"] = song_tags.groupby("song").size().reindex(songs.index, fill_value=0)
    songs["top_tags"] = song_tags.groupby("song")["tag"].agg(lambda t: ", ".join(t[:5]))
    return song_tags


def print_coverage(songs, song_tags):
    fields = {"title / artist": songs["title"].notna(), "year": songs["year"].notna(),
              "genre (tagtraum)": songs["genre"].notna(), "Last.fm tags": songs["n_tags"] > 0}
    w = songs["total_plays"] / songs["total_plays"].sum()
    print(f"\n{len(songs):,} songs, {len(song_tags):,} (song, tag) pairs, "
          f"{song_tags['tag'].nunique():,} distinct tags\n")
    print(f"{'field':<18}{'songs':>9}{'% of songs':>12}{'% of plays':>12}")
    for name, m in fields.items():
        print(f"{name:<18}{m.sum():>9,}{m.mean():>12.1%}{w[m].sum():>12.1%}")


def main():
    songs = load_songs()
    tracks = load_tracks(songs)
    songs = add_descriptive(songs, tracks)
    songs = add_genre(songs, tracks)
    song_tags = load_tags(songs, tracks)

    songs = songs[["song_id", "track_id", "n_tracks", "mismatched", "title", "artist", "artist_id",
                   "release", "year", "duration", "artist_familiarity", "artist_hotttnesss", "genre",
                   "n_tags", "top_tags", "n_listeners", "total_plays"]]
    songs.reset_index().to_parquet(DATA / "song_meta.parquet", index=False)
    song_tags.to_parquet(DATA / "song_tags.parquet", index=False)
    print_coverage(songs, song_tags)
    print(f"\nwrote {DATA / 'song_meta.parquet'} and {DATA / 'song_tags.parquet'}")


if __name__ == "__main__":
    main()
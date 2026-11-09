# 02807final_project

## Getting the data

The data is not in git (`data/` is ignored). Download it once into `data/` in the repo root. The
commands below are run from the repo root and download about 2 GB in total.

### 1. Play counts (Taste Profile / Echo Nest triplets)

`train_triplets.txt` has one row per `user  song  play_count` (48M rows, ~3 GB unzipped).

```bash
mkdir -p data
curl -L -o data/train_triplets.txt.zip http://millionsongdataset.com/sites/default/files/challenge/train_triplets.txt.zip
unzip data/train_triplets.txt.zip -d data/ && rm data/train_triplets.txt.zip
```

### 2. Song metadata

Used by `scripts/build_song_metadata.py`. Each file goes into its own folder under `data/external/`.

```bash
B=http://millionsongdataset.com/sites/default/files
mkdir -p data/external/{unique_tracks,sid_mismatches,track_metadata,lastfm_tags,tagtraum}

curl -L -o data/external/unique_tracks/unique_tracks.txt   $B/AdditionalFiles/unique_tracks.txt
curl -L -o data/external/sid_mismatches/sid_mismatches.txt $B/tasteprofile/sid_mismatches.txt
curl -L -o data/external/track_metadata/track_metadata.db  $B/AdditionalFiles/track_metadata.db
curl -L -o data/external/lastfm_tags/lastfm_tags.db        $B/lastfm/lastfm_tags.db
curl -L -o data/external/tagtraum/msd_tagtraum_cd2.cls.zip https://www.tagtraum.com/genres/msd_tagtraum_cd2.cls.zip
unzip data/external/tagtraum/msd_tagtraum_cd2.cls.zip -d data/external/tagtraum/
```

| File | What it is |
|---|---|
| `unique_tracks.txt` | Maps track IDs (`TR…`) to the song IDs (`SO…`) used in the triplets |
| `sid_mismatches.txt` | Song–track pairs known to be matched wrongly; these are removed |
| `track_metadata.db` | Title, artist, album, year, duration, artist popularity (SQLite) |
| `lastfm_tags.db` | Last.fm user tags per track (SQLite) |
| `msd_tagtraum_cd2.cls` | One genre label per track (tagtraum) |

The two `.db` files are 600–750 MB each and the server is slow, so expect them to take 15–20 minutes.

### 3. Generated files

Run these in order to build the rest of `data/`:

1. `testing_data.ipynb` creates `triplets_int.npz`, `id_maps.pkl` and `filtered_100_10.npz`.
2. `python scripts/build_song_metadata.py` (~10 s) creates:
   - `song_meta.parquet`: one row per song. Row `song = i` is song `i` in `filtered_100_10.npz`. It has
     title, artist, album, year, duration, artist popularity, genre, the top 5 Last.fm tags, listeners and plays.
   - `song_tags.parquet`: all Last.fm tags per song, as `song, tag, weight` (weight 0–100).

```python
import pandas as pd
songs = pd.read_parquet("data/song_meta.parquet")
tags = pd.read_parquet("data/song_tags.parquet")
```

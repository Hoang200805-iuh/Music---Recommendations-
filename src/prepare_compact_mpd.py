"""Export cleaned MPD Parquet to the existing ALS/Hybrid input contract.

python -m src.prepare_compact_mpd --input music_clean_compact --out hdfs_sim/music
SQLite sorts (pid, pos) on disk; the full interaction table never enters RAM.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import tempfile
import time

from recommender.evaluation.split_playlists import assign_splits, write_assignment
from recommender.hybrid.playlist_titles import normalize_title

TABLES = {
    "playlists": ("pid", "name"),
    "tracks": ("track_id", "track_name", "artist_name"),
    "track_mapping": ("track_id", "track_uri"),
    "playlist_tracks": ("pid", "pos", "track_id"),
}
OUTPUTS = {
    "flat": "processed/playlist_tracks_flat.tsv",
    "split": "processed/playlist_split.tsv",
    "titles": "processed/playlist_titles.tsv",
    "popularity": "analytics/track_popularity_train/part-00000",
}


def inspect_source(source):
    import pyarrow.parquet as pq
    source = Path(source).resolve()
    spec = {"format": 1, "source": str(source), "files": {}}
    for table, columns in TABLES.items():
        files = sorted((source / table).glob("*.parquet"))
        if not files:
            raise ValueError(f"No Parquet files in {source / table}")
        details = []
        for path in files:
            parquet = pq.ParquetFile(path)
            if not set(columns) <= set(parquet.schema_arrow.names):
                raise ValueError(f"Missing {columns} in {path}")
            stat = path.stat()
            details.append({"name": path.name, "bytes": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns, "rows": parquet.metadata.num_rows})
        spec["files"][table] = details
    return spec


def batches(source, table, batch_size=65536):
    import pyarrow.parquet as pq
    for path in sorted((Path(source) / table).glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=list(TABLES[table])):
            yield zip(*(column.to_pylist() for column in batch.columns))


def clean_text(value):
    return (value or "").translate(str.maketrans({"\t": " ", "\r": " ", "\n": " "}))


def prepare(source, output, work_dir, seed=42):
    source, output, work_dir = Path(source).resolve(), Path(output).resolve(), Path(work_dir).resolve()
    spec = {**inspect_source(source), "seed": seed, "ratios": [0.70, 0.15, 0.15]}
    manifest_path = output / "processed/compact_mpd_manifest.json"
    targets = {name: output / relative for name, relative in OUTPUTS.items()}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("input") == spec and all(
            path.is_file() and manifest.get("outputs", {}).get(name) ==
            {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for name, path in targets.items()
        ):
            print("[prepare] Existing export matches source; reusing inputs.", flush=True)
            return manifest
        raise ValueError("Existing export differs; choose another --out directory to preserve its data/models")
    if any(path.exists() for path in targets.values()):
        raise ValueError("Input files already exist without a matching manifest; choose another --out")
    work_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="compact-mpd-", dir=work_dir) as directory:
        stage = Path(directory)
        database = sqlite3.connect(stage / "sort.sqlite")
        previous_temp = database.execute("PRAGMA temp_store_directory").fetchone()
        try:
            # SQLite's external index sort must also stay on the data drive.
            escaped = str(stage).replace("'", "''")
            database.execute(f"PRAGMA temp_store_directory='{escaped}'")
            database.execute("PRAGMA temp_store=FILE")
            # The database is disposable staging data; publish files only on success.
            database.execute("PRAGMA journal_mode=OFF")
            database.execute("PRAGMA synchronous=OFF")
            database.execute("PRAGMA cache_size=-65536")
            database.execute("PRAGMA foreign_keys=ON")
            database.executescript("""
                CREATE TABLE playlists(pid INTEGER PRIMARY KEY, name TEXT NOT NULL);
                CREATE TABLE tracks(track_id INTEGER PRIMARY KEY, name TEXT NOT NULL,
                                    artist TEXT NOT NULL, uri TEXT);
                CREATE TABLE mapping(track_id INTEGER PRIMARY KEY REFERENCES tracks(track_id), uri TEXT NOT NULL UNIQUE);
                CREATE TABLE interactions(pid INTEGER NOT NULL REFERENCES playlists(pid),
                    pos INTEGER NOT NULL CHECK(pos >= 0), track_id INTEGER NOT NULL REFERENCES tracks(track_id));
            """)
            for rows in batches(source, "playlists"):
                database.executemany("INSERT INTO playlists VALUES (?, ?)",
                                     ((pid, normalize_title(name or "")) for pid, name in rows))
            for rows in batches(source, "tracks"):
                database.executemany("INSERT INTO tracks VALUES (?, ?, ?, NULL)",
                                     ((item, clean_text(name), clean_text(artist)) for item, name, artist in rows))
            for rows in batches(source, "track_mapping"):
                database.executemany("INSERT INTO mapping VALUES (?, ?)", rows)
            database.execute("UPDATE tracks SET uri=(SELECT uri FROM mapping WHERE mapping.track_id=tracks.track_id)")
            if database.execute("SELECT 1 FROM tracks WHERE uri IS NULL OR uri='' LIMIT 1").fetchone():
                raise ValueError("Missing URI in track_mapping")
            database.execute("DROP TABLE mapping")
            assignment = assign_splits((row[0] for row in database.execute("SELECT pid FROM playlists")), seed=seed)
            counts_split = dict(Counter(assignment.values()))
            write_assignment(assignment, stage / "split.tsv")
            with (stage / "titles.tsv").open("w", encoding="utf-8", newline="\n") as stream:
                for pid, title in database.execute("SELECT pid, name FROM playlists WHERE name<>'' ORDER BY pid"):
                    stream.write(f"{pid}\t{title}\n")
            print(f"[prepare] Playlist split: {counts_split}", flush=True)
            n_rows = 0
            last = time.monotonic()
            for rows in batches(source, "playlist_tracks"):
                before = database.total_changes
                database.executemany("INSERT INTO interactions VALUES (?, ?, ?)", rows)
                n_rows += database.total_changes - before
                if time.monotonic() - last >= 10:
                    print(f"[prepare] Imported {n_rows:,} interactions | {time.monotonic()-start:.0f}s", flush=True)
                    last = time.monotonic()
            database.commit()
            print(f"[prepare] Sorting {n_rows:,} interactions by playlist position on disk...", flush=True)
            database.execute("CREATE UNIQUE INDEX playlist_order ON interactions(pid, pos)")
            if database.execute("SELECT 1 FROM playlists p WHERE NOT EXISTS "
                                "(SELECT 1 FROM interactions i WHERE i.pid=p.pid) LIMIT 1").fetchone():
                raise ValueError("Playlist has no interactions; cannot preserve split/evaluation semantics")
            popularity = Counter()
            cursor = database.execute("SELECT i.pid, i.track_id, t.uri, t.name, t.artist "
                                      "FROM interactions i INDEXED BY playlist_order "
                                      "JOIN tracks t ON t.track_id=i.track_id ORDER BY i.pid, i.pos")
            exported = 0
            last = time.monotonic()
            with (stage / "flat.tsv").open("w", encoding="utf-8", newline="\n", buffering=8 * 1024**2) as stream:
                while rows := cursor.fetchmany(32768):
                    for pid, item, uri, name, artist in rows:
                        stream.write(f"{pid}\t{uri}\t{name}\t{artist}\n")
                        if assignment[pid] == "train":
                            popularity[item] += 1
                    exported += len(rows)
                    if time.monotonic() - last >= 10:
                        print(f"[prepare] Exported {exported:,}/{n_rows:,} interactions", flush=True)
                        last = time.monotonic()
            if exported != n_rows:
                raise ValueError("Interaction rows were lost during export")
            with (stage / "popularity.tsv").open("w", encoding="utf-8", newline="\n") as stream:
                for item, uri, name, artist in database.execute("SELECT track_id, uri, name, artist FROM tracks ORDER BY track_id"):
                    if item in popularity:
                        stream.write(f"{uri}\t{popularity[item]}\t{name}\t{artist}\n")
            # Detect source changes before publishing a complete, coherent export.
            if {**inspect_source(source), "seed": seed, "ratios": [0.70, 0.15, 0.15]} != spec:
                raise ValueError("Parquet inputs changed during export")
            for name, path in targets.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                (stage / f"{name}.tsv").replace(path)
            manifest = {"input": spec, "n_playlists": len(assignment), "split_counts": counts_split,
                        "interaction_rows": exported, "train_tracks": len(popularity),
                        "outputs": {name: {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                                    for name, path in targets.items()}}
            temporary_manifest = manifest_path.with_suffix(".json.tmp")
            temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            temporary_manifest.replace(manifest_path)
        finally:
            previous = previous_temp[0].replace("'", "''") if previous_temp else ""
            database.execute(f"PRAGMA temp_store_directory='{previous}'")
            database.close()
    print(f"[prepare] READY: {output} | {time.monotonic()-start:.0f}s", flush=True)
    return manifest


def main():
    import sys
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("music_clean_compact"))
    parser.add_argument("--out", type=Path, default=Path("hdfs_sim/music"))
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/prepare_compact"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.input, args.out, args.work_dir, args.seed)


if __name__ == "__main__":
    main()

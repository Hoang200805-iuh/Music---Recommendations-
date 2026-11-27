"""Kiểm tra notebook và ETL thực tế từ JSON đến Parquet."""
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_notebook_etl_code_compiles():
    path = ROOT / "notebooks/prepare-mdp.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(path), "exec")


def _playlist(pid, track_ids):
    return {
        "pid": pid, "name": "   Sample   mix ", "num_tracks": 999, "num_followers": -1,
        "tracks": [
            {"pos": pos, "track_uri": f"spotify:track:t{track:02d}",
             "track_name": f"  Track   {track} ", "artist_name": "Artist"}
            for pos, track in enumerate(track_ids)
        ],
    }


@pytest.mark.skipif(os.environ.get("RUN_SPARK_TESTS") != "1",
                    reason="Đặt RUN_SPARK_TESTS=1 để kiểm tra ETL với Spark và Java")
def test_spark_etl_cli_writes_cleaned_parquet(tmp_path):
    pytest.importorskip("pyspark")
    parquet = pytest.importorskip("pyarrow.parquet")

    raw = tmp_path / "raw"
    nested = raw / "nested"
    nested.mkdir(parents=True)
    output = tmp_path / "clean"
    first = _playlist(0, [*range(1, 11), 1])
    first["tracks"] += [first["tracks"][0], {"pos": -1, "track_uri": "spotify:track:bad"}]
    playlists = [
        first,
        _playlist(1, range(1, 10)),  # Jaccard đúng 0,9 với playlist 0.
        _playlist(2, range(1, 12)),  # Jaccard 10/11 > 0,9 với playlist 0.
        _playlist(3, reversed(range(1, 11))),  # Cùng tập bài với playlist 0.
        _playlist(4, [12]),  # Dưới ngưỡng số bài.
    ]
    (nested / "mpd.slice.0-4.json").write_text(
        json.dumps({"playlists": playlists}, indent=2), encoding="utf-8")
    (raw / "ignored.json").write_text("invalid JSON", encoding="utf-8")

    env = dict(os.environ, SPARK_LOCAL_IP="127.0.0.1",
               PYSPARK_PYTHON=sys.executable, PYTHONIOENCODING="utf-8")
    command = [sys.executable, "-m", "spark.pipeline", "--input", raw.as_posix(),
               "--output", output.as_posix(), "--master", "local[2]", "--shuffle-partitions", "2",
               "--min-track-playlists", "1", "--export-streaming"]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr

    cleaned = {row["pid"]: row for row in parquet.read_table(output / "playlists").to_pylist()}
    assert set(cleaned) == {0, 1}
    assert cleaned[0] == {"pid": 0, "name": "Sample mix", "num_tracks": 11, "num_followers": 0}
    assert cleaned[1]["num_tracks"] == 9
    tracks = parquet.read_table(output / "tracks").to_pylist()
    assert len(tracks) == 10
    assert {row["track_id"]: row["track_name"] for row in tracks}[1] == "Track 1"
    mapping = parquet.read_table(output / "track_mapping").to_pylist()
    assert {row["track_id"]: row["track_uri"] for row in mapping} == {
        track: f"spotify:track:t{track:02d}" for track in range(1, 11)}
    positions = parquet.read_table(output / "playlist_tracks").to_pylist()
    assert sorted(row["pos"] for row in positions if row["pid"] == 0 and row["track_id"] == 1) == [0, 10]
    audit = parquet.read_table(output / "dedup_audit").to_pylist()
    assert {(row["pid"], row["matched_lower_pid"], row["reason"]) for row in audit} == {
        (2, 0, "near"), (3, 0, "exact")}

    report = json.loads((output / "cleaning_report.json").read_text(encoding="utf-8"))
    assert report["checks_passed"] and report["output_playlist_track_rows"] == 20
    assert report["slice_files"] == 1
    assert report["invalid_track_rows_removed"] == report["duplicate_position_rows_removed"] == 1
    assert json.loads((output / "_SUCCESS").read_text(encoding="utf-8")) == {"ok": True}
    staging = output / "_staging"
    assert not staging.exists() or not list(staging.iterdir())

    streaming = []
    for part in (output / "streaming_tracks_jsonl").glob("part-*.gz"):
        with gzip.open(part, "rt", encoding="utf-8") as stream:
            streaming.extend(json.loads(line) for line in stream)
    assert len(streaming) == 20
    assert {"pid": 0, "pos": 10, "track_id": 1, "track_uri": "spotify:track:t01"} in streaming

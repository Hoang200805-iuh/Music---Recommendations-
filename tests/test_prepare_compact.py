import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from recommender.evaluation.evaluate_models import load_train
from recommender.evaluation.split_playlists import assign_splits, read_assignment
from recommender.popularity.popularity_recommender import read_mapreduce_counts
from src.prepare_compact_mpd import OUTPUTS, prepare


def parquet_fixture(tmp_path):
    source = tmp_path / "parquet"
    tables = {
        "playlists": [{"pid": p, "name": "ROCK!!"} for p in range(10)],
        "tracks": [{"track_id": 100, "track_name": "A\tB\r\nC", "artist_name": "Artist"},
                   {"track_id": 500, "track_name": "Other", "artist_name": "Artist"}],
        "track_mapping": [{"track_id": 500, "track_uri": "uri:b"}, {"track_id": 100, "track_uri": "uri:a"}],
        "playlist_tracks": [{"pid": p, "pos": pos, "track_id": item}
                            for p in reversed(range(10)) for pos, item in [(2, 100), (0, 100), (1, 500)]],
    }
    for name, rows in tables.items():
        directory = source / name
        directory.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), directory / "part.parquet")
    return source


def test_order_uri_mapping_split_and_train_only_popularity(tmp_path):
    source = parquet_fixture(tmp_path)
    out = tmp_path / "music"
    report = prepare(source, out, tmp_path / "work")
    split = read_assignment(out / OUTPUTS["split"])
    assert split == assign_splits(range(10))
    assert report["split_counts"] == {"train": 7, "val": 1, "test": 2}
    rows = (out / OUTPUTS["flat"]).read_text(encoding="utf-8").splitlines()
    assert len(rows) == 30 and all(len(row.split("\t")) == 4 for row in rows)
    assert [r.split("\t")[1] for r in rows[:3]] == ["uri:a", "uri:b", "uri:a"]
    assert read_mapreduce_counts(out / OUTPUTS["popularity"]) == {"uri:a": 14, "uri:b": 7}
    train = load_train(out / OUTPUTS["flat"], split)
    assert train.X.shape == (7, 2) and train.X.nnz == 14
    assert "0\trock\n" in (out / OUTPUTS["titles"]).read_text()
    assert prepare(source, out, tmp_path / "work") == report
    assert not list((tmp_path / "work").iterdir())


def test_orphan_track_is_rejected_before_publishing(tmp_path):
    import sqlite3
    source = parquet_fixture(tmp_path)
    pq.write_table(pa.Table.from_pylist([{"pid": 0, "pos": 0, "track_id": 9999}]),
                   source / "playlist_tracks/part.parquet")
    out = tmp_path / "music"
    with pytest.raises(sqlite3.IntegrityError):
        prepare(source, out, tmp_path / "work")
    assert not (out / OUTPUTS["flat"]).exists()


def test_modified_source_does_not_overwrite_export(tmp_path):
    source = parquet_fixture(tmp_path)
    out = tmp_path / "music"
    prepare(source, out, tmp_path / "work")
    path = out / OUTPUTS["flat"]
    before = path.read_bytes()
    with (source / "tracks/part.parquet").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises((ValueError, pa.ArrowInvalid)):
        prepare(source, out, tmp_path / "work")
    assert path.read_bytes() == before

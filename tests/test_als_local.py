import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from recommender.als.train_als import training_digest
from recommender.evaluation import evaluate_models
from recommender.evaluation.data_cache import load_eval_cached, load_train_cached

ROOT = Path(__file__).resolve().parents[1]
RUNNER = runpy.run_path(str(ROOT / "scripts/run_als_local.py"))


def fixture_data(tmp_path):
    flat = tmp_path / "flat.tsv"
    flat.write_text("0\ta\tA\tAA\n0\tb\tB\tBB\n1\tb\tB\tBB\n1\tc\tC\tCC\n"
                    "2\ta\tA\tAA\n2\tnew\tNew\tDD\n3\tc\tC\tCC\n", encoding="utf-8")
    return flat, {0: "train", 1: "train", 2: "val", 3: "test"}


def test_cache_skips_train_tsv_and_keeps_exact_item_ids(tmp_path, monkeypatch):
    flat, assignment = fixture_data(tmp_path)
    first = load_train_cached(flat, assignment, tmp_path / "cache")
    monkeypatch.setattr(evaluate_models, "load_train", lambda *args: pytest.fail("Không đọc lại TSV đã cache"))
    second = load_train_cached(flat, assignment, tmp_path / "cache")
    assert training_digest(first) == training_digest(second)
    assert first.vocab == second.vocab and first.artist_ids == second.artist_ids
    assert np.array_equal(first.item_artists, second.item_artists)


def test_cache_keeps_unseen_tracks_artists_and_order(tmp_path, monkeypatch):
    flat, assignment = fixture_data(tmp_path)
    train = load_train_cached(flat, assignment)
    first_artists, second_artists = dict(train.artist_ids), dict(train.artist_ids)
    first, unseen = load_eval_cached(flat, assignment, ["val"], train.vocab, first_artists, tmp_path / "cache")
    monkeypatch.setattr(evaluate_models, "load_eval_playlists", lambda *args: pytest.fail("Không đọc lại eval TSV"))
    second, cached_unseen = load_eval_cached(flat, assignment, ["val"], train.vocab, second_artists, tmp_path / "cache")
    assert list(first["val"][2]) == list(second["val"][2]) == [0, -1]
    assert unseen == cached_unseen and first_artists == second_artists


def test_staging_resumes_partial_copy_and_preserves_source_mtime(tmp_path):
    source, target = tmp_path / "original.tsv", tmp_path / "working/input.tsv"
    data = b"0123456789" * 10000
    source.write_bytes(data)
    target.parent.mkdir()
    target.with_name(target.name + ".part").write_bytes(data[:1234])
    stat = source.stat()
    target.with_name(target.name + ".copy.json").write_text(json.dumps(
        {"source": str(source.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}))
    result = RUNNER["stage_file"](source, target)
    assert result.read_bytes() == data
    assert result.stat().st_mtime_ns == source.stat().st_mtime_ns
    before = result.stat().st_mtime_ns
    RUNNER["stage_file"](source, target)
    assert result.stat().st_mtime_ns == before


def test_staging_discards_partial_copy_if_source_changed(tmp_path):
    source, target = tmp_path / "original.tsv", tmp_path / "input.tsv"
    source.write_bytes(b"new data")
    target.with_name(target.name + ".part").write_bytes(b"old")
    target.with_name(target.name + ".copy.json").write_text(json.dumps({"source": "old", "size": 3, "mtime_ns": 0}))
    RUNNER["stage_file"](source, target)
    assert target.read_bytes() == b"new data"


def test_local_snapshot_contains_all_runtime_modules_without_notebook(tmp_path):
    source = RUNNER["stage_source"](tmp_path)
    for relative in RUNNER["SOURCES"]:
        assert (source / "recommender" / relative).read_bytes() == (ROOT / "recommender" / relative).read_bytes()
    assert (source / "recommender/evaluation/data_cache.py").is_file()
    assert not (source / "notebooks").exists()


def test_direct_inputs_use_original_files_without_copying(tmp_path):
    directory, work = tmp_path / "project/hdfs_sim/music", tmp_path / "project/outputs/als_local"
    for relative in RUNNER["INPUTS"].values():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original")
    files = RUNNER["prepare_inputs"](directory, work)
    assert all(path == (directory / RUNNER["INPUTS"][kind]).resolve() for kind, path in files.items())
    assert not (work / "data").exists()
    assert RUNNER["default_work_dir"]() == ROOT / "outputs/als_local"


def test_local_evaluation_writes_new_report_when_existing_csv_is_locked(tmp_path, monkeypatch):
    inputs, work = tmp_path / "inputs", tmp_path / "work"
    for relative in RUNNER["INPUTS"].values():
        path = inputs / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("input")
    target = work / "reports/results_with_als.csv"
    target.parent.mkdir(parents=True)
    target.write_text("old CSV open in Excel")
    replace = Path.replace

    def locked(self, other):
        if Path(other) == target:
            raise PermissionError("Excel lock")
        return replace(self, other)

    def fake_run(command, *args):
        if "recommender.evaluation.evaluate_models" in command:
            output = Path(command[command.index("--out") + 1])
            assert output != target
            output.write_text("model,ndcg\nals,0.25\n")

    namespace = RUNNER["main"].__globals__
    monkeypatch.setitem(namespace, "configure_environment", lambda: None)
    monkeypatch.setitem(namespace, "report_memory", lambda: None)
    monkeypatch.setitem(namespace, "run", fake_run)
    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", ["run_als_local.py", "--input-dir", str(inputs), "--work-dir", str(work)])
    RUNNER["main"]()
    assert target.read_text() == "old CSV open in Excel"
    assert (work / "reports/results_with_als.latest.csv").read_text() == "model,ndcg\nals,0.25\n"

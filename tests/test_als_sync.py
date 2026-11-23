"""Entrypoint parity, algorithm identity and conservative legacy migration."""
import csv
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from recommender import als_tuning_state as state, local_als_runtime as runtime
from recommender.als import tune_als
from recommender.als.recommend_als import ALSRecommender
from recommender.als.train_als import save_bundle
from recommender.evaluation import evaluate_models as evaluation
from recommender.evaluation.data_cache import write_json

ROOT = Path(__file__).resolve().parents[1]
RUNNER = runpy.run_path(str(ROOT / "scripts/run_als_local.py"))


def args_for(tmp_path):
    flat, popularity = tmp_path / "flat.tsv", tmp_path / "popularity"
    flat.write_text("0\ta\tA\tArtist\n")
    popularity.write_text("a\t1\n")
    return SimpleNamespace(flat=flat, popularity=popularity, max_playlists=2, num_blocks=10,
                           checkpoint_dir=tmp_path / "checkpoints", model_out=tmp_path / "model")


def rows_for(score=0.2):
    return [{"k": k, "model": "als", "split": "val", "n_playlists": 2,
             "seconds": 1, "train_seconds": 2,
             **{metric: score for metric in evaluation.METRICS},
             **{f"{metric}_ci95": 0.01 for metric in evaluation.METRICS}} for k in (1, 5, 25)]


def legacy_bundle(tmp_path):
    args = args_for(tmp_path)
    assignment, train_sha, vocab = {0: "train"}, "train-sha", {b"a": 0, b"b": 1}
    manifest = json.loads((ROOT / "recommender/als_checkpoint_compat.json").read_text())
    legacy = manifest["versions"][0]
    assert legacy["algorithm_sha256"] == state.algorithm_digest()
    old = state.signature_hash(state.signature_spec(args, assignment, train_sha, legacy["legacy_code_sha256"]))
    params = {"rank": 20, "regParam": 0.1, "alpha": 10.0}
    selection = {"split": "val", "metric": "mean_ndcg@10", "score": 0.2,
                 "seed_sizes": [1, 5, 25], "max_playlists": 2, "seed": 42, "run_signature": old}
    model = ALSRecommender(np.ones((2, 20), dtype=np.float32), None)
    save_bundle(args.model_out, model, vocab, params, selection=selection, train_sha256=train_sha)
    path = state.checkpoint_path(args.checkpoint_dir, old, params)
    write_json(path, {"signature": old, "params": params, "rows": rows_for()})
    new = state.run_signature(args, assignment, train_sha)
    return args, assignment, train_sha, vocab, old, new, path


def test_entrypoints_have_same_algorithm_identity_and_ignore_unrelated_scripts(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    before = state.run_signature(args, {0: "train"}, "train")
    source = RUNNER["stage_source"](tmp_path / "portable") / "recommender"
    assert state.algorithm_digest(source) == state.algorithm_digest()
    for relative in ("evaluation/tune_cooccurrence.py", "evaluation/tune_hybrid.py", "report_files.py", "local_als_runtime.py"):
        (source / relative).write_text("unrelated execution-only revision")
    assert state.algorithm_digest(source) == state.algorithm_digest()
    digest = state.algorithm_digest
    monkeypatch.setattr(state, "algorithm_digest", lambda: digest(source))
    assert state.run_signature(args, {0: "train"}, "train") == before
    path = source / "evaluation/ndcg_at_k.py"
    path.write_text(path.read_text(encoding="utf-8") + "\nALGORITHM_CHANGED = True\n", encoding="utf-8")
    assert state.run_signature(args, {0: "train"}, "train") != before


@pytest.mark.parametrize("change", ["blocks", "sample", "split", "train", "popularity"])
def test_signature_changes_when_experiment_changes(tmp_path, change):
    args = args_for(tmp_path)
    assignment, train = {0: "train"}, "train"
    before = state.run_signature(args, assignment, train)
    if change == "blocks":
        args.num_blocks = 2
    elif change == "sample":
        args.max_playlists = 3
    elif change == "split":
        assignment = {0: "val"}
    elif change == "train":
        train = "other-train"
    else:
        args.popularity.write_text("a\t123456\n")
    assert state.run_signature(args, assignment, train) != before


def test_direct_cli_delegates_and_keeps_optional_path_arguments(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runtime, "coordinate", calls.append)
    argv = ["--flat", "flat", "--split-file", "split", "--popularity", "pop",
            "--out", str(tmp_path / "val.csv"), "--model-out", str(tmp_path / "bundle")]
    tune_als.main(argv)
    assert len(calls) == 1
    args = calls[0]
    assert args.master == "local[*]" and args.max_playlists == 1000 and args.num_blocks == 10
    assert args.cache_dir == tmp_path / "bundle.work/cache"
    assert args.checkpoint_dir.is_dir() and args.temp_dir.is_dir()
    calls.clear()
    tune_als.main(argv + ["--cache-dir", str(tmp_path / "custom-cache"),
                          "--checkpoint-dir", str(tmp_path / "custom-cp"),
                          "--temp-dir", str(tmp_path / "custom-tmp"), "--master", "local[2]"])
    assert calls[0].cache_dir == tmp_path / "custom-cache" and calls[0].master == "local[2]"


def test_legacy_migration_backs_up_and_never_rewrites_factors(tmp_path):
    args, assignment, train, vocab, old, new, path = legacy_bundle(tmp_path)
    metadata = args.model_out / "metadata.json"
    factors = args.model_out / "item_factors.npy"
    original_metadata, original_trial, original_factors = metadata.read_bytes(), path.read_bytes(), factors.read_bytes()
    factor_mtime = factors.stat().st_mtime_ns
    assert state.migrate_legacy(args, assignment, train, vocab, new, {k: 2 for k in (1, 5, 25)})
    assert path.read_bytes() == original_trial
    assert factors.read_bytes() == original_factors and factors.stat().st_mtime_ns == factor_mtime
    backup, = (args.checkpoint_dir / "legacy_backups").iterdir()
    assert (backup / "metadata.json").read_bytes() == original_metadata
    assert (backup / path.name).read_bytes() == original_trial
    payload = json.loads(path.read_text())
    migrated = state.checkpoint_path(args.checkpoint_dir, new, payload["params"])
    assert json.loads(migrated.read_text())["rows"] == payload["rows"]
    assert runtime.valid_best(args, new, train, vocab) == pytest.approx(0.2)
    assert not state.migrate_legacy(args, assignment, train, vocab, new, {k: 2 for k in (1, 5, 25)})
    assert len(list((args.checkpoint_dir / "legacy_backups").iterdir())) == 1


@pytest.mark.parametrize("damage", ["blocks", "train", "sample", "algorithm", "factors", "trial", "best-score", "case-count"])
def test_migration_refuses_unverified_results_without_writing(tmp_path, monkeypatch, damage):
    args, assignment, train, vocab, old, new, path = legacy_bundle(tmp_path)
    counts = {k: 2 for k in (1, 5, 25)}
    if damage == "blocks":
        args.num_blocks = 2
    elif damage == "train":
        train = "changed"
    elif damage == "sample":
        args.max_playlists = 3
    elif damage == "algorithm":
        monkeypatch.setattr(state, "algorithm_digest", lambda: "changed")
    elif damage == "factors":
        factors = args.model_out / "item_factors.npy"
        content = factors.read_bytes()
        factors.write_bytes(content[:-4] + b"xxxx")
    elif damage == "trial":
        payload = json.loads(path.read_text())
        payload["rows"][0]["ndcg@10"] = float("nan")
        write_json(path, payload)
    elif damage == "best-score":
        metadata = json.loads((args.model_out / "metadata.json").read_text())
        metadata["selection"]["score"] = 0.9
        write_json(args.model_out / "metadata.json", metadata)
    else:
        counts[25] = 1
    before = (args.model_out / "metadata.json").read_bytes()
    assert not state.migrate_legacy(args, assignment, train, vocab, new, counts)
    assert (args.model_out / "metadata.json").read_bytes() == before
    assert not (args.checkpoint_dir / "legacy_backups").exists()
    assert not (args.checkpoint_dir / new).exists()


def test_partial_report_handles_excel_lock(tmp_path, monkeypatch):
    target = tmp_path / "val.partial.csv"
    target.write_text("Excel open file")
    replace = Path.replace

    def locked(self, other):
        if Path(other) == target:
            raise PermissionError("Excel lock")
        return replace(self, other)

    monkeypatch.setattr(Path, "replace", locked)
    actual = tune_als.write_partial_report(target, [({"rank": 20}, rows_for())])
    assert actual == tmp_path / "val.partial.latest.csv"
    assert target.read_text() == "Excel open file"
    with actual.open() as stream:
        assert len(list(csv.DictReader(stream))) == 3


def test_interrupted_migration_leaves_original_metadata_and_trials_active(tmp_path, monkeypatch):
    args, assignment, train, vocab, old, new, path = legacy_bundle(tmp_path)
    before_metadata = (args.model_out / "metadata.json").read_bytes()
    before_trial = path.read_bytes()
    write = state.write_json

    def interrupted(target, payload):
        if Path(target).parent == args.checkpoint_dir / new:
            raise OSError("Simulated interrupted checkpoint copy")
        return write(target, payload)

    monkeypatch.setattr(state, "write_json", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        state.migrate_legacy(args, assignment, train, vocab, new, {k: 2 for k in (1, 5, 25)})
    assert (args.model_out / "metadata.json").read_bytes() == before_metadata
    assert path.read_bytes() == before_trial
    monkeypatch.setattr(state, "write_json", write)
    assert state.migrate_legacy(args, assignment, train, vocab, new, {k: 2 for k in (1, 5, 25)})


def test_evaluation_cli_preserves_locked_csv(tmp_path, monkeypatch):
    from scipy import sparse
    target = tmp_path / "test.csv"
    target.write_text("Excel open file")
    train = evaluation.TrainData(sparse.csr_matrix([[1, 1]]), {b"a": 0, b"b": 1}, [0],
                                 np.array([0, 1]), {b"Artist A": 0, b"Artist B": 1})
    monkeypatch.setattr(evaluation, "read_assignment", lambda _: {})
    from recommender.evaluation import data_cache
    monkeypatch.setattr(data_cache, "load_train_cached", lambda *a: train)
    monkeypatch.setattr(data_cache, "load_eval_cached", lambda *a: ({"test": {1: [0, 1]}}, {}))
    monkeypatch.setattr(evaluation, "read_mapreduce_counts", lambda _: {"a": 1, "b": 1})
    expected = {key: value for key, value in rows_for()[0].items() if key not in {"split", "train_seconds"}}
    monkeypatch.setattr(evaluation, "evaluate", lambda *a, **kw: [expected])
    monkeypatch.setattr(sys, "argv", ["evaluate_models", "--flat", "flat", "--split-file", "split",
                                      "--popularity", "pop", "--out", str(target)])
    replace = Path.replace

    def locked(self, other):
        if Path(other) == target:
            raise PermissionError("Excel lock")
        return replace(self, other)

    monkeypatch.setattr(Path, "replace", locked)
    evaluation.main()
    assert target.read_text() == "Excel open file"
    with (tmp_path / "test.latest.csv").open() as stream:
        row, = csv.DictReader(stream)
    assert row["split"] == "test" and float(row["ndcg@10"]) == pytest.approx(0.2)

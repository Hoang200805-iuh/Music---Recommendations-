import json
from pathlib import Path
from types import SimpleNamespace

from recommender import local_als_runtime as runtime
from recommender.als.tune_als import checkpoint_path
from recommender.evaluation.data_cache import write_json


def test_isolation_preserves_three_legacy_trials_and_cleans_each_scratch_directory(tmp_path, monkeypatch):
    args = SimpleNamespace(flat=tmp_path / "flat", split_file=tmp_path / "split", popularity=tmp_path / "pop",
                           out=tmp_path / "reports/val.csv", model_out=tmp_path / "model", cache_dir=tmp_path / "cache",
                           checkpoint_dir=tmp_path / "checkpoints", context=tmp_path / "context.json",
                           max_playlists=1000, master="local[4]", driver_memory="4g", num_blocks=10,
                           temp_dir=tmp_path / "temporary")
    signature = "unchanged-original-signature"

    def metrics(params):
        score = params["rank"] / 1000 + (0.2 if params["regParam"] == 0.01 else 0)
        score += 0.3 if params["alpha"] == 40 else 0
        return [{"k": k, "model": "als", "split": "val", "n_playlists": 2,
                 "seconds": 1, "train_seconds": 1,
                 **{name: score for name in runtime.METRICS},
                 **{f"{name}_ci95": 0.01 for name in runtime.METRICS}}
                for k in (1, 5, 25)]

    legacy = {}
    for rank in (50, 20, 100):
        params = {"rank": rank, "regParam": 0.1, "alpha": 10.0}
        path = checkpoint_path(args.checkpoint_dir, signature, params)
        write_json(path, {"signature": signature, "params": params, "rows": metrics(params)})
        legacy[path] = path.read_bytes()
    launched, scratches = [], []
    partial = Path(str(args.out) + ".partial.csv")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text("CSV đang mở trong Excel", encoding="utf-8")
    replace = Path.replace

    def locked_partial(self, target):
        if Path(target) == partial:
            raise PermissionError("Simulated Excel lock")
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", locked_partial)

    def fake_child(command, **kwargs):
        if "--prepare" in command:
            write_json(args.context, {"signature": signature, "train_sha256": "train", "can_resume": True})
            return
        assert all(not old.exists() for old in scratches)
        scratch = Path(kwargs["env"]["SPARK_LOCAL_DIRS"]).parent
        assert scratch.is_relative_to(args.temp_dir)
        (scratch / "spark/large_shuffle.tmp").write_bytes(b"x" * 1024)
        scratches.append(scratch)
        params = json.loads(command[command.index("--worker-params") + 1])
        launched.append(params)
        write_json(checkpoint_path(args.checkpoint_dir, signature, params),
                   {"signature": signature, "params": params, "rows": metrics(params)})

    monkeypatch.setattr(runtime.subprocess, "run", fake_child)
    runtime.coordinate(args)
    assert len(launched) == 4
    assert all(params["rank"] == 100 for params in launched)
    assert all(not scratch.exists() for scratch in scratches)
    assert all(path.read_bytes() == original for path, original in legacy.items())
    assert args.out.is_file()
    assert partial.read_text(encoding="utf-8") == "CSV đang mở trong Excel"
    assert partial.with_name(partial.stem + ".latest.csv").is_file()


def test_child_uses_same_original_hyperparameters(tmp_path):
    args = SimpleNamespace(flat=tmp_path / "flat", split_file=tmp_path / "split", popularity=tmp_path / "pop",
                           out=tmp_path / "val.csv", model_out=tmp_path / "model", cache_dir=tmp_path / "cache",
                           checkpoint_dir=tmp_path / "checkpoints", context=tmp_path / "context.json",
                           max_playlists=1000, master="local[4]", driver_memory="4g", num_blocks=10)
    command = runtime.child_command(args)
    assert command[command.index("--num-blocks") + 1] == "10"
    assert command[command.index("--max-playlists") + 1] == "1000"
    assert "recommender.local_als_runtime" in command
    assert runtime.MAX_ITER == 10

"""Stable algorithm fingerprints and checked migration of legacy ALS checkpoints."""
import ast
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

from recommender.als.train_als import DEFAULT_PARAMS, MAX_ITER
from recommender.evaluation.data_cache import source_key, write_json

# Explicit dependencies, identical in the project and the portable source snapshot.
# CLI/report/process management is excluded; training, inference, sampling and
# metric implementations remain part of the fingerprint.
SOURCE_MANIFEST = (
    "als/train_als.py", "als/recommend_als.py", "als/load_als.py", "als/tune_als.py",
    "evaluation/evaluate_models.py", "evaluation/data_cache.py", "evaluation/split_playlists.py",
    "evaluation/recall_at_k.py", "evaluation/precision_at_k.py", "evaluation/ndcg_at_k.py",
    "evaluation/hit_rate_at_k.py", "evaluation/r_precision.py", "popularity/popularity_recommender.py",
)
TUNING_NODES = {"PARAMETER_STAGES", "SEED_SIZES", "selection_score", "tune_one_at_a_time"}
SEED_SIZES = (1, 5, 25)


def algorithm_digest(package=None):
    package = Path(package) if package is not None else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for relative in SOURCE_MANIFEST:
        tree = ast.parse((package / relative).read_text(encoding="utf-8-sig"))
        nodes = []
        for node in tree.body:
            if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'":
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {"main", "add_spark_arguments"}:
                continue
            if isinstance(node, ast.ImportFrom) and node.module == "recommender.report_files":
                continue
            if relative == "als/tune_als.py":
                names = {node.name} if isinstance(node, ast.FunctionDef) else set()
                if isinstance(node, ast.Assign):
                    names = {target.id for target in node.targets if isinstance(target, ast.Name)}
                if not names & TUNING_NODES:
                    continue
            nodes.append(node)
        digest.update(relative.encode())
        digest.update(ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False).encode())
    return digest.hexdigest()


def signature_spec(args, assignment, train_sha256, code):
    popularity = Path(args.popularity).stat()
    return {"train": train_sha256, "data": source_key(args.flat, assignment),
            "popularity_size": popularity.st_size, "popularity_mtime": popularity.st_mtime_ns,
            "n": args.max_playlists, "k": SEED_SIZES, "sample_seed": 42,
            "maxIter": MAX_ITER, "als_seed": 42, "blocks": args.num_blocks, "code": code}


def signature_hash(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def run_signature(args, assignment, train_sha256):
    spec = signature_spec(args, assignment, train_sha256, algorithm_digest())
    return signature_hash({"signature_version": 2, **spec})


def checkpoint_path(directory, signature, params):
    name = "_".join(f"{key}-{params[key]:g}" for key in DEFAULT_PARAMS)
    return Path(directory) / signature / f"{name}.json"


def valid_rows(rows, max_playlists):
    from recommender.evaluation.evaluate_models import METRICS
    if (not isinstance(rows, list) or len(rows) != len(SEED_SIZES)
            or not all(isinstance(row, dict) for row in rows)):
        return False
    if {row.get("k") for row in rows} != set(SEED_SIZES):
        return False
    for row in rows:
        if row.get("model") != "als" or row.get("split") != "val":
            return False
        if not isinstance(row.get("n_playlists"), int) or not 0 < row["n_playlists"] <= max_playlists:
            return False
        for metric in METRICS:
            if not math.isfinite(row.get(metric, math.nan)) or not 0 <= row[metric] <= 1:
                return False
            ci = row.get(f"{metric}_ci95", math.nan)
            if row["n_playlists"] > 1 and (not math.isfinite(ci) or ci < 0):
                return False
        if any(not math.isfinite(row.get(key, math.nan)) or row[key] < 0
               for key in ("seconds", "train_seconds")):
            return False
    return True


def migrate_legacy(args, assignment, train_sha256, vocab, signature, case_counts):
    """Copy verified trials, back up old metadata, then atomically switch metadata.

    The compatibility file pins the *old* directory hashes and their algorithm
    digest. Changed algorithms, data, sampling or blocks cannot claim old scores.
    No checkpoint is deleted and item_factors.npy is never rewritten.
    """
    from recommender.als.load_als import validate_bundle
    from recommender.als.tune_als import PARAMETER_STAGES, selection_score

    if not (Path(args.model_out) / "metadata.json").is_file():
        return False
    try:
        metadata, factors = validate_bundle(args.model_out, vocab, train_sha256)
        del factors
        selection = metadata.get("selection") or {}
        old_signature = selection.get("run_signature")
        if old_signature == signature:
            return False
        compatibility = json.loads(Path(__file__).with_name("als_checkpoint_compat.json").read_text(encoding="utf-8"))
        compatible = [entry for entry in compatibility["versions"]
                      if entry["algorithm_sha256"] == algorithm_digest()]
        expected = {signature_hash(signature_spec(args, assignment, train_sha256, entry["legacy_code_sha256"]))
                    for entry in compatible}
        if old_signature not in expected:
            return False
        if (selection.get("split") != "val" or selection.get("metric") != "mean_ndcg@10"
                or selection.get("seed_sizes") != list(SEED_SIZES) or selection.get("seed") != 42
                or selection.get("max_playlists") != args.max_playlists
                or not metadata.get("item_factors_sha256")
                or metadata.get("inference") != "seed_fold_in_als_wr"):
            return False
        allowed = dict(PARAMETER_STAGES)
        records = []
        for path in sorted((Path(args.checkpoint_dir) / old_signature).glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            params, rows = payload["params"], payload["rows"]
            if (set(params) != set(DEFAULT_PARAMS) or any(params[key] not in allowed[key] for key in params)
                    or path != checkpoint_path(args.checkpoint_dir, old_signature, params)
                    or payload["signature"] != old_signature or not valid_rows(rows, args.max_playlists)
                    or any(row["n_playlists"] != case_counts[row["k"]] for row in rows)):
                return False
            records.append((path, payload))
        if not records:
            return False
        best_score = max(selection_score(payload["rows"]) for _, payload in records)
        best_params = {key: metadata[key] for key in DEFAULT_PARAMS}
        if (not math.isclose(best_score, selection["score"], rel_tol=0, abs_tol=1e-12)
                or not any(payload["params"] == best_params and
                           math.isclose(selection_score(payload["rows"]), best_score, rel_tol=0, abs_tol=1e-12)
                           for _, payload in records)):
            return False
    except (ValueError, OSError, KeyError, TypeError, EOFError):
        print("[migration] Checkpoint cũ chưa đủ điều kiện kiểm chứng; giữ nguyên bản cũ.", flush=True)
        return False

    # All validation precedes writes. A failed/interrupted copy leaves the old
    # metadata active; retrying migration is safe because originals are intact.
    backups = Path(args.checkpoint_dir) / "legacy_backups"
    backups.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix=f"{old_signature[:12]}-", dir=backups))
    shutil.copy2(Path(args.model_out) / "metadata.json", backup / "metadata.json")
    for path, _ in records:
        shutil.copy2(path, backup / path.name)
    write_json(backup / "migration.json", {"from": old_signature, "to": signature,
               "train_sha256": train_sha256, "algorithm_sha256": algorithm_digest(),
               "item_factors_sha256": metadata["item_factors_sha256"]})
    for _, payload in records:
        write_json(checkpoint_path(args.checkpoint_dir, signature, payload["params"]),
                   {**payload, "signature": signature, "migrated_from": old_signature})
    metadata["selection"] = {**selection, "run_signature": signature, "migrated_from": old_signature}
    write_json(Path(args.model_out) / "metadata.json", metadata)
    print(f"[migration] Đã kiểm chứng và chuyển {len(records)} trial; bản sao lưu: {backup}", flush=True)
    return True

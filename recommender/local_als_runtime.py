"""Tuning ALS với một Python/JVM riêng cho mỗi trial, dùng checkpoint cũ.

Tái sử dụng nguyên fit_als, fold-in, metrics, lưới tham số và run_signature.
Fingerprint dùng manifest thuật toán cố định; thay đổi quản lý tiến trình/báo cáo
không vô hiệu hóa trial đã hoàn tất. Hai CLI tuning dùng chung module này.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from recommender.als.load_als import validate_bundle
from recommender.als.recommend_als import ALSRecommender
from recommender.als.train_als import (
    MAX_ITER, create_spark, extract_item_factors, fit_als, release_factors, save_bundle,
    training_digest, training_ratings,
)
from recommender.als.tune_als import (
    SEED_SIZES, checkpoint_path, run_signature, selection_score, tune_one_at_a_time, write_partial_report,
)
from recommender.als_tuning_state import migrate_legacy, valid_rows
from recommender.evaluation.data_cache import load_eval_cached, load_train_cached, write_json
from recommender.evaluation.evaluate_models import METRICS, evaluate, sample_cases
from recommender.evaluation.split_playlists import read_assignment
from recommender.popularity.popularity_recommender import PopularityRecommender, read_mapreduce_counts
from recommender.report_files import write_report


def load_data(args):
    assignment = read_assignment(args.split_file)
    train = load_train_cached(args.flat, assignment, args.cache_dir)
    train_sha = training_digest(train)
    playlists, unseen = load_eval_cached(args.flat, assignment, ["val"], train.vocab,
                                         train.artist_ids, args.cache_dir)
    playlists = playlists["val"]
    for k in SEED_SIZES:
        if not sample_cases(playlists, k, args.max_playlists):
            raise ValueError(f"Validation không có đáp án khi k={k}")
    counts = read_mapreduce_counts(args.popularity)
    if counts.keys() != {uri.decode("utf-8") for uri in train.vocab}:
        raise ValueError("Popularity không khớp vocabulary train")
    popularity = PopularityRecommender({train.vocab[uri.encode("utf-8")]: count for uri, count in counts.items()})

    def artist_of(item):
        return train.item_artists[item] if item >= 0 else unseen[item]

    return train, playlists, popularity, artist_of, train_sha, run_signature(args, assignment, train_sha)


def valid_best(args, signature, train_sha, vocab):
    try:
        metadata, factors = validate_bundle(args.model_out, vocab, train_sha)
        del factors
        selection = metadata.get("selection") or {}
        if selection.get("run_signature") == signature and np.isfinite(selection.get("score", np.nan)):
            return float(selection["score"])
    except (ValueError, OSError, KeyError, EOFError):
        pass
    return None


def prepare(args):
    train, playlists, _, _, train_sha, signature = load_data(args)
    best = valid_best(args, signature, train_sha, train.vocab)
    if best is None:
        assignment = read_assignment(args.split_file)
        counts = {k: len(sample_cases(playlists, k, args.max_playlists)) for k in SEED_SIZES}
        if migrate_legacy(args, assignment, train_sha, train.vocab, signature, counts):
            best = valid_best(args, signature, train_sha, train.vocab)
    context = {"signature": signature, "train_sha256": train_sha, "can_resume": best is not None}
    write_json(args.context, context)
    print(f"[isolated] Dữ liệu/checkpoint đã kiểm tra; resume={context['can_resume']}", flush=True)


def train_one(args, params):
    context = json.loads(args.context.read_text(encoding="utf-8"))
    train, playlists, popularity, artist_of, train_sha, signature = load_data(args)
    if signature != context["signature"] or train_sha != context["train_sha256"]:
        raise ValueError("Dữ liệu/source thay đổi trong lúc tuning; dừng để giữ kết quả nhất quán")
    previous_best = valid_best(args, signature, train_sha, train.vocab)
    spark = create_spark(args.master, args.driver_memory, "PlaylistALSOneTrial")
    application_id = spark.sparkContext.applicationId
    start = time.time()
    try:
        with training_ratings(spark, train, args.temp_dir) as ratings:
            print(f"[isolated] PID={os.getpid()}, app={application_id}, params={params}, maxIter={MAX_ITER}", flush=True)
            model = fit_als(ratings, **params, num_blocks=args.num_blocks)
            try:
                recommender = ALSRecommender(extract_item_factors(model, len(train.vocab)), popularity,
                                             params["regParam"], params["alpha"])
                seconds = round(time.time() - start, 1)
                rows = evaluate({"als": recommender}, playlists, SEED_SIZES, args.max_playlists, artist_of=artist_of)
                for row in rows:
                    row.update(split="val", train_seconds=seconds)
                score = selection_score(rows)
                if not np.isfinite(score):
                    raise ValueError("Điểm validation không hữu hạn")
                if previous_best is None or score > previous_best:
                    save_bundle(args.model_out, recommender, train.vocab, params,
                                selection={"split": "val", "metric": "mean_ndcg@10", "score": score,
                                           "seed_sizes": list(SEED_SIZES), "max_playlists": args.max_playlists,
                                           "seed": 42, "run_signature": signature}, train_sha256=train_sha)
                write_json(checkpoint_path(args.checkpoint_dir, signature, params),
                           {"signature": signature, "params": params, "rows": rows,
                            "worker_pid": os.getpid(), "spark_application_id": application_id})
            finally:
                release_factors(model)
    finally:
        spark.stop()


def child_command(args):
    return [sys.executable, "-u", "-m", "recommender.local_als_runtime",
            "--flat", str(args.flat), "--split-file", str(args.split_file),
            "--popularity", str(args.popularity), "--out", str(args.out),
            "--model-out", str(args.model_out), "--cache-dir", str(args.cache_dir),
            "--checkpoint-dir", str(args.checkpoint_dir), "--context", str(args.context),
            "--max-playlists", str(args.max_playlists), "--master", args.master,
            "--driver-memory", args.driver_memory, "--num-blocks", str(args.num_blocks)]


def write_final_report(path, rows):
    fields = ["stage", "rank", "regParam", "alpha", "maxIter", "selected", "reused", "mean_ndcg@10",
              "split", "k", "model", "n_playlists"]
    for metric in METRICS:
        fields += [metric, f"{metric}_ci95"]
    fields += ["train_seconds", "seconds"]
    def write(staged):
        with staged.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return write_report(path, write)


def coordinate(args):
    command = child_command(args)
    subprocess.run([*command, "--prepare"], check=True)
    context = json.loads(args.context.read_text(encoding="utf-8"))
    root = args.temp_dir.resolve() / "isolated_trials"
    root.mkdir(parents=True, exist_ok=True)
    trials = []

    def run_trial(params):
        record = checkpoint_path(args.checkpoint_dir, context["signature"], params)
        reused = context["can_resume"] and record.is_file()
        if reused:
            try:
                payload = json.loads(record.read_text(encoding="utf-8"))
                reused = (payload["signature"] == context["signature"] and payload["params"] == params
                          and valid_rows(payload["rows"], args.max_playlists))
            except (OSError, ValueError, KeyError, TypeError):
                reused = False
        if reused:
            print(f"[resume] Bỏ qua trial đã hoàn tất: {params}", flush=True)
        else:
            free = shutil.disk_usage(root).free / 2**30
            print(f"[disk] Trước trial {params}: {free:.1f} GiB trống", flush=True)
            if free < 5:
                raise RuntimeError("Dưới 5 GiB trống; giải phóng đĩa trước khi thử trial tiếp")
            # Thư mục riêng do coordinator tạo; chỉ dọn nó sau khi child đã thoát.
            with tempfile.TemporaryDirectory(prefix="als-trial-", dir=root) as directory:
                scratch = Path(directory).resolve()
                if not scratch.is_relative_to(root):
                    raise RuntimeError("Thư mục trial nằm ngoài thư mục tạm dự kiến")
                (scratch / "ratings").mkdir()
                (scratch / "spark").mkdir()
                env = dict(os.environ, SPARK_LOCAL_DIRS=str(scratch / "spark"))
                subprocess.run([*command, "--worker-params", json.dumps(params),
                                "--temp-dir", str(scratch / "ratings")], env=env, check=True)
            print(f"[disk] Sau trial: {shutil.disk_usage(root).free / 2**30:.1f} GiB trống; JVM đã thoát", flush=True)
            payload = json.loads(record.read_text(encoding="utf-8"))
        trials.append((params, payload["rows"]))
        write_partial_report(str(args.out) + ".partial.csv", trials)
        print(f"[checkpoint] Xong trial {len(trials)}/7, reused={reused}", flush=True)
        return payload["rows"]

    best, rows = tune_one_at_a_time(run_trial)
    report = write_final_report(args.out, rows)
    print(f"Tham số ALS đã chọn trên val: {best}; maxIter={MAX_ITER}", flush=True)
    print(f"Báo cáo: {report}\nBundle: {args.model_out}", flush=True)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    os.environ.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("flat", "split-file", "popularity", "out", "model-out"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--checkpoint-dir", type=Path)
    ap.add_argument("--temp-dir", type=Path)
    ap.add_argument("--context", type=Path)
    ap.add_argument("--max-playlists", type=int, default=1000)
    ap.add_argument("--master", default="local[*]")
    ap.add_argument("--driver-memory", default="4g")
    ap.add_argument("--num-blocks", type=int, default=10)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--worker-params")
    args = ap.parse_args(argv)
    if args.max_playlists <= 0:
        ap.error("--max-playlists phải > 0")
    if args.num_blocks <= 0:
        ap.error("--num-blocks phải > 0")
    # Omitted optional paths stay beside the requested model, on any machine.
    work = args.model_out.resolve().parent / (args.model_out.name + ".work")
    args.cache_dir = args.cache_dir or work / "cache"
    args.checkpoint_dir = args.checkpoint_dir or work / "checkpoints"
    args.temp_dir = args.temp_dir or work / "temporary"
    for directory in (args.cache_dir, args.checkpoint_dir, args.temp_dir):
        directory.mkdir(parents=True, exist_ok=True)
    args.context = args.context or args.out.with_suffix(".context.json")
    if args.prepare:
        prepare(args)
    elif args.worker_params is not None:
        train_one(args, json.loads(args.worker_params))
    else:
        coordinate(args)


if __name__ == "__main__":
    main()

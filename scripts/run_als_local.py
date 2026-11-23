"""Chạy ALS trên máy local: kiểm tra Spark, stage input, cache, tuning và so sánh.

python scripts/run_als_local.py --check-only
python scripts/run_als_local.py
python scripts/run_als_local.py --smoke-test --work-dir <thư mục thử riêng>
"""
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
SOURCES = (
    "local_als_runtime.py", "als_tuning_state.py", "als_checkpoint_compat.json",
    "report_files.py",
    "als/train_als.py", "als/tune_als.py", "als/load_als.py", "als/recommend_als.py",
    "evaluation/evaluate_models.py", "evaluation/data_cache.py", "evaluation/split_playlists.py",
    "evaluation/recall_at_k.py", "evaluation/precision_at_k.py", "evaluation/ndcg_at_k.py",
    "evaluation/hit_rate_at_k.py", "evaluation/r_precision.py",
    "cooccurrence/cooccurrence_recommender.py", "hybrid/hybrid_recommender.py",
    "hybrid/playlist_titles.py", "popularity/popularity_recommender.py",
)
INPUTS = {
    "flat": "processed/playlist_tracks_flat.tsv", "split": "processed/playlist_split.tsv",
    "titles": "processed/playlist_titles.tsv", "popularity": "analytics/track_popularity_train/part-00000",
}


def default_work_dir():
    return REPO / "outputs/als_local"


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def stage_file(source, target):
    """Copy có progress/resume .part; không sửa input nguồn."""
    source, target = Path(source).resolve(), Path(target).resolve()
    stat = source.stat()
    target.parent.mkdir(parents=True, exist_ok=True)
    if source == target:
        return target
    if target.is_file() and target.stat().st_size == stat.st_size and target.stat().st_mtime_ns == stat.st_mtime_ns:
        print(f"[input cache] {target.name}: đã có", flush=True)
        return target
    part = target.with_name(target.name + ".part")
    manifest = target.with_name(target.name + ".copy.json")
    spec = {"source": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    resume = part.is_file() and manifest.is_file() and json.loads(manifest.read_text()) == spec
    offset = part.stat().st_size if resume and part.stat().st_size <= stat.st_size else 0
    atomic_json(manifest, spec)
    print(f"[copy] {source} -> {target} (từ {offset / 1e6:.0f} MB)", flush=True)
    start = last = time.monotonic()
    copied = offset
    with source.open("rb") as src, part.open("ab" if offset else "wb") as dst:
        src.seek(offset)
        for block in iter(lambda: src.read(4 * 1024 * 1024), b""):
            dst.write(block)
            copied += len(block)
            now = time.monotonic()
            if now - last >= 5:
                print(f"[copy] {target.name}: {copied / max(stat.st_size, 1):.1%} | "
                      f"{(copied - offset) / max(now - start, 0.001) / 1e6:.1f} MB/s", flush=True)
                last = now
    if copied != stat.st_size or source.stat().st_mtime_ns != stat.st_mtime_ns:
        raise RuntimeError(f"Input thay đổi hoặc copy chưa đủ: {source}")
    shutil.copystat(source, part)
    part.replace(target)
    print(f"[copy] Xong {target.name}: {copied / 1e6:.1f} MB", flush=True)
    return target


def stage_source(work):
    destination = work / "source/recommender"
    for relative in SOURCES:
        source = REPO / "recommender" / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    return destination.parent


def prepare_inputs(input_dir, work, copy_inputs=False):
    """Dùng trực tiếp input; chỉ nhân bản khi người chạy yêu cầu --copy-inputs."""
    files = {kind: (Path(input_dir) / relative).resolve() for kind, relative in INPUTS.items()}
    if copy_inputs:
        return {kind: stage_file(path, Path(work) / "data" / INPUTS[kind]) for kind, path in files.items()}
    print(f"[input] Dùng trực tiếp dữ liệu: {Path(input_dir).resolve()}", flush=True)
    return files


def configure_environment():
    for package in ("numpy", "scipy", "pyspark"):
        try:
            print(f"[env] {package}={importlib.metadata.version(package)}", flush=True)
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError("Thiếu thư viện; chạy python -m pip install -r requirements-als.txt")
    java = shutil.which("java")
    if not java:
        raise RuntimeError("Không tìm thấy Java trong PATH")
    result = subprocess.run([java, "-XshowSettings:properties", "-version"], capture_output=True, text=True, check=True)
    text = result.stdout + result.stderr
    home = re.search(r"^\s*java.home\s*=\s*(.+)$", text, re.MULTILINE)
    if home is None:
        raise RuntimeError("Không đọc được java.home")
    os.environ.update(JAVA_HOME=home[1].strip(), PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                      PYTHONUNBUFFERED="1", SPARK_LOCAL_IP="127.0.0.1",
                      PYSPARK_PYTHON=sys.executable, PYSPARK_DRIVER_PYTHON=sys.executable)
    if os.name == "nt":
        # Gateway nhỏ ở temp riêng của user; ratings/shuffle lớn có đường dẫn riêng.
        user_temp = Path(os.environ["LOCALAPPDATA"]) / "Temp"
        os.environ["TEMP"] = os.environ["TMP"] = str(user_temp)
    print(f"[env] JAVA_HOME={os.environ['JAVA_HOME']}", flush=True)


def report_memory():
    if os.name == "nt":
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        *[(name, ctypes.c_ulonglong) for name in ("total", "available", "page_total", "page_available",
                                                                  "virtual_total", "virtual_available", "extended")]]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            print(f"[RAM] Trống {status.available / 2**30:.1f}/{status.total / 2**30:.1f} GiB", flush=True)
            # ullAvailPageFile is remaining system commit capacity, not free disk.
            print(f"[commit] Còn cấp phát {status.page_available / 2**30:.1f}/"
                  f"{status.page_total / 2**30:.1f} GiB (RAM + pagefile)", flush=True)
            if status.available < 6 * 2**30:
                print("[RAM] RAM trống thấp; đóng bớt ứng dụng trước khi train toàn bộ để hạn chế paging.", flush=True)
            return {"physical_available": status.available, "commit_available": status.page_available}


def ensure_memory_budget(memory, driver_memory, smoke_test=False):
    """Conservative early check, not a guarantee of peak ALS memory usage."""
    if memory is None:
        return
    match = re.fullmatch(r"([1-9][0-9]*)([kmgt])", driver_memory.lower())
    if match is None:
        raise ValueError("--driver-memory cần đơn vị k/m/g/t, ví dụ 2g hoặc 4096m")
    heap = int(match[1]) * 1024 ** ("kmgt".index(match[2]) + 1)
    # CSR, validation, Python dictionaries and JVM native allocations use memory
    # outside the configured Java heap. Keep an explicit minimum headroom.
    required = heap + (1 if smoke_test else 2) * 2**30
    if memory["commit_available"] < required:
        raise RuntimeError(
            f"Thiếu bộ nhớ cấp phát Windows: còn {memory['commit_available'] / 2**30:.1f} GiB; "
            f"cần dự phòng ít nhất {required / 2**30:.1f} GiB cho heap {driver_memory} + Python/JVM. "
            "Đóng ứng dụng chiếm RAM hoặc tăng pagefile rồi chạy lại. "
            "Dừng trước khi train để tránh JVM lỗi paging file (errno=1455)."
        )


def run(command, source, log_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(source) + os.pathsep + env.get("PYTHONPATH", "")
    command = list(map(str, command))
    print("\n[command] " + " ".join(command), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n[command] " + " ".join(command) + "\n")
        log.flush()
        with subprocess.Popen(command, cwd=source, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
    if code:
        raise RuntimeError(f"Lệnh thất bại (exit={code}); log: {log_path}")


def fixture_inputs(directory):
    directory.mkdir(parents=True, exist_ok=True)
    counts = {}
    with (directory / "flat.tsv").open("w", encoding="utf-8") as f:
        for pid in range(16):
            tracks = [(pid * 7 + i) % 70 for i in range(32)]
            if pid >= 12:
                tracks[-1] = 100 + pid
            for item in tracks:
                f.write(f"{pid}\turi:{item}\tTrack {item}\tArtist {item % 10}\n")
                if pid < 12:
                    counts[item] = counts.get(item, 0) + 1
    (directory / "split.tsv").write_text("".join(
        f"{pid}\t{'train' if pid < 12 else 'val' if pid < 14 else 'test'}\n" for pid in range(16)), encoding="utf-8")
    (directory / "titles.tsv").write_text("".join(f"{pid}\trock\n" for pid in range(16)), encoding="utf-8")
    (directory / "popularity.tsv").write_text("".join(
        f"uri:{item}\t{count}\tTrack {item}\tArtist {item % 10}\n" for item, count in counts.items()), encoding="utf-8")
    return {kind: directory / f"{kind if kind != 'popularity' else 'popularity'}.tsv" for kind in INPUTS}


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, default=REPO / "hdfs_sim/music")
    ap.add_argument("--work-dir", type=Path, default=default_work_dir())
    ap.add_argument("--master", default="local[4]")
    ap.add_argument("--driver-memory", default="4g")
    ap.add_argument("--num-blocks", type=int, default=10)
    ap.add_argument("--val-playlists", type=int, default=1000)
    ap.add_argument("--test-playlists", type=int, default=10000)
    ap.add_argument("--check-only", action="store_true", help="chỉ kiểm tra file/temp/Java/Spark, không nạp dữ liệu lớn")
    ap.add_argument("--smoke-test", action="store_true", help="chỉ chạy mẫu nhỏ đủ 7 trials và 5 models")
    ap.add_argument("--tune-only", action="store_true")
    ap.add_argument("--copy-inputs", action="store_true", help="tùy chọn copy input vào work-dir/data; mặc định đọc trực tiếp")
    args = ap.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    for name in ("temporary/ratings", "temporary/spark", "cache", "checkpoints", "reports", "models"):
        (work / name).mkdir(parents=True, exist_ok=True)
    print(f"[local] Thư mục làm việc: {work}", flush=True)
    print(f"[local] Trống: {shutil.disk_usage(work).free / 2**30:.1f} GiB", flush=True)
    memory = report_memory()
    ensure_memory_budget(memory, args.driver_memory, args.smoke_test)
    with tempfile.TemporaryDirectory(prefix="als-preflight-", dir=work / "temporary") as directory:
        probe = Path(directory) / "probe.txt"
        probe.write_text("ok", encoding="utf-8")
        assert probe.read_text(encoding="utf-8") == "ok"
    configure_environment()
    os.environ["SPARK_LOCAL_DIRS"] = str(work / "temporary/spark")
    source = stage_source(work)
    log = work / "pipeline.log"
    if not args.smoke_test:
        for name, relative in INPUTS.items():
            path = args.input_dir / relative
            if not path.is_file():
                raise FileNotFoundError(f"Thiếu input {name}: {path}")
            print(f"[input] {path}: {path.stat().st_size / 1e6:.1f} MB", flush=True)
    check = ("from recommender.als.train_als import create_spark; "
             f"s=create_spark({args.master!r},{args.driver_memory!r},'ALSLocalPreflight'); "
             "assert s.range(3).count()==3; print('SPARK_PREFLIGHT_OK',flush=True); s.stop()")
    run([sys.executable, "-u", "-c", check], source, log)
    if args.check_only:
        print("LOCAL_CHECK_OK: input/temp/Java/Spark hợp lệ; chưa chạy dữ liệu lớn.", flush=True)
        return
    if args.smoke_test:
        files = fixture_inputs(work / "data")
        args.val_playlists = args.test_playlists = 2
    else:
        files = prepare_inputs(args.input_dir, work, args.copy_inputs)
    common = ["--flat", files["flat"], "--split-file", files["split"], "--popularity", files["popularity"]]
    model = work / "models/als_implicit"
    tuning = work / "reports/als_tuning_val.csv"
    tune = [sys.executable, "-u", "-m", "recommender.local_als_runtime", *common,
            "--out", tuning, "--model-out", model, "--cache-dir", work / "cache",
            "--checkpoint-dir", work / "checkpoints", "--temp-dir", work / "temporary/ratings",
            "--max-playlists", args.val_playlists, "--master", args.master,
            "--driver-memory", args.driver_memory, "--num-blocks", args.num_blocks]
    run(tune, source, log)
    if not args.tune_only:
        sys.path.insert(0, str(source))
        from recommender.report_files import new_report_temp, publish_report
        result_target = work / "reports/results_with_als.csv"
        result = new_report_temp(result_target)
        run([sys.executable, "-u", "-m", "recommender.evaluation.evaluate_models", *common,
             "--out", result, "--als-model", model, "--cache-dir", work / "cache",
             "--titles", files["titles"], "--tuned", "0.5", "0.25", "10", "--hybrid", "1", "0.1",
             "--max-playlists", args.test_playlists], source, log)
        result = publish_report(result, result_target)
        print(f"[report] Kết quả so sánh: {result}", flush=True)
    if args.smoke_test:
        with tuning.open(encoding="utf-8") as f:
            assert len(list(csv.DictReader(f))) == 27
        if not args.tune_only:
            with result.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 20 and len({row["model"] for row in rows}) == 5
        offset = log.stat().st_size
        run(tune, source, log)
        resumed = log.read_bytes()[offset:].decode("utf-8")
        assert resumed.count("[resume] Bỏ qua trial đã hoàn tất") == 7
        assert "[ratings]" not in resumed and "[load train]" not in resumed
        print("LOCAL_SMOKE_OK: 7 ALS trials, 27 val rows, 5 models, cache/resume OK.", flush=True)
    print(f"LOCAL_DONE: {work / 'reports'}", flush=True)


if __name__ == "__main__":
    main()

"""Chỉnh từng tham số implicit ALS trên validation, luôn giữ maxIter=10.

Khởi đầu rank=50, regParam=0.1, alpha=10. Thử rank trước, giữ rank tốt nhất
để thử regParam, rồi giữ cả hai để thử alpha. Tiêu chí chọn: trung bình
NDCG@10 ở k seed = 1, 5, 25, giống tune_cooccurrence. Không đọc tập test.

python -m recommender.als.tune_als --help
"""
import csv

import numpy as np

from recommender.als.train_als import DEFAULT_PARAMS, MAX_ITER
from recommender.als_tuning_state import checkpoint_path, run_signature
from recommender.report_files import write_report

PARAMETER_STAGES = (("rank", (20, 50, 100)), ("regParam", (0.01, 0.1, 1.0)), ("alpha", (1.0, 10.0, 40.0)))
SEED_SIZES = (1, 5, 25)


def write_partial_report(path, trials):
    rows = [{**params, **row} for params, values in trials for row in values]
    if not rows:
        return None

    def write(staged):
        with staged.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return write_report(path, write)


def selection_score(rows):
    return float(np.mean([row["ndcg@10"] for row in rows]))


def tune_one_at_a_time(run_trial):
    """run_trial(params) trả các dòng evaluate; cùng bộ tham số chỉ train một lần.

    Hòa điểm thì giữ tham số hiện tại. Ghi đủ ba ứng viên của từng giai đoạn
    vào báo cáo, kể cả ứng viên tái sử dụng từ cache (tổng 7 lần fit khác nhau).
    """
    current = dict(DEFAULT_PARAMS)
    cache = {}
    report = []

    def trial(params):
        key = tuple(params[name] for name in DEFAULT_PARAMS)
        reused = key in cache
        if not reused:
            cache[key] = run_trial(dict(params))
        return cache[key], reused

    current_rows, _ = trial(current)
    for parameter, values in PARAMETER_STAGES:
        best = dict(current)
        best_rows = current_rows
        stage_rows = []
        for value in values:
            candidate = {**current, parameter: value}
            rows, reused = trial(candidate)
            score = selection_score(rows)
            for row in rows:
                stage_rows.append({"stage": parameter, **candidate, "maxIter": MAX_ITER,
                                   "mean_ndcg@10": score, "reused": reused, **row})
            if score > selection_score(best_rows):
                best, best_rows = candidate, rows
        for row in stage_rows:
            row["selected"] = all(row[name] == best[name] for name in DEFAULT_PARAMS)
        report.extend(stage_rows)
        current, current_rows = best, best_rows
        print(f"Chọn sau bước {parameter}: {current}, NDCG@10={selection_score(current_rows):.6f}", flush=True)
    return current, report


def main(argv=None):
    # Local import avoids a cycle: runtime reuses the pure tuning helpers above.
    from recommender.local_als_runtime import main as run
    return run(argv)


if __name__ == "__main__":
    main()

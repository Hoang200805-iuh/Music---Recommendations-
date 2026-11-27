"""Lọc tần suất tương tác và khử playlist trùng hoặc gần trùng."""
from fractions import Fraction

from pyspark.sql import functions as F

from spark.spark_session import has_rows, log, release, snapshot


def filter_core(events, min_playlist_tracks=5, min_track_playlists=5, max_rounds=100):
    """Lọc bipartite k-core trên các cặp (playlist, bài) phân biệt đến khi ổn định."""
    current = events
    history = []
    before = current.count()
    for step in range(1, max_rounds + 1):
        good_tracks = (current.select("pid", "track_id").distinct()
                       .groupBy("track_id").count()
                       .filter(F.col("count") >= min_track_playlists).select("track_id"))
        narrowed = current.join(good_tracks, "track_id", "left_semi")
        good_playlists = (narrowed.groupBy("pid").agg(
            F.countDistinct("track_id").alias("n"))
            .filter(F.col("n") >= min_playlist_tracks).select("pid"))
        updated = snapshot(narrowed.join(good_playlists, "pid", "left_semi"))
        after = updated.count()
        history.append({"round": step, "rows_before": before, "rows_after": after})
        log(f"core {step}: {before:,} -> {after:,} dòng")
        release(current)
        current = updated
        if before == after:
            return current, history
        before = after
    raise RuntimeError("Lọc tần suất chưa hội tụ; tăng --max-core-rounds.")


def playlist_sets(events):
    """Tạo tập bài phân biệt của mỗi playlist để so sánh."""
    return events.groupBy("pid").agg(F.sort_array(F.collect_set("track_id")).alias("items"))


def exact_duplicates(sets):
    """Tìm playlist có tập bài giống hệt nhau và giữ pid nhỏ nhất trong mỗi nhóm."""
    representatives = sets.groupBy("items").agg(F.min("pid").alias("matched_lower_pid"))
    return (sets.join(representatives, "items")
            .filter(F.col("pid") != F.col("matched_lower_pid"))
            .select("pid", "matched_lower_pid"))


def find_near_pairs(events, threshold="0.9"):
    """Tìm chính xác các cặp pid_a < pid_b có Jaccard lớn hơn ngưỡng."""
    ratio = Fraction(str(threshold))
    if not 0 < ratio < 1:
        raise ValueError("Ngưỡng Jaccard phải nằm giữa 0 và 1.")
    p, q = ratio.numerator, ratio.denominator
    distinct_edges = events.select("pid", "track_id").distinct()
    frequencies = distinct_edges.groupBy("track_id").count().withColumnRenamed("count", "df")
    # Mọi playlist dùng cùng thứ tự token theo (tần suất, track_id).
    ordered = snapshot(distinct_edges.join(frequencies, "track_id").groupBy("pid").agg(
        F.sort_array(F.collect_list(F.struct("df", "track_id"))).alias("tokens"))
        .withColumn("items", F.expr("transform(tokens, x -> x.track_id)"))
        .select("pid", "items").withColumn("n", F.size("items")))
    # Nếu Jaccard > p/q, các tiền tố độ dài n - floor(p*n/q) phải có token chung.
    prefix = snapshot(ordered.select(
        "pid", "n", F.explode(F.expr(
            f"slice(items, 1, n - CAST((CAST(n AS BIGINT) * {p}) DIV {q} AS INT))"
        )).alias("token")))
    a, b = prefix.alias("a"), prefix.alias("b")
    candidates = snapshot(a.join(b, (
        (F.col("a.token") == F.col("b.token")) & (F.col("a.pid") < F.col("b.pid"))
        & (q * F.least(F.col("a.n"), F.col("b.n"))
           > p * F.greatest(F.col("a.n"), F.col("b.n")))
    )).select(F.col("a.pid").alias("pid_a"), F.col("b.pid").alias("pid_b")).distinct())
    left = ordered.select(F.col("pid").alias("pid_a"), F.col("items").alias("items_a"))
    right = ordered.select(F.col("pid").alias("pid_b"), F.col("items").alias("items_b"))
    scored = (candidates.join(left, "pid_a").join(right, "pid_b")
              .withColumn("intersection", F.size(F.array_intersect("items_a", "items_b")))
              .withColumn("union", F.size("items_a") + F.size("items_b") - F.col("intersection")))
    # So sánh bằng số nguyên để phân biệt đúng 90% với lớn hơn 90%.
    result = snapshot(scored.filter(q * F.col("intersection") > p * F.col("union"))
                      .select("pid_a", "pid_b", "intersection", "union",
                              (F.col("intersection") / F.col("union")).alias("jaccard")))
    release(candidates)
    release(prefix)
    release(ordered)
    return result


def clean_until_stable(spark, events, args):
    """Lọc lại tần suất và playlist trùng sau mỗi vòng loại bỏ."""
    audit = spark.createDataFrame([], "pid long, matched_lower_pid long, reason string, round int")
    rounds = []
    current = events
    for outer in range(1, args.max_clean_rounds + 1):
        log(f"Vòng làm sạch {outer}")
        current, core_history = filter_core(current, args.min_playlist_tracks,
                                             args.min_track_playlists, args.max_core_rounds)
        if not has_rows(current):
            raise ValueError("Không còn dữ liệu sau lọc. Kiểm tra tập dữ liệu và các ngưỡng.")
        sets = snapshot(playlist_sets(current))
        exact = snapshot(exact_duplicates(sets))
        release(sets)
        exact_count = exact.count()
        if exact_count:
            updated = snapshot(current.join(exact.select("pid"), "pid", "left_anti"))
            release(current)
            current = updated
        pairs = find_near_pairs(current, args.near_threshold)
        near = snapshot(pairs.groupBy(F.col("pid_b").alias("pid")).agg(
            F.min("pid_a").alias("matched_lower_pid")))
        near_count = near.count()
        release(pairs)
        for removed, reason, count in ((exact, "exact", exact_count), (near, "near", near_count)):
            if count:
                updated_audit = snapshot(audit.unionByName(
                    removed.withColumn("reason", F.lit(reason)).withColumn("round", F.lit(outer))))
                release(audit)
                audit = updated_audit
        log(f"Bỏ {exact_count:,} playlist trùng, {near_count:,} playlist gần trùng")
        rounds.append({"round": outer, "core": core_history,
                       "exact_removed": exact_count, "near_removed": near_count})
        if near_count:
            updated = snapshot(current.join(near.select("pid"), "pid", "left_anti"))
            release(current)
            current = updated
        release(exact)
        release(near)
        if exact_count + near_count == 0:
            return current, audit, rounds
    raise RuntimeError("Làm sạch chưa hội tụ; tăng --max-clean-rounds.")

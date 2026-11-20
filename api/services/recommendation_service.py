"""
Tạo gợi ý cho một request: đổi URI thành id bài, gọi mô hình, gắn điểm và tên bài.

Điểm (score) theo từng mô hình:
    popularity                  số đếm MapReduce của bài
    cooccurrence(_tuned)        điểm Co-occurrence
    hybrid                      tổng điểm Co-occurrence và tên playlist (mỗi loại đã đưa về 0-1)
    rerank                      điểm LightGBM, chỉ dùng để so thứ tự (khi không có seed thì không xếp lại,
                                điểm là điểm của mô hình gốc: tên playlist)
    als                         tích vô hướng vector playlist và vector bài
Bài được bù từ danh sách phổ biến vì mô hình không đủ bài có điểm có score = null, fallback = true.
"""
from api.services.explain_service import Explainer
from recommender.hybrid.hybrid_recommender import combine


class UnknownModelError(ValueError):
    pass


def split_name(name):
    artist, _, track = name.partition(" – ")
    return artist, track


def _scored(service, model_name, seed, title, n):
    """(danh sách id bài, {id bài: điểm}) theo mô hình model_name."""
    model = service.models[model_name]
    if model_name == "popularity":
        items = model.recommend(seed, n)
        return items, {i: float(service.counts[i]) for i in items}
    if model_name == "rerank":
        candidates, lgbm_scores = model.score(seed, title)
        items = model.recommend(seed, n, title=title, scored=(candidates, lgbm_scores))
        if lgbm_scores is None:
            # Không có seed: thứ tự gốc, mà không có seed thì điểm gốc chỉ còn phần tên playlist
            builder = model.builder
            base = combine(builder.cooccurrence.score(seed), builder.titles.scores(title), builder.title_weight)
            return items, {i: float(base[i]) for i in items if base[i] > 0}
        return items, dict(zip(candidates.tolist(), lgbm_scores.tolist()))
    if model_name == "hybrid":
        scores = model.score(seed, title)
        items = model.recommend(seed, n, title=title, scores=scores)
    else:
        scores = model.score(seed)
        items = model.recommend(seed, n, scores)
    return items, {i: float(scores[i]) for i in items if scores[i] != 0}


def recommend(service, seed_tracks, title, n, model_name=None):
    model_name = model_name or service.default_model
    if model_name not in service.models:
        raise UnknownModelError(f"Mô hình '{model_name}' không có; các mô hình đã nạp: {sorted(service.models)}")
    seed, unknown = [], []
    for uri in seed_tracks:
        item = service.uris.item(uri)
        (unknown if item is None else seed).append(uri if item is None else item)
    title = title.strip() if title else None

    items, scores = _scored(service, model_name, seed, title, n)
    items = items[:n]
    fallback = [item not in scores for item in items]
    reasons = Explainer(service).explain(items, seed, title, fallback, model_name)
    results = []
    for rank, (item, is_fallback, why) in enumerate(zip(items, fallback, reasons), 1):
        artist, track = split_name(service.names[item])
        results.append({
            "rank": rank,
            "track_uri": service.uris[item],
            "track_name": track,
            "artist_name": artist,
            "score": scores.get(item),
            "fallback": is_fallback,
            "reasons": why,
        })
    return {
        "model": model_name,
        "n_seed_tracks_used": len(seed),
        "unknown_seed_tracks": unknown,
        "title_matched_playlists": service.titles.n_playlists(title) if title else 0,
        "recommendations": results,
    }

"""Map ALS bundle rows to application IDs by URI; inference needs no Spark."""
import json
from pathlib import Path

import numpy as np

from recommender.als.recommend_als import ALSRecommender as FoldInRecommender


class ALSRecommender:
    def __init__(self, factors, item_ids, fallback, reg_param=0.1, alpha=10.0):
        self.core = FoldInRecommender(factors, None, reg_param, alpha)
        self.factors = self.core.item_factors
        self.item_ids = np.asarray(item_ids, dtype=np.int64)
        if len(self.item_ids) != len(self.factors):
            raise ValueError("ALS rows and item IDs must have equal length")
        known = self.item_ids[self.item_ids >= 0]
        if len(np.unique(known)) != len(known):
            raise ValueError("Duplicate ALS track URI")
        self.rows = {int(item): row for row, item in enumerate(self.item_ids) if item >= 0}
        self.fallback = fallback

    def score(self, seed_items):
        rows = [self.rows[item] for item in set(seed_items) if item in self.rows]
        user = self.core.fold_in(rows)
        scores = np.full(max(self.fallback.ranked_items, default=-1) + 1, np.nan, dtype=np.float32)
        if user is not None:
            valid = self.item_ids >= 0
            scores[self.item_ids[valid]] = (self.factors @ user.astype(np.float32))[valid]
        return scores

    def recommend(self, seed_items, n, scores=None):
        if n <= 0:
            return []
        seed = set(seed_items)
        if not any(item in self.rows for item in seed):
            return self.fallback.recommend(seed, n)
        scores = self.score(seed) if scores is None else scores
        candidates = np.array([item for item in self.rows if item not in seed], dtype=np.int64)
        if len(candidates) > n:
            values = scores[candidates]
            cutoff = np.partition(values, len(values) - n)[len(values) - n]
            higher = candidates[values > cutoff]
            tied = np.sort(candidates[values == cutoff])[:n - len(higher)]
            candidates = np.concatenate((higher, tied))
        order = np.lexsort((candidates, -scores[candidates]))
        items = candidates[order[:n]].tolist()
        if len(items) < n:
            items += self.fallback.recommend(seed | set(items), n - len(items))
        return items


def load_als(path, uris, fallback):
    """Read the validated training bundle, or the older URI-list export."""
    path = Path(path)
    if not (path / "item_factors.npy").exists():
        return None, 0
    if (path / "metadata.json").exists():
        from recommender.als.load_als import validate_bundle
        metadata, factors = validate_bundle(path)
        item_ids = np.full(len(factors), -1, dtype=np.int64)
        seen = set()
        with (path / "items.tsv").open(encoding="utf-8") as stream:
            for line in stream:
                row, uri = line.rstrip("\r\n").split("\t", 1)
                row = int(row)
                if not 0 <= row < len(factors) or row in seen:
                    raise ValueError("Invalid or duplicate ALS row ID")
                seen.add(row)
                item = uris.item(uri)
                item_ids[row] = -1 if item is None else item
        if len(seen) != len(factors):
            raise ValueError("ALS items.tsv is incomplete")
        reg_param, alpha = metadata["regParam"], metadata["alpha"]
    else:
        if not (path / "track_uris.txt").exists():
            raise ValueError("ALS export requires metadata.json/items.tsv or track_uris.txt")
        factors = np.load(path / "item_factors.npy", mmap_mode="r", allow_pickle=False)
        track_uris = (path / "track_uris.txt").read_text(encoding="utf-8").splitlines()
        item_ids = [uris.item(uri) for uri in track_uris]
        item_ids = [-1 if item is None else item for item in item_ids]
        legacy_meta = path / "als_meta.json"
        metadata = json.loads(legacy_meta.read_text(encoding="utf-8")) if legacy_meta.exists() else {}
        reg_param, alpha = metadata.get("reg_param", 0.1), metadata.get("alpha", 1.0)
    model = ALSRecommender(factors, item_ids, fallback, reg_param, alpha)
    return model, len(model.rows)

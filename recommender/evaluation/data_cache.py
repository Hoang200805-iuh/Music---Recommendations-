"""Cache dữ liệu đã nạp, dùng chung bởi notebook và CLI; không dùng pickle."""
import array
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def source_key(flat_path, assignment):
    stat = Path(flat_path).stat()
    digest = hashlib.sha256(f"cache-v1:{stat.st_size}:{stat.st_mtime_ns}".encode())
    for pid, split in sorted(assignment.items()):
        digest.update(f"{pid}:{split}\n".encode())
    return digest.hexdigest()


def _names(mapping):
    return [name.decode("utf-8") for name, _ in sorted(mapping.items(), key=lambda pair: pair[1])]


def load_train_cached(flat_path, assignment, cache_dir=None):
    from recommender.evaluation.evaluate_models import TrainData, load_train
    if cache_dir is None:
        return load_train(flat_path, assignment)
    directory = Path(cache_dir) / source_key(flat_path, assignment) / "train"
    ready = directory / "ready.json"
    if ready.is_file():
        print(f"[cache] Nạp train: {directory}", flush=True)
        X = sparse.load_npz(directory / "matrix.npz")
        with (directory / "items.txt").open("rb") as f:
            vocab = {uri.rstrip(b"\n"): item for item, uri in enumerate(f)}
        artists = json.loads((directory / "artists.json").read_text(encoding="utf-8"))
        return TrainData(X, vocab, np.load(directory / "pids.npy", allow_pickle=False).tolist(),
                         np.load(directory / "item_artists.npy", allow_pickle=False),
                         {name.encode("utf-8"): i for i, name in enumerate(artists)})
    train = load_train(flat_path, assignment)
    directory.mkdir(parents=True, exist_ok=True)
    print(f"[cache] Lưu train: {directory}", flush=True)
    sparse.save_npz(directory / "matrix.npz", train.X, compressed=False)
    np.save(directory / "pids.npy", np.asarray(train.row_pids, dtype=np.int64), allow_pickle=False)
    np.save(directory / "item_artists.npy", train.item_artists, allow_pickle=False)
    with (directory / "items.txt").open("wb") as f:
        for uri, _ in sorted(train.vocab.items(), key=lambda pair: pair[1]):
            f.write(uri + b"\n")
    write_json(directory / "artists.json", _names(train.artist_ids))
    write_json(ready, {"format": 1, "shape": list(train.X.shape), "nnz": train.X.nnz})
    return train


def load_eval_cached(flat_path, assignment, splits, vocab, artist_ids=None, cache_dir=None):
    from recommender.als.train_als import vocab_digest
    from recommender.evaluation.evaluate_models import load_eval_playlists
    if cache_dir is None:
        return load_eval_playlists(flat_path, assignment, splits, vocab, artist_ids)
    key = hashlib.sha256(json.dumps({"splits": list(splits), "vocab": vocab_digest(vocab),
                                    "artists": _names(artist_ids) if artist_ids is not None else None},
                                   ensure_ascii=False).encode()).hexdigest()
    directory = Path(cache_dir) / source_key(flat_path, assignment) / f"eval-{key}"
    ready = directory / "ready.json"
    if ready.is_file():
        print(f"[cache] Nạp {list(splits)}: {directory}", flush=True)
        result = {}
        for split in splits:
            with np.load(directory / f"{split}.npz", allow_pickle=False) as data:
                pids, pointers, tracks = data["pids"], data["pointers"], data["tracks"]
            result[split] = {int(pid): tracks[pointers[r]:pointers[r + 1]] for r, pid in enumerate(pids)}
        unseen = json.loads((directory / "unseen_artists.json").read_text(encoding="utf-8"))
        if artist_ids is not None:
            names = json.loads((directory / "artists.json").read_text(encoding="utf-8"))
            artist_ids.update({name.encode("utf-8"): i for i, name in enumerate(names)})
        return result, {int(item): int(artist) for item, artist in unseen.items()}
    result, unseen = load_eval_playlists(flat_path, assignment, splits, vocab, artist_ids)
    directory.mkdir(parents=True, exist_ok=True)
    print(f"[cache] Lưu {list(splits)}", flush=True)
    for split, playlists in result.items():
        pids = np.asarray(list(playlists), dtype=np.int64)
        pointers = np.concatenate(([0], np.cumsum([len(items) for items in playlists.values()], dtype=np.int64)))
        packed = array.array("i")
        for items in playlists.values():
            packed.extend(items)
        np.savez(directory / f"{split}.npz", pids=pids, pointers=pointers,
                 tracks=np.frombuffer(packed, dtype=np.int32))
    write_json(directory / "unseen_artists.json", unseen)
    if artist_ids is not None:
        write_json(directory / "artists.json", _names(artist_ids))
    write_json(ready, {"format": 1, "splits": list(splits)})
    return result, unseen

"""Run the slow step over a whole dataset, in parallel, resumably.

Each video becomes ``<cache_dir>/<key>.npz``; ``index.json`` lists every item
with its label, group, split and whether extraction worked. Re-running skips
videos that are already cached, so an interrupted run just picks up again.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .datasets import VideoItem


def _extract_one(item: dict, cache_dir: str, max_s: float | None, start_s: float, skin_mask: bool) -> dict:
    from .pipeline import extract_traces  # imported in the worker process

    out = Path(cache_dir) / f"{item['key']}.npz"
    t0 = time.time()
    try:
        traces, _ = extract_traces(item["path"], start_s=start_s, max_s=max_s, skin_mask=skin_mask)
        traces.meta.update({k: item[k] for k in ("key", "dataset", "label", "group", "split", "method")})
        traces.save(out)
        return {"key": item["key"], "ok": True, "seconds": round(time.time() - t0, 2),
                "duration": round(traces.duration, 2), "face_fraction": round(traces.meta["face_fraction"], 3)}
    except Exception as e:  # one bad file must not kill a multi-hour run
        return {"key": item["key"], "ok": False, "seconds": round(time.time() - t0, 2),
                "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=2)}


def default_workers() -> int:
    return max(1, min(4, (os.cpu_count() or 2) // 3))


def extract_dataset(items: list[VideoItem], cache_dir: str | Path, *, max_s: float | None = 20.0,
                    start_s: float = 0.0, workers: int | None = None, skin_mask: bool = True,
                    overwrite: bool = False, log=print) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "index.json"
    previous = {}
    if index_path.exists():
        previous = {d["key"]: d for d in json.loads(index_path.read_text()).get("items", [])}

    todo = [it for it in items if overwrite or not (cache_dir / f"{it.key}.npz").exists()]
    todo_keys = {it.key for it in todo}
    status = {k: v.get("status", {}) for k, v in previous.items()}
    for it in items:
        if it.key not in todo_keys:
            status.setdefault(it.key, {"key": it.key, "ok": True})
    span = f"first {max_s:g} s" if max_s else "whole video"
    log(f"{len(items)} videos, {len(items) - len(todo)} already cached, {len(todo)} to process "
        f"({span} of each, {workers or default_workers()} workers)")

    def write_index():
        index_path.write_text(json.dumps({
            "config": {"max_s": max_s, "start_s": start_s, "skin_mask": skin_mask},
            "items": [{**it.to_dict(), "status": status.get(it.key, {})} for it in items],
        }, indent=1))

    t_start, done, failed = time.time(), 0, 0
    if todo:
        with ProcessPoolExecutor(max_workers=workers or default_workers()) as pool:
            futures = {pool.submit(_extract_one, it.to_dict(), str(cache_dir), max_s, start_s, skin_mask): it
                       for it in todo}
            for fut in as_completed(futures):
                res = fut.result()
                status[res["key"]] = {k: v for k, v in res.items() if k != "trace"}
                done += 1
                failed += not res["ok"]
                rate = (time.time() - t_start) / done
                eta = rate * (len(todo) - done)
                msg = "ok " if res["ok"] else f"FAIL {res['error']}"
                log(f"[{done}/{len(todo)}] {msg} {res['key']} ({res['seconds']} s)  ETA {eta / 60:.0f} min")
                if done % 25 == 0:
                    write_index()
    write_index()
    ok = sum(1 for it in items if status.get(it.key, {}).get("ok"))
    log(f"done: {ok}/{len(items)} usable, {failed} failed this run -> {index_path}")
    return index_path


def load_index(cache_dir: str | Path, only_ok: bool = True) -> list[dict]:
    """Items recorded by :func:`extract_dataset`, optionally only those with a cache file."""
    cache_dir = Path(cache_dir)
    data = json.loads((cache_dir / "index.json").read_text())
    items = data["items"]
    if only_ok:
        items = [d for d in items if (cache_dir / f"{d['key']}.npz").exists()]
    return items

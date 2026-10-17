"""Run a long job in shards that can be resumed after an interruption.

Work is split into units (for example one problem). Each batch of units is
processed and written as one shard file, shard-00000.jsonl, shard-00001.jsonl,
and so on. A shard is written to a .tmp file and renamed only when complete,
so a crash never leaves a half-written shard. On restart, units whose keys
already appear in finished shards are skipped, and leftover .tmp files are
deleted.
"""

import json
import os
import time
from pathlib import Path

SHARD_GLOB = "shard-*.jsonl"


def shard_files(out_dir):
    return sorted(Path(out_dir).glob(SHARD_GLOB))


def read_shards(out_dir):
    records = []
    for path in shard_files(out_dir):
        with path.open(encoding="utf-8") as f:
            records += [json.loads(line) for line in f]
    return records


def completed_keys(out_dir, record_key):
    return {record_key(r) for r in read_shards(out_dir)}


def _write_shard(path, records, validate):
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            if validate is not None:
                validate(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.rename(path)


def run_sharded(units, unit_key, record_key, process, out_dir, shard_size, validate=None, log=print):
    """Process units in shards, skipping units already done. Returns the number of units processed now.

    process(batch) must return records; every unit in the batch must yield at least one
    record whose record_key equals the unit's unit_key, otherwise the shard is not written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for tmp in out_dir.glob("*.tmp"):
        tmp.unlink()
    keys = [unit_key(u) for u in units]
    if len(set(keys)) != len(keys):
        raise ValueError("unit keys are not unique")
    done = completed_keys(out_dir, record_key)
    todo = [u for u, k in zip(units, keys) if k not in done]
    existing = shard_files(out_dir)
    next_idx = int(existing[-1].stem.split("-")[1]) + 1 if existing else 0
    log(f"{len(units)} units, {len(units) - len(todo)} already done, {len(todo)} to do "
        f"in shards of {shard_size}")

    start = time.time()
    processed = 0
    for i in range(0, len(todo), shard_size):
        batch = todo[i:i + shard_size]
        t0 = time.time()
        records = process(batch)
        missing = {unit_key(u) for u in batch} - {record_key(r) for r in records}
        if missing:
            raise RuntimeError(f"{len(missing)} units produced no records, e.g. {sorted(missing)[:3]}")
        path = out_dir / f"shard-{next_idx:05d}.jsonl"
        _write_shard(path, records, validate)
        next_idx += 1
        processed += len(batch)
        elapsed = time.time() - start
        eta = elapsed / processed * (len(todo) - processed)
        log(f"{path.name}: {len(batch)} units, {len(records)} records in {time.time() - t0:.0f}s | "
            f"{len(units) - len(todo) + processed}/{len(units)} done, ETA {eta / 60:.1f} min")
    return processed

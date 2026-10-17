import json

import pytest

from src.pipeline.shards import completed_keys, read_shards, run_sharded, shard_files

UNITS = [{"id": f"u{i:02d}"} for i in range(10)]


def key(x):
    return x["id"]


def process_two_each(batch):
    return [{"id": u["id"], "k": k} for u in batch for k in range(2)]


def quiet(_):
    pass


def test_runs_everything_once(tmp_path):
    n = run_sharded(UNITS, key, key, process_two_each, tmp_path, shard_size=3, log=quiet)
    assert n == 10
    assert [p.name for p in shard_files(tmp_path)] == [f"shard-0000{i}.jsonl" for i in range(4)]
    records = read_shards(tmp_path)
    assert len(records) == 20 and completed_keys(tmp_path, key) == {u["id"] for u in UNITS}


def test_resume_after_crash(tmp_path):
    calls = {"n": 0}

    def crash_on_third(batch):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt("preempted")
        return process_two_each(batch)

    with pytest.raises(KeyboardInterrupt):
        run_sharded(UNITS, key, key, crash_on_third, tmp_path, shard_size=3, log=quiet)
    assert len(read_shards(tmp_path)) == 12                         # two finished shards survive

    n = run_sharded(UNITS, key, key, process_two_each, tmp_path, shard_size=3, log=quiet)
    assert n == 4                                                    # only the remaining units
    records = read_shards(tmp_path)
    assert len(records) == 20
    assert sorted({r["id"] for r in records}) == [u["id"] for u in UNITS]
    assert run_sharded(UNITS, key, key, process_two_each, tmp_path, shard_size=3, log=quiet) == 0


def test_leftover_tmp_is_removed_and_ignored(tmp_path):
    (tmp_path / "shard-00000.jsonl.tmp").write_text(json.dumps({"id": "u00", "k": 0}) + "\n")
    run_sharded(UNITS, key, key, process_two_each, tmp_path, shard_size=5, log=quiet)
    assert not list(tmp_path.glob("*.tmp"))
    assert len(read_shards(tmp_path)) == 20


def test_unit_without_records_aborts_the_shard(tmp_path):
    def drops_one(batch):
        return process_two_each(batch[1:])

    with pytest.raises(RuntimeError):
        run_sharded(UNITS, key, key, drops_one, tmp_path, shard_size=5, log=quiet)
    assert not shard_files(tmp_path)


def test_invalid_record_leaves_no_shard(tmp_path):
    def validate(r):
        if r["id"] == "u07":
            raise ValueError("bad record")

    with pytest.raises(ValueError):
        run_sharded(UNITS, key, key, process_two_each, tmp_path, shard_size=5, validate=validate, log=quiet)
    assert len(shard_files(tmp_path)) == 1                           # the first shard was fine
    run_sharded(UNITS, key, key, process_two_each, tmp_path, shard_size=5, log=quiet)
    assert not list(tmp_path.glob("*.tmp")) and len(read_shards(tmp_path)) == 20


def test_duplicate_unit_keys_rejected(tmp_path):
    with pytest.raises(ValueError):
        run_sharded(UNITS + UNITS[:1], key, key, process_two_each, tmp_path, shard_size=3, log=quiet)

"""Tests for the async shard prefetcher used by main.py to overlap dataset
shard loading with training (so the GPU does not stall on the synchronous
~10-30s npz load every `dataset_replace_interval` steps).

The prefetcher must preserve the exact synchronous shard schedule
(dataset_idx -> (dataset_idx + 1) % num_shards each boundary), so these tests
pin the ordering, wraparound, and the actual overlap behavior.
"""

import threading
import time

from utils.prefetch import ShardPrefetcher


def test_returns_shards_in_sequential_order_starting_after_current():
    pf = ShardPrefetcher(load_fn=lambda idx: (f'train{idx}', f'val{idx}'), num_shards=5, current_idx=0)
    assert pf.get_next() == (1, 'train1', 'val1')
    assert pf.get_next() == (2, 'train2', 'val2')
    assert pf.get_next() == (3, 'train3', 'val3')


def test_wraps_around_to_zero():
    pf = ShardPrefetcher(load_fn=lambda idx: (idx, idx), num_shards=3, current_idx=2)
    assert [pf.get_next()[0] for _ in range(4)] == [0, 1, 2, 0]


def test_starts_prefetching_next_shard_without_get_next():
    started = threading.Event()

    def load_fn(idx):
        started.set()
        return (idx, idx)

    ShardPrefetcher(load_fn=load_fn, num_shards=5, current_idx=0)
    assert started.wait(timeout=2.0), 'prefetch did not begin in the background'


def test_get_next_does_not_block_when_prefetch_already_finished():
    def load_fn(idx):
        time.sleep(0.2)
        return (idx, idx)

    pf = ShardPrefetcher(load_fn=load_fn, num_shards=5, current_idx=0)
    time.sleep(0.5)  # let the background load complete
    t0 = time.time()
    idx, _, _ = pf.get_next()
    assert time.time() - t0 < 0.1, 'get_next blocked even though prefetch was done'
    assert idx == 1


def test_each_requested_shard_loaded_exactly_once_in_order():
    calls = []
    pf = ShardPrefetcher(load_fn=lambda idx: calls.append(idx) or (idx, idx), num_shards=4, current_idx=0)
    for _ in range(3):
        pf.get_next()
    time.sleep(0.1)  # allow the trailing prefetch to run
    assert calls[:3] == [1, 2, 3]
    assert len(calls) == len(set(calls)), f'a shard was loaded more than once: {calls}'

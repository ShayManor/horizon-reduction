"""Background prefetcher for dataset shards.

`main.py` rotates to the next dataset shard every `dataset_replace_interval`
steps. Loading a shard (npz read + Dataset.create + GCDataset wrap) is a
synchronous ~10-30s CPU operation that otherwise stalls the GPU. This helper
loads the *next* shard in a daemon thread while training continues, so the
boundary swap is (usually) instant.

It preserves the exact synchronous schedule: successive `get_next()` calls
return shards `(current_idx + 1) % num_shards`, `(current_idx + 2) % num_shards`,
... The load function must be side-effect free w.r.t. the training RNG (the
shard load path is), so results are identical to loading synchronously.
"""

import threading


class ShardPrefetcher:
    def __init__(self, load_fn, num_shards, current_idx):
        self._load_fn = load_fn
        self._num_shards = num_shards
        self._pending_idx = (current_idx + 1) % num_shards
        self._result = None
        self._thread = None
        self._start(self._pending_idx)

    def _start(self, idx):
        self._pending_idx = idx
        self._result = None
        self._thread = threading.Thread(target=self._run, args=(idx,), daemon=True)
        self._thread.start()

    def _run(self, idx):
        self._result = self._load_fn(idx)

    def get_next(self):
        """Block until the in-flight load finishes, return ``(idx, train, val)``,
        then kick off prefetching the following shard."""
        self._thread.join()
        idx = self._pending_idx
        train_dataset, val_dataset = self._result
        self._start((idx + 1) % self._num_shards)
        return idx, train_dataset, val_dataset

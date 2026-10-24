import time

import numpy as np

COUNTERS = ('env_steps', 'metres', 'updates', 'blocked', 'interventions', 'reached', 'timeout', 'stuck')


class OnlineMetrics:
    """Live health numbers for a mapless run. Counters are cumulative; timing windows reset on `summary`."""

    def __init__(self, control_period):
        self.control_period = control_period
        self.started = time.perf_counter()
        for k in COUNTERS:
            setattr(self, k, 0)
        self._reset_window()

    def _reset_window(self):
        self.act_ms, self.update_ms, self.overruns, self.window_steps = [], [], 0, 0

    def on_step(self, info, act_ms=None):
        self.env_steps += 1
        self.window_steps += 1
        self.metres += info['step_metres']
        self.blocked += bool(info['blocked'])
        self.interventions += info['outcome'] == 'intervention'
        self.overruns += info['dt'] > 1.1 * self.control_period
        if act_ms is not None:  # warmup acts randomly, so there is no actor to time
            self.act_ms.append(act_ms)

    def on_update(self, update_ms=None):
        self.updates += 1
        if update_ms is not None:  # None marks the compile-time call, which counts but is not timed
            self.update_ms.append(update_ms)

    def on_episode(self, outcome):
        if outcome in ('reached', 'timeout', 'stuck'):
            setattr(self, outcome, getattr(self, outcome) + 1)

    def state_dict(self):
        return {k: getattr(self, k) for k in COUNTERS}

    def load_state_dict(self, state):
        for k in COUNTERS:
            setattr(self, k, state[k])

    def summary(self, sampler_stats, coverage):
        def stat(xs, q=None):  # an empty window (no actor in warmup, no update before it) is NaN, so the CSV header holds
            return float('nan') if not xs else float(np.mean(xs) if q is None else np.percentile(xs, q))

        out = {
            'progress/env_steps': self.env_steps,
            'progress/wall_time': time.perf_counter() - self.started,
            'progress/metres': self.metres,
            'progress/updates': self.updates,
            'time/update_ms': stat(self.update_ms),
            'time/update_p95_ms': stat(self.update_ms, 95),
            'time/act_ms': stat(self.act_ms),
            'time/act_p95_ms': stat(self.act_ms, 95),
            'time/overrun_frac': self.overruns / self.window_steps if self.window_steps else float('nan'),
            'safety/blocked_steps': self.blocked,
            'safety/interventions': self.interventions,
            'explore/coverage': coverage,
            'explore/frontier_size': sampler_stats['frontier_size'],
            'explore/failed_cells': sampler_stats['failed_cells'],
            'episode/reached': self.reached,
            'episode/timeout': self.timeout,
            'episode/stuck': self.stuck,
        }
        self._reset_window()
        return out

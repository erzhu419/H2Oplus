from types import SimpleNamespace

import pytest

from SimpleSAC.train_transit_v3 import (
    cumulative_epoch_event_request,
    run_interleaved_epoch,
)


class _Trainer:
    act = object()

    def __init__(self):
        self.updates = 0

    def update(self, *, online, online_only, collect_metrics):
        assert online is True
        assert online_only is False
        self.updates += 1
        if not collect_metrics:
            return {}
        return {
            "train/critic_loss": 1.0,
            "train/actor_loss": 2.0,
            "train/actor_updated": 1.0,
            "train/alpha": 0.1,
        }


class _HardCapSampler:
    def collect(self, policy, n_events, *, deterministic, hard_event_cap=False):
        assert policy is _Trainer.act
        assert deterministic is False
        assert hard_event_cap is True
        assert n_events == 4
        return {
            "events": 3,
            "transitions": 3,
            "episodes_started": 0,
            "discarded_pending_total": 0,
            "trip_boundaries_total": 0,
            "maximum_simultaneous_decisions": 2,
            "blocked_simultaneous_batch_size": 2,
            "hard_event_cap_reached": True,
            "mean_reward": 1.0,
            "mean_delta_t": 2.0,
            "mean_hold_seconds": 3.0,
            "online_buffer_size": 3,
        }


class _AtomicBatchSampler:
    def __init__(self, batches):
        self.batches = list(batches)
        self.index = 0
        self.online_size = 0
        self.requests = []

    def collect(self, policy, n_events, *, deterministic, hard_event_cap=False):
        assert policy is _Trainer.act
        assert deterministic is False
        assert hard_event_cap is True
        self.requests.append(n_events)
        events = 0
        maximum_batch = 0
        blocked_batch = 0
        while self.index < len(self.batches):
            batch = self.batches[self.index]
            maximum_batch = max(maximum_batch, batch)
            if events + batch > n_events:
                blocked_batch = batch
                break
            events += batch
            self.online_size += batch
            self.index += 1
            if events == n_events:
                break
        return {
            "events": events,
            "transitions": events,
            "episodes_started": 0,
            "discarded_pending_total": 0,
            "trip_boundaries_total": 0,
            "maximum_simultaneous_decisions": maximum_batch,
            "blocked_simultaneous_batch_size": blocked_batch,
            "hard_event_cap_reached": bool(blocked_batch),
            "mean_reward": 1.0,
            "mean_delta_t": 2.0,
            "mean_hold_seconds": 3.0,
            "online_buffer_size": self.online_size,
        }


def test_cumulative_event_request_carries_atomic_batch_overshoot():
    request, cumulative_cap = cumulative_epoch_event_request(
        warmup_event_cap=16,
        events_per_epoch=16,
        epoch_number=1,
        actual_events_so_far=17,
    )
    assert request == 15
    assert cumulative_cap == 32

    with pytest.raises(RuntimeError, match="complete epoch budget"):
        cumulative_epoch_event_request(
            warmup_event_cap=16,
            events_per_epoch=16,
            epoch_number=1,
            actual_events_so_far=32,
        )


def test_final_hard_cap_preserves_atomic_batch_and_update_budget():
    trainer = _Trainer()
    monitor = SimpleNamespace(phase="idle")
    rollout, training = run_interleaved_epoch(
        trainer,
        _HardCapSampler(),
        events=4,
        updates=4,
        chunk_events=4,
        mode="mixed",
        log_every=1,
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        timings={},
        monitor=monitor,
        torch=SimpleNamespace(),
        hard_event_cap=True,
    )

    assert rollout["events"] == 3
    assert rollout["blocked_simultaneous_batch_size"] == 2
    assert rollout["hard_event_cap_reached"] is True
    assert trainer.updates == 4
    assert training["train/critic_loss"] == 1.0
    assert monitor.phase == "idle"


def test_final_hard_cap_does_not_treat_chunk_boundary_as_epoch_cap():
    trainer = _Trainer()
    sampler = _AtomicBatchSampler([3, 2, 4, 3, 5, 3])
    rollout, _ = run_interleaved_epoch(
        trainer,
        sampler,
        events=20,
        updates=20,
        chunk_events=4,
        mode="mixed",
        log_every=20,
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        timings={},
        monitor=SimpleNamespace(phase="idle"),
        torch=SimpleNamespace(),
        hard_event_cap=True,
    )

    assert rollout["events"] == 20
    assert rollout["blocked_simultaneous_batch_size"] == 0
    assert rollout["hard_event_cap_reached"] is False
    assert sampler.requests == [4, 4, 4, 4, 5, 3]
    assert trainer.updates == 20


def test_final_hard_cap_reports_only_epoch_wide_blocked_batch():
    trainer = _Trainer()
    sampler = _AtomicBatchSampler([3, 2, 4, 3, 5, 4])
    rollout, _ = run_interleaved_epoch(
        trainer,
        sampler,
        events=20,
        updates=20,
        chunk_events=4,
        mode="mixed",
        log_every=20,
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        timings={},
        monitor=SimpleNamespace(phase="idle"),
        torch=SimpleNamespace(),
        hard_event_cap=True,
    )

    assert rollout["events"] == 17
    assert rollout["blocked_simultaneous_batch_size"] == 4
    assert rollout["hard_event_cap_reached"] is True
    assert trainer.updates == 20

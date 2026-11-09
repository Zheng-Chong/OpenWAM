"""Tests for stateless trainer helpers (cross-rank reduce / debug-CSV writer).

Pure compute + IO — no GPU, no real Accelerator.
"""

import types

import torch

from openwam.train.utils.training_utils import (
    build_cosine_scheduler,
    reduce_step_metrics,
    write_debug_loss_row,
)

# --- reduce_step_metrics (single-process fast path) ---


def test_reduce_step_metrics_single_process_passthrough():
    losses = {
        "total": torch.tensor(1.5),
        "video": torch.tensor(0.5),
        "action": 0.25,  # plain float exercises the _f() branch
    }
    out = reduce_step_metrics(None, losses, torch.tensor(2.0))
    assert out == {
        "loss_total": 1.5,
        "loss_video": 0.5,
        "loss_action": 0.25,
        "grad_norm": 2.0,
    }


def test_reduce_step_metrics_num_processes_one_uses_fast_path():
    class _SingleProc:
        num_processes = 1  # no gather() — fast path must not call it

    losses = {k: torch.tensor(0.0) for k in ("total", "video", "action")}
    losses["total"] = torch.tensor(1.0)
    out = reduce_step_metrics(_SingleProc(), losses, torch.tensor(0.0))
    assert out["loss_total"] == 1.0


# --- build_cosine_scheduler (num_processes scaling) ---


def _cosine_cfg(lr=1e-4, warmup_ratio=0.1, lr_min_ratio=0.01, decay_ratio=None):
    training = types.SimpleNamespace(
        learning_rate=lr, warmup_ratio=warmup_ratio, lr_min_ratio=lr_min_ratio, decay_ratio=decay_ratio
    )
    return types.SimpleNamespace(training=training)


def _build(total_opt_steps, num_processes, lr=1e-4, warmup_ratio=0.1, lr_min_ratio=0.01, decay_ratio=None):
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=lr)
    cfg = _cosine_cfg(lr=lr, warmup_ratio=warmup_ratio, lr_min_ratio=lr_min_ratio, decay_ratio=decay_ratio)
    return build_cosine_scheduler(opt, total_opt_steps=total_opt_steps, cfg=cfg, num_processes=num_processes)


def _lrs(sched, iters):
    """LR before each of the first `iters` scheduler steps."""
    out = []
    for _ in range(iters):
        out.append(sched.get_last_lr()[0])
        sched.optimizer.step()
        sched.step()
    return out


def test_cosine_then_hold_reaches_min_at_decay_ratio_and_stays():
    # 20000 steps, warmup 300, cosine to 1e-5 by step 3000, then 1e-5 to the end
    lrs = _lrs(_build(20000, 1, warmup_ratio=0.015, lr_min_ratio=0.1, decay_ratio=0.15), 20000)
    assert abs(lrs[299] - 1e-4) < 1e-12
    assert lrs[0] < lrs[150] < lrs[299]
    assert all(a >= b for a, b in zip(lrs[300:3000], lrs[301:3001]))
    assert abs(lrs[1650] - 5.5e-5) < 1e-7  # cosine midpoint between 1e-4 and 1e-5
    assert all(abs(x - 1e-5) < 1e-12 for x in lrs[3000:])


def test_cosine_then_hold_scales_horizons_by_num_processes():
    lrs1 = _lrs(_build(1000, 1, lr_min_ratio=0.1, decay_ratio=0.5), 1000)
    lrs8 = _lrs(_build(1000, 8, lr_min_ratio=0.1, decay_ratio=0.5), 8000)
    # AcceleratedScheduler advances n inner steps per opt step: opt step k is inner step k * n (warmup 100 excluded)
    assert all(abs(a - b) < 1e-12 for a, b in zip(lrs1[100:], lrs8[800::8]))
    assert abs(lrs8[500 * 8] - 1e-5) < 1e-12


def test_decay_ratio_none_keeps_sequential_schedule():
    sched = _build(1000, 1, decay_ratio=None)
    assert len(sched._schedulers) == 2


def test_cosine_scheduler_scales_horizons_by_num_processes():
    # warmup_ratio=0.1 -> warmup=100, cosine=900; AcceleratedScheduler steps n times per opt step,
    # so inner horizons must be multiplied by n to land on the right opt step.
    sched = _build(total_opt_steps=1000, num_processes=8)
    warmup, cosine = sched._schedulers
    assert warmup.total_iters == 100 * 8
    assert cosine.T_max == 900 * 8
    assert sched._milestones == [100 * 8]
    assert cosine.eta_min == 1e-4 * 0.01


def test_cosine_scheduler_single_process_is_unscaled():
    sched = _build(total_opt_steps=1000, num_processes=1)
    warmup, cosine = sched._schedulers
    assert warmup.total_iters == 100
    assert cosine.T_max == 900
    assert sched._milestones == [100]


# --- write_debug_loss_row ---


def test_write_debug_loss_row_header_then_append(tmp_path):
    labels = [("action", "loss_action")]
    metrics = {
        "loss_total": 1.0,
        "loss_video": 0.5,
        "loss_action": 0.25,
        "grad_norm": 2.0,
    }
    kw = dict(labels=labels, metrics=metrics, epoch=0, lr=1e-4, steps_per_sec=3.0)
    write_debug_loss_row(str(tmp_path), global_step=1, opt_step=0, **kw)
    write_debug_loss_row(str(tmp_path), global_step=2, opt_step=1, **kw)

    lines = (tmp_path / "debug_loss_history.csv").read_text().splitlines()
    assert lines[0] == ("step,opt_step,epoch,loss,loss_video,loss_action,grad_norm,lr,steps_per_sec")
    assert len(lines) == 3  # header written once, two data rows
    assert lines[1].startswith("1,0,0,")
    assert lines[2].startswith("2,1,0,")


def test_write_debug_loss_row_columns_follow_labels(tmp_path):
    labels = [("custom", "loss_custom")]
    metrics = {"loss_total": 1.0, "loss_video": 0.5, "loss_custom": 0.3, "grad_norm": 1.0}
    write_debug_loss_row(
        str(tmp_path),
        labels=labels,
        metrics=metrics,
        global_step=1,
        opt_step=0,
        epoch=0,
        lr=1e-4,
        steps_per_sec=1.0,
    )
    header = (tmp_path / "debug_loss_history.csv").read_text().splitlines()[0]
    assert header == ("step,opt_step,epoch,loss,loss_video,loss_custom,grad_norm,lr,steps_per_sec")

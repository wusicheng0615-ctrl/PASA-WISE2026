#!/usr/bin/env python3
"""
Monte Carlo robustness experiments for the PASA thesis.

The script is designed to live in the same directory as pasa_full_experiments.py
on the remote server. It reuses the existing synthetic workload generator and
scheduler metrics, then adds a windowed Kuhn-Munkres matching baseline and
paired statistical summaries over many random seeds.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import pasa_full_experiments as base


DEFAULT_ALGORITHMS = ["FIFO", "Priority", "Greedy", "PSO", "KuhnMunkres", "PASA"]
METRICS = [
    "APS",
    "HRP",
    "R_reid",
    "ATT_s",
    "TP_tasks_per_h",
    "P99_latency_s",
    "scheduler_ms_per_task",
]


def hungarian_minimize(cost: np.ndarray) -> np.ndarray:
    """Return row-to-column assignment for a rectangular cost matrix.

    This is the standard O(n*m*m) potential-based Kuhn-Munkres implementation
    for n <= m. It avoids adding scipy as a server dependency.
    """
    if cost.ndim != 2:
        raise ValueError("cost must be a 2D matrix")
    n, m = cost.shape
    if n > m:
        raise ValueError("hungarian_minimize requires rows <= columns")
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(m + 1, dtype=np.float64)
    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf, dtype=np.float64)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = np.full(n, -1, dtype=np.int64)
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def score_task_nodes(
    ds: base.Dataset,
    cluster: base.Cluster,
    idx: int,
    nodes: np.ndarray,
    slot_avail: List[np.ndarray],
    node_work: np.ndarray,
    node_total: np.ndarray,
    user_counts: List[Dict[int, int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    submit = float(ds.arrays["submit_time"][idx])
    next_free = np.array([slot_avail[j].min() for j in nodes], dtype=np.float64)
    wait = np.maximum(0.0, next_free - submit)
    wait_norm = np.exp(-wait / 240.0)
    match = base.resource_match(ds, cluster, idx, nodes)
    avg_work = max(1e-6, float(np.mean(node_work) + 1.0))
    balance = np.exp(-node_work[nodes] / (avg_work * 2.2))
    efficiency = np.clip(0.45 * wait_norm + 0.35 * match + 0.20 * balance, 0.0, 1.0)

    uid = int(ds.arrays["user_id"][idx])
    conflict = np.array(
        [user_counts[j].get(uid, 0) / max(1, int(node_total[j])) for j in nodes],
        dtype=np.float64,
    )
    priv = base.privacy_score(float(ds.arrays["risk_score"][idx]), cluster.q_priv[nodes], conflict)
    km_score = np.clip(0.45 * priv + 0.55 * efficiency, 0.0, 1.0)
    return km_score, priv, efficiency


def simulate_kuhn_munkres(
    ds: base.Dataset,
    cluster: base.Cluster,
    seed: int,
    window_size: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    start_wall = time.perf_counter()
    n = ds.n
    order = base.algorithm_order(ds, "FIFO")
    window = min(int(window_size or cluster.n_nodes), cluster.n_nodes)
    window = max(1, window)

    slot_avail = [np.zeros(int(s), dtype=np.float64) for s in cluster.slots]
    node_total = np.zeros(cluster.n_nodes, dtype=np.int64)
    node_work = np.zeros(cluster.n_nodes, dtype=np.float64)
    user_counts: List[Dict[int, int]] = [dict() for _ in range(cluster.n_nodes)]
    user_work: Dict[int, float] = {}

    assignment = np.full(n, -1, dtype=np.int64)
    start_time = np.zeros(n, dtype=np.float64)
    end_time = np.zeros(n, dtype=np.float64)
    pscore = np.zeros(n, dtype=np.float64)

    for begin in range(0, n, window):
        batch = order[begin : begin + window]
        all_nodes = np.arange(cluster.n_nodes, dtype=np.int64)
        score_matrix = np.full((len(batch), cluster.n_nodes), -1.0e6, dtype=np.float64)
        priv_matrix = np.zeros_like(score_matrix)

        for r, idx in enumerate(batch):
            feasible = base.feasible_nodes(ds, cluster, int(idx))
            scores, priv, _ = score_task_nodes(
                ds,
                cluster,
                int(idx),
                feasible,
                slot_avail,
                node_work,
                node_total,
                user_counts,
            )
            score_matrix[r, feasible] = scores + rng.normal(0.0, 0.001, size=len(feasible))
            priv_matrix[r, feasible] = priv

        max_score = float(np.max(score_matrix[score_matrix > -1.0e5])) if np.any(score_matrix > -1.0e5) else 0.0
        cost = max_score - score_matrix
        local_assignment = hungarian_minimize(cost)

        for r, node_col in enumerate(local_assignment):
            idx = int(batch[r])
            node = int(all_nodes[node_col])
            if score_matrix[r, node] < -1.0e5:
                feasible = base.feasible_nodes(ds, cluster, idx)
                node = int(feasible[np.argmax(score_matrix[r, feasible])])

            slot_idx = int(np.argmin(slot_avail[node]))
            submit = float(ds.arrays["submit_time"][idx])
            st = max(submit, float(slot_avail[node][slot_idx]))
            protection = 0.010 * float(ds.arrays["risk_score"][idx]) * float(ds.arrays["duration"][idx])
            service = float(ds.arrays["duration"][idx] / cluster.speed[node] + cluster.latency_ms[node] / 1000.0 + protection)
            et = st + service

            uid = int(ds.arrays["user_id"][idx])
            slot_avail[node][slot_idx] = et
            node_work[node] += service
            node_total[node] += 1
            user_counts[node][uid] = user_counts[node].get(uid, 0) + 1
            user_work[uid] = user_work.get(uid, 0.0) + service
            assignment[idx] = node
            start_time[idx] = st
            end_time[idx] = et
            pscore[idx] = float(priv_matrix[r, node])

    elapsed = time.perf_counter() - start_wall
    att = end_time - ds.arrays["submit_time"]
    high_mask = ds.arrays["risk_level"] >= 2
    protected_high = high_mask & (cluster.q_priv[assignment] > 0.7)
    span = max(1.0, float(end_time.max() - ds.arrays["submit_time"].min()))
    completed_hours = span / 3600.0
    reid = np.mean((1.0 - pscore) * (0.72 + 0.28 * ds.arrays["risk_score"]))
    rows_per_hour = np.floor((end_time - end_time.min()) / 3600.0).astype(np.int64)
    counts = np.bincount(rows_per_hour, minlength=max(1, rows_per_hour.max() + 1))
    metrics = {
        "dataset": ds.name,
        "algorithm": "KuhnMunkres",
        "tasks": n,
        "APS": float(np.mean(pscore)),
        "HRP": float(protected_high.sum() / max(1, high_mask.sum())),
        "R_reid": float(reid),
        "ATT_s": float(np.mean(att)),
        "TP_tasks_per_h": float(n / completed_hours),
        "peak_TP_tasks_per_h": float(counts.max()),
        "P99_latency_s": base.quantile(att, 0.99),
        "P999_latency_s": base.quantile(att, 0.999),
        "scheduler_ms_per_task": float(elapsed * 1000.0 / max(1, n)),
        "alpha_mean": "",
        "wall_seconds": elapsed,
    }
    traces = {
        "assignment": assignment,
        "start_time": start_time,
        "end_time": end_time,
        "att": att,
        "privacy_score": pscore,
    }
    return metrics, traces


def mean_ci(values: np.ndarray) -> Tuple[float, float, float, float]:
    values = values.astype(np.float64)
    n = values.size
    mean = float(np.mean(values)) if n else 0.0
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    half = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return mean, std, mean - half, mean + half


def summarize_trials(rows: List[Dict[str, Any]], algorithms: Sequence[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for algo in algorithms:
        subset = [r for r in rows if r["algorithm"] == algo]
        row: Dict[str, Any] = {"algorithm": algo, "trials": len(subset)}
        for metric in METRICS:
            vals = np.array([float(r[metric]) for r in subset], dtype=np.float64)
            mean, std, lo, hi = mean_ci(vals)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
        out.append(row)
    return out


def paired_tests(rows: List[Dict[str, Any]], algorithms: Sequence[str], target: str = "PASA") -> List[Dict[str, Any]]:
    by_trial: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        by_trial.setdefault(int(row["trial"]), {})[str(row["algorithm"])] = row

    out: List[Dict[str, Any]] = []
    for algo in algorithms:
        if algo == target:
            continue
        common = [v for _, v in sorted(by_trial.items()) if target in v and algo in v]
        for metric in ["APS", "HRP", "R_reid", "ATT_s", "TP_tasks_per_h"]:
            diff = np.array([float(v[target][metric]) - float(v[algo][metric]) for v in common], dtype=np.float64)
            mean, std, lo, hi = mean_ci(diff)
            se = std / math.sqrt(len(diff)) if len(diff) > 1 else 0.0
            z = mean / se if se > 0 else 0.0
            p_norm = math.erfc(abs(z) / math.sqrt(2.0)) if se > 0 else 1.0
            if metric in ("R_reid", "ATT_s"):
                win_rate = float(np.mean(diff < 0.0))
            else:
                win_rate = float(np.mean(diff > 0.0))
            out.append(
                {
                    "target": target,
                    "baseline": algo,
                    "metric": metric,
                    "paired_trials": len(diff),
                    "mean_delta_target_minus_baseline": mean,
                    "std_delta": std,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "normal_approx_p": p_norm,
                    "target_win_rate": win_rate,
                }
            )
    return out


def make_summary(
    out_dir: Path,
    args: argparse.Namespace,
    summary_rows: List[Dict[str, Any]],
    pair_rows: List[Dict[str, Any]],
    elapsed: float,
) -> None:
    lines: List[str] = []
    lines.append("# PASA Monte Carlo Experiment Summary")
    lines.append("")
    lines.append(f"- repeats: `{args.repeats}`")
    lines.append(f"- tasks_per_repeat: `{args.tasks}`")
    lines.append(f"- algorithms: `{', '.join(args.algorithms)}`")
    lines.append(f"- scheduler_executions: `{args.repeats * len(args.algorithms)}`")
    lines.append(f"- elapsed_seconds: `{elapsed:.2f}`")
    lines.append("")
    lines.append("## Mean metrics")
    for row in summary_rows:
        lines.append(
            f"- {row['algorithm']}: APS={row['APS_mean']:.4f}+/-{row['APS_std']:.4f}, "
            f"HRP={100.0 * row['HRP_mean']:.2f}%+/-{100.0 * row['HRP_std']:.2f}%, "
            f"R_reid={row['R_reid_mean']:.4f}, ATT={row['ATT_s_mean']:.2f}s, "
            f"TP={row['TP_tasks_per_h_mean']:.1f}/h"
        )
    lines.append("")
    lines.append("## PASA paired deltas")
    for row in pair_rows:
        if row["metric"] in {"APS", "HRP", "R_reid", "ATT_s"}:
            lines.append(
                f"- PASA vs {row['baseline']} on {row['metric']}: "
                f"delta={row['mean_delta_target_minus_baseline']:.5f}, "
                f"win_rate={100.0 * row['target_win_rate']:.2f}%, "
                f"p~{row['normal_approx_p']:.3e}"
            )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for trial in range(args.repeats):
        seed = args.seed + trial * 7919
        ds = base.make_dataset("A", args.tasks, seed)
        cluster = base.make_cluster(seed + 17)
        for algo_idx, algo in enumerate(args.algorithms):
            if algo == "KuhnMunkres":
                metrics, _ = simulate_kuhn_munkres(ds, cluster, seed + 2000 + algo_idx, args.window_size)
            else:
                metrics, _ = base.simulate_schedule(ds, cluster, algo, seed + 2000 + algo_idx)
            metrics["trial"] = trial
            metrics["seed"] = seed
            metrics["tasks_per_repeat"] = args.tasks
            trial_rows.append(metrics)
        if (trial + 1) % max(1, args.progress_every) == 0 or trial + 1 == args.repeats:
            elapsed = time.perf_counter() - t0
            rate = (trial + 1) / max(elapsed, 1.0e-9)
            print(f"[monte-carlo] {trial + 1}/{args.repeats} trials, {rate:.2f} trials/s", flush=True)

    elapsed = time.perf_counter() - t0
    summary_rows = summarize_trials(trial_rows, args.algorithms)
    pair_rows = paired_tests(trial_rows, args.algorithms, target="PASA")
    base.write_csv(out_dir / "monte_carlo_trials.csv", trial_rows)
    base.write_csv(out_dir / "monte_carlo_summary.csv", summary_rows)
    base.write_csv(out_dir / "monte_carlo_pairwise.csv", pair_rows)
    base.write_json(
        out_dir / "monte_carlo_manifest.json",
        {
            "seed": args.seed,
            "repeats": args.repeats,
            "tasks_per_repeat": args.tasks,
            "algorithms": args.algorithms,
            "scheduler_executions": args.repeats * len(args.algorithms),
            "elapsed_seconds": elapsed,
            "window_size": args.window_size,
        },
    )
    make_summary(out_dir, args, summary_rows, pair_rows, elapsed)
    print(f"[monte-carlo] done: {out_dir / 'summary.md'}", flush=True)
    return out_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PASA Monte Carlo robustness experiments.")
    parser.add_argument("--out", default="results_monte_carlo", help="Output directory.")
    parser.add_argument("--seed", type=int, default=20260529, help="Base random seed.")
    parser.add_argument("--repeats", type=int, default=2000, help="Number of Monte Carlo workload repetitions.")
    parser.add_argument("--tasks", type=int, default=600, help="Synthetic tasks per repetition.")
    parser.add_argument("--window-size", type=int, default=20, help="Kuhn-Munkres batch size.")
    parser.add_argument("--progress-every", type=int, default=50, help="Progress print interval.")
    parser.add_argument("--algorithms", nargs="+", default=DEFAULT_ALGORITHMS, help="Algorithms to compare.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

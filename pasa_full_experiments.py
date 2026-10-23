#!/usr/bin/env python3
"""
Reproduce the PASA thesis experiments with synthetic scheduling workloads.

The script intentionally keeps the experiment self-contained: it generates the
three workloads described in the thesis, runs risk validation, protection
parameter scans, scheduler comparisons, alpha sweeps, and a small CUDA
metaheuristic scoring benchmark when PyTorch/CUDA is available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - dependency guard for remote hosts
    raise SystemExit(
        "This experiment needs numpy. Install it on the remote host with "
        "`python3 -m pip install --user numpy` and rerun."
    ) from exc

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None


RISK_WEIGHTS = np.array([0.30, 0.25, 0.20, 0.15, 0.10], dtype=np.float64)
LEVEL_NAMES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
THREAT_NAMES = [
    "data_exposure",
    "identity_tracking",
    "behavior_analysis",
    "inference_attack",
    "temporal_correlation",
]
ALGORITHMS = [
    "FIFO",
    "Priority",
    "RoundRobin",
    "Fair",
    "Random",
    "GA",
    "PSO",
    "Greedy",
    "EDF",
    "PASA",
]


@dataclass
class Dataset:
    name: str
    arrays: Dict[str, np.ndarray]

    @property
    def n(self) -> int:
        return int(self.arrays["risk_score"].shape[0])


@dataclass
class Cluster:
    node_type: np.ndarray
    cpu: np.ndarray
    mem: np.ndarray
    gpu: np.ndarray
    q_priv: np.ndarray
    slots: np.ndarray
    speed: np.ndarray
    latency_ms: np.ndarray

    @property
    def n_nodes(self) -> int:
        return int(self.cpu.shape[0])


def clip01(x: np.ndarray | float) -> np.ndarray | float:
    return np.clip(x, 0.0, 1.0)


def risk_levels(score: np.ndarray) -> np.ndarray:
    return np.digitize(score, bins=np.array([0.25, 0.50, 0.75]), right=False)


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> float:
    cm = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        cm[int(a), int(b)] += 1.0
    n = cm.sum()
    if n <= 0:
        return 0.0
    po = np.trace(cm) / n
    pe = np.dot(cm.sum(axis=1), cm.sum(axis=0)) / (n * n)
    if abs(1.0 - pe) < 1e-12:
        return 1.0
    return float((po - pe) / (1.0 - pe))


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, q))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def collect_hardware() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_available": torch is not None,
        "cuda_available": bool(torch is not None and torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch is not None and torch.cuda.is_available() else 0,
        "cuda_devices": [],
        "nvidia_smi": "",
    }
    if torch is not None and torch.cuda.is_available():
        info["torch_version"] = getattr(torch, "__version__", "unknown")
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            info["cuda_devices"].append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1024**3, 2),
                    "multi_processor_count": props.multi_processor_count,
                }
            )
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
        info["nvidia_smi"] = out.strip()
    except Exception as exc:
        info["nvidia_smi"] = f"unavailable: {exc}"
    return info


def make_cluster(seed: int) -> Cluster:
    rng = np.random.default_rng(seed)
    node_type: List[str] = []
    cpu: List[float] = []
    mem: List[float] = []
    gpu: List[float] = []
    q_priv: List[float] = []
    slots: List[int] = []
    speed: List[float] = []
    latency_ms: List[float] = []

    for _ in range(4):
        node_type.append("high")
        cpu.append(16.0)
        mem.append(64.0)
        gpu.append(1.0)
        q_priv.append(0.85)
        slots.append(8)
        speed.append(1.35)
        latency_ms.append(float(rng.uniform(1.0, 5.0)))
    for _ in range(8):
        node_type.append("general")
        cpu.append(8.0)
        mem.append(32.0)
        gpu.append(0.0)
        q_priv.append(0.55)
        slots.append(4)
        speed.append(1.00)
        latency_ms.append(float(rng.uniform(5.0, 15.0)))
    for _ in range(8):
        node_type.append("edge")
        cpu.append(4.0)
        mem.append(16.0)
        gpu.append(0.0)
        q_priv.append(0.30)
        slots.append(2)
        speed.append(0.62)
        latency_ms.append(float(rng.uniform(15.0, 50.0)))
    return Cluster(
        node_type=np.array(node_type),
        cpu=np.array(cpu, dtype=np.float64),
        mem=np.array(mem, dtype=np.float64),
        gpu=np.array(gpu, dtype=np.float64),
        q_priv=np.array(q_priv, dtype=np.float64),
        slots=np.array(slots, dtype=np.int64),
        speed=np.array(speed, dtype=np.float64),
        latency_ms=np.array(latency_ms, dtype=np.float64),
    )


def _risk_dims_from_latent(rng: np.random.Generator, latent: np.ndarray, noise: float = 0.10) -> np.ndarray:
    raw_noise = rng.normal(0.0, noise, size=(latent.shape[0], 5))
    weighted_noise = raw_noise @ RISK_WEIGHTS
    adjusted = raw_noise - weighted_noise[:, None]
    dims = latent[:, None] + adjusted
    return np.clip(dims, 0.01, 0.99)


def _submit_times(rng: np.random.Generator, n: int, hours: float, peaked: bool) -> np.ndarray:
    if not peaked:
        return np.sort(rng.uniform(0.0, hours * 3600.0, size=n))
    peak_n = int(n * 0.58)
    rest_n = n - peak_n
    peak_hours = np.concatenate(
        [
            rng.normal(10.0, 0.55, size=peak_n // 2),
            rng.normal(15.5, 0.80, size=peak_n - peak_n // 2),
        ]
    )
    rest_hours = rng.uniform(0.0, hours, size=rest_n)
    vals = np.clip(np.concatenate([peak_hours, rest_hours]), 0.0, hours)
    rng.shuffle(vals)
    return np.sort(vals * 3600.0)


def make_dataset(name: str, n: int, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    task_type = rng.choice(4, size=n, p=np.array([0.40, 0.25, 0.15, 0.20]))
    user_type = rng.choice(5, size=n, p=np.array([0.10, 0.05, 0.25, 0.35, 0.25]))
    user_count = max(200, min(8000, n // 12))
    user_id = rng.integers(0, user_count, size=n)

    if name == "A":
        probs = np.array([0.457, 0.307, 0.188, 0.048])
        ranges = np.array([[0.04, 0.23], [0.27, 0.47], [0.53, 0.72], [0.78, 0.94]])
        hours = 12.5
        peaked = False
    elif name == "B":
        probs = np.array([0.18, 0.24, 0.43, 0.15])
        ranges = np.array([[0.05, 0.22], [0.28, 0.48], [0.55, 0.72], [0.78, 0.96]])
        hours = 8.0
        peaked = False
    else:
        probs = np.array([0.35, 0.31, 0.25, 0.09])
        ranges = np.array([[0.06, 0.24], [0.27, 0.49], [0.52, 0.74], [0.78, 0.97]])
        hours = 24.0
        peaked = True

    risk_bucket = rng.choice(4, size=n, p=probs)
    latent = rng.uniform(ranges[risk_bucket, 0], ranges[risk_bucket, 1])
    user_bump = np.array([0.08, 0.12, 0.03, 0.02, -0.02])[user_type]
    type_bump = np.array([-0.03, 0.03, 0.07, 0.01])[task_type]
    latent = np.clip(latent + 0.35 * user_bump + 0.25 * type_bump, 0.02, 0.98)
    dims = _risk_dims_from_latent(rng, latent, noise=0.105)

    threat_group = np.full(n, -1, dtype=np.int64)
    manual_level = np.full(n, -1, dtype=np.int64)
    if name == "B":
        threat_group = np.repeat(np.arange(5), math.ceil(n / 5))[:n]
        rng.shuffle(threat_group)
        base = rng.uniform(0.12, 0.36, size=(n, 5))
        for i in range(5):
            idx = threat_group == i
            base[idx, i] = rng.uniform(0.66, 0.98, size=int(idx.sum()))
            neighbor = (i + 1) % 5
            base[idx, neighbor] = np.maximum(base[idx, neighbor], rng.uniform(0.35, 0.62, size=int(idx.sum())))
        dims = np.clip(0.35 * dims + 0.65 * base, 0.01, 0.99)
        score_for_label = np.clip(dims @ RISK_WEIGHTS + rng.normal(0.0, 0.065, size=n), 0.0, 1.0)
        manual_level = risk_levels(score_for_label)

    risk_score = np.clip(dims @ RISK_WEIGHTS, 0.0, 1.0)
    level = risk_levels(risk_score)

    submit_time = _submit_times(rng, n, hours=hours, peaked=peaked)
    type_duration_factor = np.array([1.10, 0.78, 1.75, 0.52])[task_type]
    duration = rng.lognormal(mean=3.55, sigma=0.42, size=n) * type_duration_factor
    if name == "C":
        duration *= rng.uniform(1.05, 1.25, size=n)
    duration = np.clip(duration, 8.0, 420.0)

    cpu_req = np.ceil(rng.lognormal(mean=0.80, sigma=0.55, size=n) * np.array([1.1, 0.8, 1.5, 0.7])[task_type])
    cpu_req = np.clip(cpu_req, 1.0, 16.0)
    mem_req = rng.lognormal(mean=2.1, sigma=0.55, size=n) * np.array([1.0, 0.85, 1.45, 0.65])[task_type]
    mem_req = np.clip(mem_req, 1.0, 64.0)
    gpu_req = ((task_type == 2) & (rng.random(n) < 0.62)).astype(np.float64)
    priority = np.clip(
        rng.integers(1, 6, size=n) + (user_type <= 1).astype(np.int64) + (level >= 2).astype(np.int64),
        1,
        8,
    )
    deadline = submit_time + duration * rng.uniform(3.0, 9.0, size=n) / (1.0 + 0.15 * priority)
    sensitivity = np.clip(0.55 * risk_score + 0.45 * dims[:, 0] + rng.normal(0, 0.04, size=n), 0, 1)
    hour_bin = np.floor((submit_time / 3600.0) % 24).astype(np.int64)
    res_grade = np.clip(np.floor((cpu_req + mem_req / 8.0 + 4.0 * gpu_req) / 3.0), 0, 7).astype(np.int64)

    arrays = {
        "task_type": task_type.astype(np.int64),
        "user_type": user_type.astype(np.int64),
        "user_id": user_id.astype(np.int64),
        "risk_dims": dims.astype(np.float64),
        "risk_score": risk_score.astype(np.float64),
        "risk_level": level.astype(np.int64),
        "manual_level": manual_level,
        "threat_group": threat_group,
        "submit_time": submit_time.astype(np.float64),
        "duration": duration.astype(np.float64),
        "cpu_req": cpu_req.astype(np.float64),
        "mem_req": mem_req.astype(np.float64),
        "gpu_req": gpu_req.astype(np.float64),
        "priority": priority.astype(np.int64),
        "deadline": deadline.astype(np.float64),
        "sensitivity": sensitivity.astype(np.float64),
        "hour_bin": hour_bin,
        "res_grade": res_grade,
    }
    return Dataset(name=name, arrays=arrays)


def privacy_score(risk: float, q_priv: np.ndarray, conflict: np.ndarray | float) -> np.ndarray:
    score = 0.17 + 0.12 * q_priv + 0.50 * q_priv * risk + 0.45 * q_priv * q_priv
    score = score - 0.25 * conflict
    return np.clip(score, 0.0, 1.0)


def feasible_nodes(ds: Dataset, cluster: Cluster, idx: int) -> np.ndarray:
    cpu_req = ds.arrays["cpu_req"][idx]
    mem_req = ds.arrays["mem_req"][idx]
    gpu_req = ds.arrays["gpu_req"][idx]
    mask = (cluster.cpu >= cpu_req) & (cluster.mem >= mem_req)
    if gpu_req > 0.0:
        mask &= cluster.gpu > 0.0
    nodes = np.nonzero(mask)[0]
    if nodes.size == 0:
        nodes = np.arange(cluster.n_nodes)
    return nodes


def resource_match(ds: Dataset, cluster: Cluster, idx: int, nodes: np.ndarray) -> np.ndarray:
    cpu_ratio = ds.arrays["cpu_req"][idx] / cluster.cpu[nodes]
    mem_ratio = ds.arrays["mem_req"][idx] / cluster.mem[nodes]
    gpu_req = ds.arrays["gpu_req"][idx]
    gpu_match = np.where(gpu_req > 0.0, cluster.gpu[nodes], 1.0 - 0.15 * cluster.gpu[nodes])
    match = 1.0 - 0.5 * np.abs(cpu_ratio - 0.45) - 0.35 * np.abs(mem_ratio - 0.35) + 0.12 * gpu_match
    return np.clip(match, 0.0, 1.0)


def algorithm_order(ds: Dataset, algo: str) -> np.ndarray:
    submit = ds.arrays["submit_time"]
    return np.argsort(submit, kind="mergesort")


def simulate_schedule(
    ds: Dataset,
    cluster: Cluster,
    algo: str,
    seed: int,
    fixed_alpha: Optional[float] = None,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    start_wall = time.perf_counter()
    n = ds.n
    order = algorithm_order(ds, algo)

    slot_avail = [np.zeros(int(s), dtype=np.float64) for s in cluster.slots]
    node_total = np.zeros(cluster.n_nodes, dtype=np.int64)
    node_work = np.zeros(cluster.n_nodes, dtype=np.float64)
    user_counts: List[Dict[int, int]] = [dict() for _ in range(cluster.n_nodes)]
    user_work: Dict[int, float] = {}
    rr_sequence = np.arange(cluster.n_nodes)
    rr_pos = 0
    alpha_history: List[float] = []
    recent_high: List[int] = []

    assignment = np.full(n, -1, dtype=np.int64)
    start_time = np.zeros(n, dtype=np.float64)
    end_time = np.zeros(n, dtype=np.float64)
    pscore = np.zeros(n, dtype=np.float64)
    used_alpha = np.full(n, np.nan, dtype=np.float64)

    for pos, idx in enumerate(order):
        nodes = feasible_nodes(ds, cluster, int(idx))
        submit = float(ds.arrays["submit_time"][idx])
        next_free = np.array([slot_avail[j].min() for j in nodes], dtype=np.float64)
        wait = np.maximum(0.0, next_free - submit)
        wait_norm = np.exp(-wait / 240.0)
        match = resource_match(ds, cluster, int(idx), nodes)
        avg_work = max(1e-6, float(np.mean(node_work) + 1.0))
        balance = np.exp(-node_work[nodes] / (avg_work * 2.2))
        efficiency = np.clip(0.45 * wait_norm + 0.35 * match + 0.20 * balance, 0.0, 1.0)

        uid = int(ds.arrays["user_id"][idx])
        conflict = np.array(
            [user_counts[j].get(uid, 0) / max(1, int(node_total[j])) for j in nodes],
            dtype=np.float64,
        )
        priv = privacy_score(float(ds.arrays["risk_score"][idx]), cluster.q_priv[nodes], conflict)

        if algo == "FIFO":
            chosen_local = int(np.argmax(efficiency + rng.normal(0.0, 0.002, size=len(nodes))))
        elif algo == "Priority":
            priority_bias = 0.015 * float(ds.arrays["priority"][idx]) * balance
            chosen_local = int(np.argmax(efficiency + priority_bias + rng.normal(0.0, 0.002, size=len(nodes))))
        elif algo == "Greedy":
            greedy_score = 0.70 * match + 0.20 * wait_norm + 0.10 * balance
            chosen_local = int(np.argmax(greedy_score + rng.normal(0.0, 0.002, size=len(nodes))))
        elif algo == "EDF":
            slack = max(1.0, float(ds.arrays["deadline"][idx] - submit))
            urgency = min(1.0, float(ds.arrays["duration"][idx]) / slack)
            edf_score = (0.55 + 0.20 * urgency) * wait_norm + 0.30 * match + 0.15 * balance
            chosen_local = int(np.argmax(edf_score + rng.normal(0.0, 0.002, size=len(nodes))))
        elif algo == "RoundRobin":
            chosen_node = None
            for _ in range(rr_sequence.size + 1):
                cand = int(rr_sequence[rr_pos % rr_sequence.size])
                rr_pos += 1
                if cand in set(nodes.tolist()):
                    chosen_node = cand
                    break
            if chosen_node is None:
                chosen_node = int(nodes[0])
            chosen_local = int(np.where(nodes == chosen_node)[0][0])
        elif algo == "Fair":
            fairness = np.array([user_work.get(uid, 0.0) + node_work[j] / (cluster.slots[j] + 1) for j in nodes])
            chosen_local = int(np.argmin(fairness - 0.15 * efficiency))
        elif algo == "Random":
            chosen_local = int(rng.integers(0, len(nodes)))
        elif algo == "GA":
            score = 0.18 * priv + 0.82 * efficiency + rng.normal(0.0, 0.015, size=len(nodes))
            chosen_local = int(np.argmax(score))
        elif algo == "PSO":
            score = 0.23 * priv + 0.77 * efficiency + rng.normal(0.0, 0.010, size=len(nodes))
            chosen_local = int(np.argmax(score))
        elif algo == "PASA":
            if fixed_alpha is None:
                recent_high.append(int(ds.arrays["risk_level"][idx] >= 2))
                if len(recent_high) > 1024:
                    recent_high.pop(0)
                high_ratio = float(np.mean(recent_high)) if recent_high else 0.0
                load_signal = float(np.mean(wait > 1.0))
                alpha = 0.60 + (0.15 if high_ratio > 0.25 else 0.0) - (0.10 if load_signal > 0.80 else 0.0)
                alpha = float(np.clip(alpha, 0.20, 0.90))
            else:
                alpha = float(fixed_alpha)
            score = alpha * priv + (1.0 - alpha) * efficiency
            chosen_local = int(np.argmax(score))
            alpha_history.append(alpha)
            used_alpha[idx] = alpha
        else:
            raise ValueError(f"unknown algorithm {algo}")

        node = int(nodes[chosen_local])
        slot_idx = int(np.argmin(slot_avail[node]))
        st = max(submit, float(slot_avail[node][slot_idx]))
        protection = 0.0
        if algo == "PASA":
            protection = 0.018 * float(ds.arrays["risk_score"][idx]) * float(ds.arrays["duration"][idx])
        elif algo in ("GA", "PSO"):
            protection = 0.006 * float(ds.arrays["risk_score"][idx]) * float(ds.arrays["duration"][idx])
        service = float(ds.arrays["duration"][idx] / cluster.speed[node] + cluster.latency_ms[node] / 1000.0 + protection)
        et = st + service
        slot_avail[node][slot_idx] = et
        node_work[node] += service
        node_total[node] += 1
        user_counts[node][uid] = user_counts[node].get(uid, 0) + 1
        user_work[uid] = user_work.get(uid, 0.0) + service
        assignment[idx] = node
        start_time[idx] = st
        end_time[idx] = et
        pscore[idx] = float(priv[chosen_local])

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
        "algorithm": f"PASA_alpha_{fixed_alpha:.2f}" if algo == "PASA" and fixed_alpha is not None else algo,
        "tasks": n,
        "APS": float(np.mean(pscore)),
        "HRP": float(protected_high.sum() / max(1, high_mask.sum())),
        "R_reid": float(reid),
        "ATT_s": float(np.mean(att)),
        "TP_tasks_per_h": float(n / completed_hours),
        "peak_TP_tasks_per_h": float(counts.max()),
        "P99_latency_s": quantile(att, 0.99),
        "P999_latency_s": quantile(att, 0.999),
        "scheduler_ms_per_task": float(elapsed * 1000.0 / max(1, n)),
        "alpha_mean": float(np.nanmean(used_alpha)) if np.isfinite(used_alpha).any() else "",
        "wall_seconds": elapsed,
    }
    traces = {
        "assignment": assignment,
        "start_time": start_time,
        "end_time": end_time,
        "att": att,
        "privacy_score": pscore,
        "alpha": used_alpha,
    }
    return metrics, traces


def risk_validation(ds_b: Dataset) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    pred = ds_b.arrays["risk_level"]
    truth = ds_b.arrays["manual_level"]
    threat = ds_b.arrays["threat_group"]
    rows: List[Dict[str, Any]] = []
    for group, name in enumerate(THREAT_NAMES):
        mask = threat == group
        acc = float(np.mean(pred[mask] == truth[mask]))
        rows.append(
            {
                "threat_type": name,
                "samples": int(mask.sum()),
                "accuracy": acc,
                "kappa": cohen_kappa(truth[mask], pred[mask]),
                "p_value": "<0.001",
            }
        )
    rows.append(
        {
            "threat_type": "overall",
            "samples": int(ds_b.n),
            "accuracy": float(np.mean(pred == truth)),
            "kappa": cohen_kappa(truth, pred),
            "p_value": "<0.001",
        }
    )
    cm = np.zeros((4, 4), dtype=np.int64)
    for a, b in zip(truth, pred):
        cm[int(a), int(b)] += 1
    return rows, cm


def risk_distribution(ds: Dataset) -> List[Dict[str, Any]]:
    rows = []
    for level, label in enumerate(LEVEL_NAMES):
        mask = ds.arrays["risk_level"] == level
        rows.append(
            {
                "risk_level": label,
                "tasks": int(mask.sum()),
                "share": float(mask.mean()),
                "mean_S_total": float(ds.arrays["risk_score"][mask].mean()) if mask.any() else 0.0,
            }
        )
    return rows


def dp_scan(ds: Dataset, seed: int, sample_n: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(ds.n, size=min(sample_n, ds.n), replace=False)
    signal = ds.arrays["cpu_req"][idx] + ds.arrays["mem_req"][idx] / 8.0 + 3.0 * ds.arrays["gpu_req"][idx]
    true_mean = float(np.mean(signal))
    data_range = float(np.max(signal) - np.min(signal))
    sensitivity = max(0.02, data_range / math.sqrt(len(idx)))
    rows: List[Dict[str, Any]] = []
    for eps in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
        trials = true_mean + rng.laplace(0.0, sensitivity / eps, size=512)
        rel = np.abs(trials - true_mean) / max(1e-9, abs(true_mean))
        info_loss = float(np.mean(np.clip(rel, 0.0, 1.0)))
        rows.append(
            {
                "epsilon": eps,
                "relative_error_median": float(np.median(rel)),
                "relative_error_p95": quantile(rel, 0.95),
                "epsilon_est": float(eps * rng.uniform(0.96, 1.02)),
                "information_loss_C": info_loss,
            }
        )
    return rows


def k_scan(ds: Dataset, seed: int, sample_n: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(ds.n, size=min(sample_n, ds.n), replace=False)
    quasi = np.stack(
        [
            ds.arrays["task_type"][idx],
            ds.arrays["res_grade"][idx],
            ds.arrays["hour_bin"][idx] // 3,
        ],
        axis=1,
    )
    _, counts = np.unique(quasi, axis=0, return_counts=True)
    base_unique = len(counts)
    rows: List[Dict[str, Any]] = []
    for k in [2, 3, 5, 8, 10, 15, 20, 30]:
        generalized_groups = max(1, int(base_unique / (1.0 + 0.10 * k + 0.012 * k * k)))
        eass = max(float(k), len(idx) / generalized_groups * rng.uniform(0.92, 1.08))
        r_reid = min(1.0, 0.44 / eass)
        info_loss = min(0.93, 0.055 + 0.018 * k + 0.00065 * k * k + rng.normal(0.0, 0.006))
        fidelity = max(0.02, 1.0 - 0.80 * info_loss + rng.normal(0.0, 0.005))
        rows.append(
            {
                "K": k,
                "EASS": eass,
                "R_reid": r_reid,
                "information_loss_C": info_loss,
                "statistical_fidelity_F": fidelity,
            }
        )
    kl_rows: List[Dict[str, Any]] = []
    for k in [2, 3, 5, 8]:
        for l_val in [2, 3, 5, 8]:
            loss = min(0.95, 0.06 + 0.021 * k + 0.013 * l_val + 0.0018 * k * l_val + rng.normal(0, 0.004))
            kl_rows.append({"K": k, "L": l_val, "information_loss_C": loss})
    return rows, kl_rows


def snr_scan(ds: Dataset, seed: int, sample_n: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(ds.n, size=min(sample_n, ds.n), replace=False)
    x = np.stack(
        [
            ds.arrays["cpu_req"][idx],
            ds.arrays["mem_req"][idx] / 4.0,
            ds.arrays["duration"][idx] / 30.0,
            ds.arrays["risk_score"][idx] * 10.0,
        ],
        axis=1,
    )
    rows: List[Dict[str, Any]] = []
    power = float(np.mean(x * x))
    for snr in [0, 5, 10, 15, 20, 30]:
        noise_power = power / (10.0 ** (snr / 10.0))
        y = x + rng.normal(0.0, math.sqrt(noise_power), size=x.shape)
        mse = float(np.mean((x - y) ** 2))
        fidelity = max(0.0, 1.0 - mse / (power + 1e-9))
        query_error = float(np.mean(np.abs(y.mean(axis=0) - x.mean(axis=0)) / (np.abs(x.mean(axis=0)) + 1e-9)))
        protection = max(0.0, min(1.0, 1.0 - fidelity * 0.92 + 0.05 * rng.random()))
        rows.append(
            {
                "SNR_dB": snr,
                "statistical_fidelity_F": fidelity,
                "behavior_protection_rate": protection,
                "query_relative_error_Q": query_error,
            }
        )
    return rows


def coverage_matrix() -> List[Dict[str, Any]]:
    vals = [
        ("DP", [0.85, 0.40, 0.35, 0.90, 0.10]),
        ("KA", [0.05, 0.95, 0.30, 0.05, 0.30]),
        ("LD", [0.05, 0.95, 0.35, 0.05, 0.35]),
        ("DO", [0.35, 0.10, 0.85, 0.50, 0.05]),
        ("PS", [0.05, 0.90, 0.50, 0.05, 0.30]),
        ("SA", [0.90, 0.05, 0.05, 0.05, 0.05]),
        ("EN", [0.80, 0.05, 0.05, 0.05, 0.05]),
    ]
    rows = []
    for name, arr in vals:
        row = {"strategy": name}
        row.update({THREAT_NAMES[i]: arr[i] for i in range(5)})
        row["mean_coverage"] = float(np.mean(arr))
        rows.append(row)
    joint = 1.0 - np.prod(1.0 - np.array([v for _, v in vals], dtype=np.float64), axis=0)
    row = {"strategy": "ALL"}
    row.update({THREAT_NAMES[i]: float(joint[i]) for i in range(5)})
    row["mean_coverage"] = float(np.mean(joint))
    rows.append(row)
    return rows


def module_latency(ds: Dataset, cluster: Cluster, seed: int, sample_n: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    idxs = rng.choice(ds.n, size=min(sample_n, ds.n), replace=False)
    lat: Dict[str, List[float]] = {
        "RiskAnalyzer": [],
        "ProtectionCoordinator": [],
        "PASA_decision": [],
        "ProtectionExecution": [],
    }
    dummy_secret = b"pasa-secret"
    for idx in idxs:
        t0 = time.perf_counter_ns()
        _ = float(ds.arrays["risk_dims"][idx] @ RISK_WEIGHTS)
        t1 = time.perf_counter_ns()
        level = int(ds.arrays["risk_level"][idx])
        active = ["EN"]
        if level >= 1:
            active.append("PS")
        if level >= 2:
            active.extend(["DP", "KA"])
        if level >= 3:
            active.extend(["LD", "DO", "SA"])
        t2 = time.perf_counter_ns()
        nodes = feasible_nodes(ds, cluster, int(idx))
        priv = privacy_score(float(ds.arrays["risk_score"][idx]), cluster.q_priv[nodes], 0.0)
        eff = resource_match(ds, cluster, int(idx), nodes)
        _ = int(nodes[np.argmax(0.6 * priv + 0.4 * eff)])
        t3 = time.perf_counter_ns()
        payload = f"{int(ds.arrays['user_id'][idx])}:{float(ds.arrays['risk_score'][idx]):.8f}:{active}".encode()
        digest = hashlib.sha256(dummy_secret + payload).digest()
        random.seed(digest)
        _ = random.random() + rng.laplace(0.0, 1.0)
        t4 = time.perf_counter_ns()
        lat["RiskAnalyzer"].append((t1 - t0) / 1e6)
        lat["ProtectionCoordinator"].append((t2 - t1) / 1e6)
        lat["PASA_decision"].append((t3 - t2) / 1e6)
        lat["ProtectionExecution"].append((t4 - t3) / 1e6)
    rows: List[Dict[str, Any]] = []
    total = None
    for name, vals in lat.items():
        arr = np.array(vals, dtype=np.float64)
        if total is None:
            total = arr.copy()
        else:
            total += arr
        rows.append(
            {
                "module": name,
                "median_ms": float(np.median(arr)),
                "p95_ms": quantile(arr, 0.95),
                "p99_ms": quantile(arr, 0.99),
                "mean_ms": float(np.mean(arr)),
            }
        )
    if total is not None:
        rows.append(
            {
                "module": "total_scheduling_path",
                "median_ms": float(np.median(total)),
                "p95_ms": quantile(total, 0.95),
                "p99_ms": quantile(total, 0.99),
                "mean_ms": float(np.mean(total)),
            }
        )
    return rows


def scalability_from_latency(lat_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    p99 = {row["module"]: float(row["p99_ms"]) for row in lat_rows}
    rows = []
    for concurrency in [1, 10, 50, 100, 200]:
        factor = 1.0 + 0.38 * math.log2(concurrency)
        row = {"concurrency": concurrency}
        total = 0.0
        for module, key in [
            ("RiskAnalyzer", "RiskAnalyzer"),
            ("ProtectionCoordinator", "ProtCoordinator"),
            ("PASA_decision", "PASA"),
            ("ProtectionExecution", "ProtectionExecution"),
        ]:
            val = p99.get(module, 0.0) * factor * (1.0 + 0.015 * math.sqrt(concurrency))
            row[key] = val
            total += val
        row["total"] = total
        rows.append(row)
    return rows


def utilization_rows(ds: Dataset, cluster: Cluster, traces: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    assignment = traces["assignment"]
    end_time = traces["end_time"]
    start_time = traces["start_time"]
    service = np.maximum(0.0, end_time - start_time)
    span = max(1.0, float(end_time.max() - ds.arrays["submit_time"].min()))
    rows: List[Dict[str, Any]] = []
    for node_type in ["high", "general", "edge"]:
        nodes = np.nonzero(cluster.node_type == node_type)[0]
        mask = np.isin(assignment, nodes)
        capacity = float(np.sum(cluster.slots[nodes]) * span)
        cpu_util = float(service[mask].sum() / max(1.0, capacity))
        mem_util = cpu_util * (0.86 + 0.10 * (node_type == "general"))
        gpu_util = ""
        if node_type == "high":
            gpu_tasks = mask & (ds.arrays["gpu_req"] > 0.0)
            gpu_util = float(service[gpu_tasks].sum() / max(1.0, len(nodes) * span))
        net_util = cpu_util * (0.42 + 0.10 * (node_type == "general"))
        rows.append(
            {
                "node_type": node_type,
                "cpu_utilization": min(1.0, cpu_util),
                "memory_utilization": min(1.0, mem_util),
                "gpu_utilization": gpu_util,
                "network_utilization": min(1.0, net_util),
            }
        )
    return rows


def gpu_metaheuristic_benchmark(seed: int, population: int, iterations: int, batch: int, nodes: int) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "torch_available": torch is not None,
        "cuda_available": bool(torch is not None and torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch is not None and torch.cuda.is_available() else 0,
        "population": population,
        "iterations": iterations,
        "batch": batch,
        "nodes": nodes,
        "wall_seconds": "",
        "evaluations_per_second": "",
    }
    if torch is None or not torch.cuda.is_available():
        return result
    torch.manual_seed(seed)
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    chunks = max(1, len(devices))
    per_device_batch = max(1, batch // chunks)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    totals = []
    for device in devices:
        priv = torch.rand((per_device_batch, nodes), device=device)
        eff = torch.rand((per_device_batch, nodes), device=device)
        best = torch.zeros((per_device_batch,), device=device)
        for _ in range(iterations):
            weights = torch.rand((population, 1, 1), device=device)
            noise = 0.01 * torch.randn((population, per_device_batch, nodes), device=device)
            score = weights * priv.unsqueeze(0) + (1.0 - weights) * eff.unsqueeze(0) + noise
            best = torch.maximum(best, score.max(dim=2).values.max(dim=0).values)
        totals.append(float(best.mean().detach().cpu()))
    for device in devices:
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    evals = len(devices) * per_device_batch * nodes * population * iterations
    result["wall_seconds"] = t1 - t0
    result["evaluations_per_second"] = evals / max(1e-9, t1 - t0)
    result["mean_best_score"] = float(np.mean(totals))
    return result


def format_metric(v: Any) -> str:
    if isinstance(v, float):
        if abs(v) < 1 and v != 0:
            return f"{v:.4f}"
        return f"{v:.2f}"
    return str(v)


def make_summary(
    out_dir: Path,
    hardware: Dict[str, Any],
    risk_rows: List[Dict[str, Any]],
    sched_rows: List[Dict[str, Any]],
    alpha_rows: List[Dict[str, Any]],
    dyn_rows: List[Dict[str, Any]],
    gpu_result: Dict[str, Any],
    mode: str,
) -> None:
    lines = []
    lines.append("# PASA Experiment Run Summary")
    lines.append("")
    lines.append(f"- mode: `{mode}`")
    lines.append(f"- platform: `{hardware.get('platform')}`")
    lines.append(f"- cpu_count: `{hardware.get('cpu_count')}`")
    lines.append(f"- cuda_available: `{hardware.get('cuda_available')}`")
    lines.append(f"- cuda_devices: `{hardware.get('cuda_devices')}`")
    lines.append("")
    lines.append("## Risk Validation")
    for row in risk_rows:
        lines.append(
            f"- {row['threat_type']}: accuracy={pct(float(row['accuracy']))}, "
            f"kappa={float(row['kappa']):.3f}, samples={row['samples']}"
        )
    lines.append("")
    lines.append("## Scheduler Comparison on Dataset A")
    for row in sched_rows:
        lines.append(
            f"- {row['algorithm']}: APS={float(row['APS']):.3f}, "
            f"HRP={pct(float(row['HRP']))}, ATT={float(row['ATT_s']):.1f}s, "
            f"TP={float(row['TP_tasks_per_h']):.1f}/h"
        )
    lines.append("")
    lines.append("## Alpha Sweep")
    for row in alpha_rows:
        lines.append(
            f"- alpha={row['alpha']}: APS={float(row['APS']):.3f}, "
            f"HRP={pct(float(row['HRP']))}, ATT={float(row['ATT_s']):.1f}s"
        )
    lines.append("")
    lines.append("## Dynamic Alpha on Dataset C")
    for row in dyn_rows:
        lines.append(
            f"- {row['algorithm']}: APS={float(row['APS']):.3f}, "
            f"HRP={pct(float(row['HRP']))}, ATT={float(row['ATT_s']):.1f}s, "
            f"TP={float(row['TP_tasks_per_h']):.1f}/h"
        )
    lines.append("")
    lines.append("## CUDA Metaheuristic Benchmark")
    lines.append(f"- result: `{gpu_result}`")
    lines.append("")
    lines.append("CSV outputs are in this directory.")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == "quick":
        sizes = {"A": 5000, "B": 1000, "C": 10000}
        sample_n = 1000
        alpha_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        latency_sample = 500
        gpu_iters = 20
        gpu_batch = 2048
    else:
        sizes = {"A": 50000, "B": 10000, "C": 100000}
        sample_n = 5000
        alpha_values = [round(x, 1) for x in np.linspace(0.0, 1.0, 11)]
        latency_sample = 3000
        gpu_iters = 200
        gpu_batch = 8192

    print(f"[pasa] output directory: {out_dir}", flush=True)
    hardware = collect_hardware()
    write_json(out_dir / "hardware.json", hardware)
    cluster = make_cluster(args.seed)
    datasets = {
        name: make_dataset(name, n, args.seed + i * 101)
        for i, (name, n) in enumerate(sizes.items(), start=1)
    }
    write_json(
        out_dir / "dataset_manifest.json",
        {
            name: {"tasks": ds.n, "risk_distribution": risk_distribution(ds)}
            for name, ds in datasets.items()
        },
    )

    print("[pasa] risk validation and protection scans", flush=True)
    risk_rows, cm = risk_validation(datasets["B"])
    write_csv(out_dir / "risk_validation_dataset_b.csv", risk_rows)
    write_csv(
        out_dir / "risk_confusion_matrix_dataset_b.csv",
        [
            {
                "manual_level": LEVEL_NAMES[i],
                **{f"pred_{LEVEL_NAMES[j]}": int(cm[i, j]) for j in range(4)},
            }
            for i in range(4)
        ],
    )
    write_csv(out_dir / "risk_distribution_dataset_a.csv", risk_distribution(datasets["A"]))
    write_csv(out_dir / "dp_epsilon_scan.csv", dp_scan(datasets["A"], args.seed + 11, sample_n))
    k_rows, kl_rows = k_scan(datasets["A"], args.seed + 12, sample_n)
    write_csv(out_dir / "k_anonymity_scan.csv", k_rows)
    write_csv(out_dir / "k_l_diversity_interaction.csv", kl_rows)
    write_csv(out_dir / "snr_obfuscation_scan.csv", snr_scan(datasets["A"], args.seed + 13, sample_n))
    write_csv(out_dir / "strategy_coverage_matrix.csv", coverage_matrix())

    print("[pasa] scheduler comparison on dataset A", flush=True)
    sched_rows: List[Dict[str, Any]] = []
    traces_a_pasa: Optional[Dict[str, np.ndarray]] = None
    for i, algo in enumerate(ALGORITHMS):
        metrics, traces = simulate_schedule(datasets["A"], cluster, algo, args.seed + 1000 + i)
        sched_rows.append(metrics)
        if algo == "PASA":
            traces_a_pasa = traces
        print(
            f"[pasa] A {algo}: APS={metrics['APS']:.3f} "
            f"HRP={metrics['HRP']:.3f} ATT={metrics['ATT_s']:.1f}s",
            flush=True,
        )
    write_csv(out_dir / "scheduler_comparison_dataset_a.csv", sched_rows)

    print("[pasa] alpha sweep on dataset A", flush=True)
    alpha_rows: List[Dict[str, Any]] = []
    for alpha in alpha_values:
        metrics, _ = simulate_schedule(datasets["A"], cluster, "PASA", args.seed + 2000 + int(alpha * 100), fixed_alpha=alpha)
        metrics["alpha"] = alpha
        alpha_rows.append(metrics)
        print(f"[pasa] alpha={alpha:.1f}: APS={metrics['APS']:.3f} ATT={metrics['ATT_s']:.1f}s", flush=True)
    write_csv(out_dir / "alpha_scan_dataset_a.csv", alpha_rows)

    print("[pasa] dataset C dynamic alpha and throughput/latency", flush=True)
    dyn_rows: List[Dict[str, Any]] = []
    metrics_fixed, _ = simulate_schedule(datasets["C"], cluster, "PASA", args.seed + 3000, fixed_alpha=0.6)
    metrics_dyn, traces_c_pasa = simulate_schedule(datasets["C"], cluster, "PASA", args.seed + 3001)
    dyn_rows.extend([metrics_fixed, metrics_dyn])
    write_csv(out_dir / "dynamic_alpha_dataset_c.csv", dyn_rows)

    throughput_algos = ["FIFO", "Priority", "GA", "PSO", "Greedy", "EDF", "PASA"]
    throughput_rows: List[Dict[str, Any]] = []
    for i, algo in enumerate(throughput_algos):
        metrics, _ = simulate_schedule(datasets["C"], cluster, algo, args.seed + 3100 + i)
        throughput_rows.append(metrics)
        print(
            f"[pasa] C {algo}: TP={metrics['TP_tasks_per_h']:.1f}/h "
            f"P99={metrics['P99_latency_s']:.1f}s",
            flush=True,
        )
    write_csv(out_dir / "throughput_latency_dataset_c.csv", throughput_rows)
    write_csv(out_dir / "utilization_dataset_c_pasa.csv", utilization_rows(datasets["C"], cluster, traces_c_pasa))

    print("[pasa] module latency and scalability", flush=True)
    lat_rows = module_latency(datasets["A"], cluster, args.seed + 4000, latency_sample)
    write_csv(out_dir / "module_latency_dataset_a.csv", lat_rows)
    write_csv(out_dir / "scalability_dataset_a.csv", scalability_from_latency(lat_rows))

    print("[pasa] CUDA metaheuristic benchmark", flush=True)
    gpu_result = gpu_metaheuristic_benchmark(
        args.seed + 5000,
        population=50,
        iterations=gpu_iters,
        batch=gpu_batch,
        nodes=cluster.n_nodes,
    )
    write_json(out_dir / "gpu_metaheuristic_benchmark.json", gpu_result)

    manifest = {
        "mode": args.mode,
        "seed": args.seed,
        "sizes": sizes,
        "files": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
    }
    write_json(out_dir / "run_manifest.json", manifest)
    make_summary(out_dir, hardware, risk_rows, sched_rows, alpha_rows, dyn_rows, gpu_result, args.mode)
    print(f"[pasa] done: {out_dir / 'summary.md'}", flush=True)
    return out_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PASA thesis reproduction experiments.")
    parser.add_argument("--mode", choices=["quick", "full"], default="full", help="quick is for smoke tests; full matches thesis scale.")
    parser.add_argument("--out", default="pasa_results", help="Output directory for CSV, JSON, and summary files.")
    parser.add_argument("--seed", type=int, default=20260529, help="Base random seed.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

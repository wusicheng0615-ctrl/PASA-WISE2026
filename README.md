# PASA-WISE2026
Python/NumPy simulator and experimental scripts for PASA, WISE 2026.
# PASA — WISE 2026

Experimental code accompanying:

**PASA: Risk-Aware Privacy-Preserving Task Scheduling for Web-Scale Cloud-Edge Information Systems**

## Contents

- `pasa_full_experiments.py`: synthetic workload generation, scheduling simulation, parameter sweeps, and auxiliary diagnostics.
- `monte_carlo_experiments.py`: paired Monte Carlo experiments, including the windowed Kuhn–Munkres reference.

Keep both scripts in the same directory.

## Installation

The scheduling simulator uses Python and NumPy on the CPU.

```bash
python -m pip install numpy
```

PyTorch is optional. If installed with CUDA support, the main runner may additionally execute a separate GPU scoring benchmark.

## Quick test

```bash
python pasa_full_experiments.py --mode quick --out results_quick
```

Quick mode uses smaller workloads and does not reproduce the paper-scale numerical results.

## Full experiment runner

```bash
python pasa_full_experiments.py --mode full --seed 20260529 --out results_full
```

## Paper's Monte Carlo configuration

```bash
python monte_carlo_experiments.py --repeats 1000 --tasks 600 --window-size 20 --seed 20260529 --out results_monte_carlo
```

## Interpretation and scope

All workloads and risk-axis scores are synthetic. The privacy-facing metrics measure consistency with the scheduling objective; they do not measure the success rate of an independent privacy attack.

The archived GA and PSO labels identify noisy weighted scoring baselines, not population-based genetic or particle-swarm optimizers.

The legacy result field `R_reid` corresponds to the scheduling linkability proxy described in the WISE paper.

The main runner retains auxiliary thesis-era scans and a separate optional GPU benchmark. These should not be interpreted as evidence that the WISE scheduling experiments execute privacy-protection mechanisms.

Execution-time measurements depend on the machine and software environment.

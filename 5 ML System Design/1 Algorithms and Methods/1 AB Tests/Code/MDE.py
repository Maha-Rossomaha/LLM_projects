"""
This script estimates sample size, power, and duration for A/B experiments.
It supports both binary metrics (conversion rate) and continuous metrics (means).

Outputs:
    - required sample size per group (nA, nB)
    - estimated experiment duration (days)
    - sensitivity table for multiple MDEs
"""

import argparse
import math
import numpy as np
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize


def n_per_group_binary(
    p_baseline: float, 
    mde_rel: float, 
    alpha: float, 
    power: float, 
    ratio: float = 1.0
):
    p_alt = p_baseline * (1.0 + mde_rel)
    effect_size = proportion_effectsize(p_baseline, p_alt)
    analysis = NormalIndPower()
    nA = analysis.solve_power(
        effect_size=effect_size, 
        alpha=alpha,
        power=power,
        ratio=ratio
    )
    return math.ceil(nA), math.ceil(nA * ratio)


def n_per_group_continuous(
    sigma: float, 
    mde_abs: float, 
    alpha: float, 
    power: float, 
    ratio: float = 1.0
):
    d = mde_abs / sigma
    analysis = TTestIndPower()
    nA = analysis.solve_power(
        effect_size=d, 
        alpha=alpha, 
        power=power, 
        ratio=ratio
    )
    return math.ceil(nA), math.ceil(nA * ratio)


def estimate_days(
    nA: int, 
    nB: int, 
    traffic: int, 
    wA: float = 0.5, 
    wB: float = 0.5
):
    days_A = nA / (traffic * wA)
    days_B = nB / (traffic * wB)
    return math.ceil(max(days_A, days_B))


def sensitivity_table_binary(
    p_baseline: float, 
    alpha: float, 
    power: float, 
    mde_rel_values
):
    print("\nSensitivity table (binary metric):")
    print(f"Baseline = {p_baseline:.3f}, alpha={alpha}, power={power}\n")
    print(f"{'MDE, %':>8} | {'n per group':>12}")
    print("-" * 25)
    for mde_rel in mde_rel_values:
        nA, _ = n_per_group_binary(p_baseline, mde_rel, alpha, power)
        print(f"{mde_rel*100:8.2f} | {nA:12}")


def main():
    parser = argparse.ArgumentParser(description="Power Analysis & MDE Estimator")
    parser.add_argument("--metric", choices=["binary", "continuous"], required=True)
    parser.add_argument("--baseline", type=float, required=True, help="Baseline CR (binary) or mean (continuous)")
    parser.add_argument("--mde_rel", type=float, default=None, help="Relative MDE (for binary)")
    parser.add_argument("--mde_abs", type=float, default=None, help="Absolute MDE (for continuous)")
    parser.add_argument("--sigma", type=float, default=None, help="Std deviation for continuous metric")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.8)
    parser.add_argument("--traffic", type=int, default=100_000)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--weights", nargs=2, type=float, default=[0.5, 0.5], help="Traffic weights for A/B")
    args = parser.parse_args()

    if args.metric == "binary":
        if args.mde_rel is None:
            raise ValueError("--mde_rel required for binary metric")
        nA, nB = n_per_group_binary(
            args.baseline, 
            args.mde_rel, 
            args.alpha, 
            args.power, 
            args.ratio
        )
    else:
        if args.mde_abs is None or args.sigma is None:
            raise ValueError("--mde_abs and --sigma required for continuous metric")
        nA, nB = n_per_group_continuous(
            args.sigma,
            args.mde_abs,
            args.alpha,
            args.power,
            args.ratio
        )

    days = estimate_days(nA, nB, args.traffic, *args.weights)

    print("\n=== Power Analysis Result ===")
    print(f"Metric type: {args.metric}")
    print(f"Alpha = {args.alpha}, Power = {args.power}")
    print(f"Required sample size: nA = {nA}, nB = {nB}")
    print(f"Estimated duration: ~{days} days\n")

    if args.metric == "binary":
        mde_range = np.linspace(0.01, 0.10, 10)
        sensitivity_table_binary(
            args.baseline,
            args.alpha,
            args.power,
            mde_range
        )


if __name__ == "__main__":
    main()
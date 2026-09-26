"""Aggregate the Stage 5A matched 24-block replications without retraining."""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def final_ppl(path: Path) -> float:
    milestone_path = path.with_name(f"{path.stem}_milestones.json")
    if milestone_path.exists():
        milestones = json.loads(milestone_path.read_text(encoding="utf-8")).get("milestones", [])
        completed = [m for m in milestones if abs(float(m.get("tokens_M", -1)) - 50.0) < 0.1]
        if completed:
            return float(completed[-1]["val_ppl"])
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Empty log: {path}")
    return float(frame.iloc[-1]["val_perplexity"])


def bootstrap_mean_ci(values: np.ndarray, draws=20_000, seed=20260913):
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default="logs_stage5")
    parser.add_argument("--stage4-logs-dir", default="logs_stage4")
    parser.add_argument("--output-dir", default="results_stage5")
    args = parser.parse_args()

    logs = Path(args.logs_dir)
    stage4 = Path(args.stage4_logs_dir)
    rows = []
    for seed, directory in ((42, stage4), (123, logs), (2026, logs)):
        seq = directory / f"seq24_seed{seed}.csv"
        par = directory / f"par24_2_seed{seed}.csv"
        if not (seq.exists() and par.exists()):
            print(f"Skipping seed {seed}: paired logs not both complete.")
            continue
        seq_ppl, par_ppl = final_ppl(seq), final_ppl(par)
        rows.append({
            "seed": seed,
            "seq24_ppl": seq_ppl,
            "par24_2_ppl": par_ppl,
            "absolute_gap": par_ppl - seq_ppl,
            "relative_penalty_pct": 100 * (par_ppl - seq_ppl) / seq_ppl,
        })
    if not rows:
        raise RuntimeError("No completed matched seed pairs found.")

    table = pd.DataFrame(rows).sort_values("seed")
    penalty = table["relative_penalty_pct"].to_numpy()
    gap = table["absolute_gap"].to_numpy()
    summary = {
        "n_paired_seeds": int(len(table)),
        "mean_seq24_ppl": float(table["seq24_ppl"].mean()),
        "mean_par24_2_ppl": float(table["par24_2_ppl"].mean()),
        "mean_absolute_gap": float(gap.mean()),
        "sd_absolute_gap": float(gap.std(ddof=1)) if len(gap) > 1 else None,
        "mean_relative_penalty_pct": float(penalty.mean()),
        "sd_relative_penalty_pct": float(penalty.std(ddof=1)) if len(penalty) > 1 else None,
        "bootstrap95_mean_penalty_pct": bootstrap_mean_ci(penalty) if len(penalty) > 1 else None,
        "decision_rule": "strong if mean <= 2% and upper bootstrap bound <= 3%",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    summary["decision"] = (
        "PENDING_MORE_SEEDS" if len(table) < 3 else
        "STRONG_GO" if summary["mean_relative_penalty_pct"] <= 2 and summary["bootstrap95_mean_penalty_pct"][1] <= 3
        else "MIXED_OR_NO_GO"
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "stage5a_paired_results.csv", index=False)
    (output / "stage5a_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

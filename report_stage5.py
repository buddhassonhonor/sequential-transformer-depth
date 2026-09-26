"""Build a concise evidence-only report from completed Stage 5 artifacts."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results_stage5")
    args = parser.parse_args()
    root = Path(args.results_dir)
    summary = json.loads((root / "stage5a_summary.json").read_text(encoding="utf-8"))
    paired = pd.read_csv(root / "stage5a_paired_results.csv")
    ablations = []
    for path in sorted(root.glob("par24_2_seed*_branch_ablation.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out = data["outcomes"]
        ablations.append({
            "seed": path.stem.split("_seed")[1].split("_")[0],
            "full": out["full"]["ppl"],
            "branch1": out["branch1"]["ppl"],
            "branch2": out["branch2"]["ppl"],
            "random": out["random_single"]["ppl"],
        })
    ablation_table = pd.DataFrame(ablations)
    if not ablation_table.empty:
        ablation_table.to_csv(root / "stage5b_branch_ablation.csv", index=False)

    lines = [
        "# Stage 5 Results: 24-Block Multi-Seed Replication and Branch Utilization",
        "",
        "## Protocol",
        "",
        "- Matched models: SEQ24 (depth 24) vs PAR24_2 (12 stages x 2 branches; critical depth 12).",
        "- Both models: 61.82M non-embedding parameters; TinyStories GPT-2 BPE; 50M tokens; BF16; batch 32; context 256.",
        "- Seeds: 42, 123, 2026. The terminal validation PPL is the deterministic 40-batch estimate used throughout training.",
        "",
        "## Stage 5A — Matched multi-seed result",
        "",
        paired.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"Mean SEQ24 PPL: **{summary['mean_seq24_ppl']:.4f}**; mean PAR24_2 PPL: **{summary['mean_par24_2_ppl']:.4f}**.",
        f"Mean PAR24_2 penalty: **{summary['mean_relative_penalty_pct']:+.2f}%** "
        f"(absolute PPL gap {summary['mean_absolute_gap']:+.4f}).",
    ]
    if summary["bootstrap95_mean_penalty_pct"] is not None:
        lo, hi = summary["bootstrap95_mean_penalty_pct"]
        lines.append(f"Paired bootstrap 95% CI for the mean penalty: **[{lo:+.2f}%, {hi:+.2f}%]**.")
    lines += [
        f"Decision under the preregistered rule: **{summary['decision']}**.",
        "",
        "## Stage 5B — Branch utilization ablation",
        "",
    ]
    if ablation_table.empty:
        lines.append("Ablations were not available when this report was generated.")
    else:
        lines += [
            ablation_table.to_markdown(index=False, floatfmt=".3f"),
            "",
            "Each ablation retains only one branch at every stage; random-single chooses one branch per stage. "
            "Large degradation relative to the full model is evidence that both learned branches are required jointly.",
        ]
    lines += [
        "",
        "## Interpretation and next gate",
        "",
        "The result supports a quality-preserving 2x compression of nominal sequential depth on this workload only if the multi-seed confidence interval remains inside the stated threshold. "
        "It does not establish single-GPU latency speedup, cross-corpus generalization, or a near-lossless 3x compression; those remain the Stage 6/7 gated tests.",
        "",
        "## Reproducibility",
        "",
        f"Environment: Python {summary['environment']['python']}, PyTorch {summary['environment']['torch']}, CUDA {summary['environment']['cuda']}, GPU {summary['environment']['gpu']}.",
        "Raw per-run logs, milestone JSON, checkpoints and all derived CSV/JSON files are retained alongside this report.",
    ]
    (root / "stage5_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(root / "stage5_final_report.md")


if __name__ == "__main__":
    main()

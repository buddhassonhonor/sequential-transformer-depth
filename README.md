# Sequential Transformer Depth

Code and source data for **How Much Transformer Depth Needs to Be Sequential?**

This repository accompanies experiments that separate the number of Transformer
blocks from the number of block transformations that must be evaluated
sequentially. The principal experiments use 24 decoder blocks arranged as
`S` sequential stages with `K` independent blocks per stage, where `N = S × K`.

## Repository contents

- `model.py`: serial and stage-parallel decoder implementations.
- `prepare_wikitext103.py`: WikiText-103 download, GPT-2 BPE tokenization, and
  deterministic validation-index generation.
- `train_stage4.py`: TinyStories twofold-compression experiments.
- `train_stage6.py` and `train_stage7.py`: 50-million-token WikiText-103 runs,
  including the complete 24-block sequential-depth frontier.
- `train_stage8.py`: fresh 100-million-token longitudinal runs.
- `analysis_*.py`, `analyze_stage5.py`, and `branch_*.py`: aggregation,
  evaluation, and representation analyses.
- `results_stage5/` through `results_stage8/`: per-run numerical outputs,
  summaries, deterministic validation indices, and analysis figures used to
  support the reported results.
- `paper_nmi/make_figures.py`: script that regenerates the four main figures
  from the archived result files.

Model checkpoints are intentionally excluded because they exceed 60 GB. Raw
WikiText-103 and TinyStories text is also excluded; both datasets are obtained
from their original public sources and processed locally.

## Environment

Python 3.10 or newer is recommended. Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

CUDA-capable hardware is required for the full training runs. The reported
experiments used PyTorch with bfloat16 automatic mixed precision.

## Prepare WikiText-103

```bash
python prepare_wikitext103.py --output-dir data/wikitext103_gpt2
```

The script downloads `Salesforce/wikitext` (`wikitext-103-raw-v1`) through
Hugging Face Datasets, tokenizes it with GPT-2 BPE, and writes `train.bin`,
`val.bin`, `test.bin`, `meta.json`, and deterministic validation indices.

## Representative training commands

Serial and moderately stage-parallel baselines:

```bash
python train_stage6.py --model seq24 --seed 42 --data_dir data/wikitext103_gpt2
python train_stage6.py --model par24_3 --seed 42 --data_dir data/wikitext103_gpt2
```

Additional points on the 24-block frontier:

```bash
python train_stage7.py --model sp24_s4k6 --seed 42 --data_dir data/wikitext103_gpt2
python train_stage7.py --model sp24_s1k24 --seed 42 --data_dir data/wikitext103_gpt2
```

Fresh 100-million-token schedule:

```bash
python train_stage8.py --model seq24 --seed 42 --data_dir data/wikitext103_gpt2
python train_stage8.py --model par24_3 --seed 42 --data_dir data/wikitext103_gpt2
```

Seeds used in the paper are 42, 123, and 2026. See each script's `--help`
output for the complete configuration interface.

## Reproduce summaries and figures

The archived results are sufficient to regenerate the aggregate summaries and
main figures without model checkpoints:

```bash
python analysis_stage7b.py
python analysis_stage8b.py
python paper_nmi/make_figures.py
```

Some checkpoint-dependent analyses, such as fresh test evaluation and branch
ablation, require rerunning training first.

## Data provenance

The repository contains experimental measurements produced by the accompanying
code, not copies of the source corpora. WikiText-103 is loaded from
`Salesforce/wikitext`; TinyStories is described by Eldan and Li (2023). Users
must comply with the terms of the original datasets.

## Archival release

Versioned GitHub releases are synchronized to Figshare. Cite the version DOI
shown on the corresponding Figshare record once the release has been archived.


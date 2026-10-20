# GSTRNet 100x100 scalability experiment

This package evaluates a shared-window deployment of GSTRNet on the complete
Milan grid.  It requires the existing `trainduobu1_gate_ablation` directory in
the project root.

The 100x100 grid is divided into 25 non-overlapping 20x20 windows.  One GSTRNet
is shared by every window, so the parameter count stays constant and the
spatial-attention cost grows approximately linearly with the number of windows.

## Files

- `benchmark_scalability.py`: forward/backward memory and timing benchmark on
  20, 40, 60, 80 and 100 grids.
- `train_windowed_gstrnet.py`: train and evaluate the full 100x100 Milan grid.
- `run_windowed_multigpu.py`: schedule dataset/seed jobs on four GPUs.
- `summarize_scalability.py`: export all-run and mean/std CSVs.

## 1. Upload and environment

Upload this entire directory to the project root, alongside:

```text
trainduobu1_gate_ablation/
gstrnet_scalability/
dataset/data_git_version.h5
```

Then:

```bash
cd /path/to/GTSRNet
conda activate glsttn
```

## 2. Smoke test

```bash
python -u ./gstrnet_scalability/train_windowed_gstrnet.py \
  --project-root "$PWD" \
  --output-dir "$PWD/results_scalability_100" \
  --smoke-test \
  --gpu 0
```

The last line should start with `PASS`.

## 3. Efficiency benchmark (run first)

This does not train on the dataset.  It normally finishes much sooner than the
accuracy experiment.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u ./gstrnet_scalability/benchmark_scalability.py \
  --project-root "$PWD" \
  --output-dir "$PWD/results_scalability_100/benchmark" \
  --grid-sizes 20 40 60 80 100 \
  --window-size 20 \
  --tile-batch-size 1 \
  --batch-size 1 \
  --global-max-size 40 \
  --warmup 1 \
  --repeats 3 \
  --gpu 0 \
  --amp
```

The original quadratic global model is tested through 40x40 by default.  Larger
global grids are marked as skipped to avoid an unnecessary OOM.  Set
`--global-max-size 60` only if the GPU has enough memory.

Output:

```text
results_scalability_100/benchmark/scalability_benchmark.csv
```

## 4. One-epoch pilot on Milan-Internet

Run this before the full experiment to estimate epoch duration and memory:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u ./gstrnet_scalability/train_windowed_gstrnet.py \
  --project-root "$PWD" \
  --output-dir "$PWD/results_scalability_100_pilot" \
  --dataset milan_internet \
  --grid-size 100 \
  --window-size 20 \
  --stride 20 \
  --batch-size 16 \
  --epochs 1 \
  --patience 0 \
  --learning-rate 0.001 \
  --loss l1 \
  --branch-aux-weight 0.1 \
  --seed 23 \
  --d-model 128 \
  --e-layers 2 \
  --seg-len 6 \
  --gpu 0 \
  --amp
```

## 5. Recommended three-seed experiment

Start with Milan-Internet only:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u ./gstrnet_scalability/run_windowed_multigpu.py \
  --project-root "$PWD" \
  --output-dir "$PWD/results_scalability_100" \
  --datasets milan_internet \
  --seeds 23 29 103 \
  --gpus 0 1 2 3 \
  --grid-size 100 \
  --window-size 20 \
  --batch-size 16 \
  --epochs 100 \
  --patience 10 \
  --learning-rate 0.001 \
  --branch-aux-weight 0.1 \
  --d-model 128 \
  --e-layers 2 \
  --seg-len 6 \
  --amp
```

If the pilot and Internet runs are normal, run all three Milan traffic types:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u ./gstrnet_scalability/run_windowed_multigpu.py \
  --project-root "$PWD" \
  --output-dir "$PWD/results_scalability_100" \
  --datasets milan_sms milan_call milan_internet \
  --seeds 23 29 103 \
  --gpus 0 1 2 3 \
  --grid-size 100 \
  --window-size 20 \
  --batch-size 16 \
  --epochs 100 \
  --patience 10 \
  --learning-rate 0.001 \
  --branch-aux-weight 0.1 \
  --d-model 128 \
  --e-layers 2 \
  --seg-len 6 \
  --amp
```

Completed dataset/seed jobs are skipped automatically.  Do not add `--force`
when extending the Internet-only run to all three datasets.

If CUDA runs out of memory, change `--batch-size 16` to 8 or 4.  The window
size and architecture remain unchanged.

## 6. Summarize completed runs

```bash
python -u ./gstrnet_scalability/summarize_scalability.py \
  --results-dir "$PWD/results_scalability_100"
```

Outputs:

```text
results_scalability_100/scalability_accuracy_all_runs.csv
results_scalability_100/scalability_accuracy_mean_std.csv
```

## Interpretation boundary

The result supports scalability of **windowed/shared GSTRNet**, not unchanged
global attention over 10,000 cells.  Non-overlapping windows do not directly
model cross-window interactions; state this limitation in the paper.

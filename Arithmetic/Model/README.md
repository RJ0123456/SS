# Arithmetic GPT

A character-level, decoder-only Transformer for learning from the generated arithmetic problem text. The model predicts the next character, so it learns both the arithmetic notation and the column-style working in the training files.

## Requirements

- Python 3.10 or newer
- PyTorch

Install the project dependency with:

```powershell
python -m pip install -r Arithmetic\Model\requirements.txt
```

For a CUDA-specific PyTorch build, use the installation selector at [pytorch.org](https://pytorch.org/get-started/locally/) instead.

## Generate Training Data

From the repository root, create a dataset if one is not already present:

```powershell
python Arithmetic\Data\generate_data.py 25000 --output-dir Arithmetic\Data\generated --seed 42
```

The training script reads `.txt` files recursively from the data directory.

## Train

Run from the repository root:

```powershell
python Arithmetic\Model\train.py --data-dir Arithmetic\Data\generated
```

By default, the trainer runs 10,000 steps, evaluates every 500 steps, and saves a checkpoint to `Arithmetic/Model/checkpoints/gpt.pt`. It automatically uses CUDA when available and otherwise uses CPU.

For a short CPU smoke run:

```powershell
python Arithmetic\Model\train.py `
  --data-dir Arithmetic\Data\generated `
  --output Arithmetic\Model\checkpoints\smoke.pt `
  --steps 2 --eval-interval 1 --eval-iters 1 `
  --batch-size 2 --block-size 32 `
  --d-model 32 --n-heads 4 --n-layers 2 --d-ff 64 --device cpu
```

## Model Defaults

| Setting | Default | CLI option |
| --- | ---: | --- |
| Model width | 256 | `--d-model` |
| Attention heads | 8 | `--n-heads` |
| Transformer layers | 6 | `--n-layers` |
| Feed-forward width | 1024 | `--d-ff` |
| Dropout | 0.0 | `--dropout` |
| Context length | 256 characters | `--block-size` |

The model settings can be combined with these training options:

| Option | Default | Purpose |
| --- | ---: | --- |
| `--steps` | 10000 | Number of optimization steps |
| `--batch-size` | 64 | Sequences per training batch |
| `--learning-rate` | 0.0003 | AdamW learning rate |
| `--weight-decay` | 0.1 | AdamW weight decay |
| `--eval-interval` | 500 | Steps between loss evaluations |
| `--eval-iters` | 50 | Batches per train/validation evaluation |
| `--validation-fraction` | 0.1 | Fraction of text held out for validation |
| `--seed` | 42 | Random seed |
| `--device` | CUDA if available, otherwise CPU | PyTorch device |
| `--output` | `Arithmetic/Model/checkpoints/gpt.pt` | Checkpoint path |

The checkpoint contains the model and optimizer states, model configuration, vocabulary, and training step count.

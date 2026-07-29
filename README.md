# Beach Classification Software

This repository contains the data preparation, training, ensemble evaluation,
inference, and reporting workflow for context-aware classification of beach
cells. Each cell is represented by five 20 x 20 matrices: red, green, blue,
slope, and elevation.

The workflow has two classification stages:

1. Beach-surface classification using the labels in
   `Data/Raw/Descriptor_fs.csv`.
2. Grain-size classification using classes 11-13 from
   `Data/Raw/Descriptor_ss.csv`. Full-dataset inference in this stage is
   restricted to cells assigned to beach-surface class 1 or 2.

## Models

The experiments include five backbone models:

- Small CNN, trained from random initialization without spatial resizing.
- EfficientNetV2-S, adapted to five input channels.
- ViT-Tiny (`vit_tiny_patch16_224`), adapted to five input channels.
- Swin-Tiny, adapted to five input channels.
- ConvNeXt-Tiny, adapted to five input channels.

The four pretrained backbones are also compared using hard voting, probability
voting, globally weighted probability voting, and model-and-class weighted
probability voting. The Small CNN is retained as a standalone baseline.

## Project Layout

```text
.
|-- README.md
|-- Data/
|   |-- Raw/
|   `-- Preprocessed/
|-- Notebooks/
|-- Results/
|-- Source/
|-- requirements.txt
|-- requirements-remote-gpu-cu128.txt
|-- run_remote_notebooks.sh
`-- ss_run_remote_notebooks.sh
```

`Source/` provides the shared dataset, model, training, grid-search, ensemble,
inference, and reporting code. `Notebooks/` contains the executable workflows.
Generated outputs are written under `Results/`.

## Data Layout

The complete cell dataset is expected at:

```text
Data/Raw/beach_all/<date>/
```

For a cell identifier such as `1000956`, a date directory contains:

```text
1000956r.txt  # red
1000956g.txt  # green
1000956b.txt  # blue
1000956s.txt  # slope
1000956d.txt  # elevation
```

Each file stores a 20 x 20 matrix. Neighboring cells are loaded from the same
survey date according to the selected context radius. A radius of 0 uses only
the central cell, radius 1 produces a 3 x 3 cell context, and radius 2 produces
a 5 x 5 context.

NaN values and the `-999` sentinel are treated as missing. Channel means and
standard deviations are calculated from the training split. Missing values,
including unavailable neighboring cells, are filled with the corresponding
training-channel means. Validation and test data do not contribute to these
statistics.

The split notebooks write first-stage and second-stage CSV files to
`Data/Preprocessed/`. They contain dates, cell identifiers, labels, and split
assignments; they do not copy the matrix files.

## Installation

Create and activate a virtual environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the remote CUDA 12.8 environment used by the project:

```bash
python -m pip install -r requirements-remote-gpu-cu128.txt
```

The remote requirements pin PyTorch 2.8.0, torchvision 0.23.0, and torchaudio
2.8.0. GPU availability can be checked with:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Start JupyterLab from the repository root:

```bash
jupyter lab
```

## Workflow

Run the notebooks in this order for the first-stage dataset:

1. `create_dataset_splits.ipynb`
2. `train_small_cnn_classifier.ipynb`
3. `train_efficientnetv2_classifier.ipynb`
4. `train_vit_tiny_classifier.ipynb`
5. `train_swin_tiny_classifier.ipynb`
6. `train_convnext_classifier.ipynb`
7. `evaluate_ensemble.ipynb`
8. `example_metrics_reporting.ipynb`

The corresponding second-stage notebooks use the `ss_` prefix, beginning with
`ss_create_dataset_splits.ipynb`.

Each training notebook performs grid search using the validation split, saves
the best checkpoint and inference metadata, and evaluates the selected model on
the test split. Ensemble selection also uses validation results; test results
are reserved for final comparison.

Feature-distribution notebooks report channel statistics by class:

- `plot_feature_distributions_by_class.ipynb`
- `ss_plot_feature_distributions_by_class.ipynb`

Full-dataset classification is performed by:

- `classify_all_cells_convnext.ipynb`
- `classify_all_cells_weighted_ensemble.ipynb`
- `ss_classify_beach_surfaces_weighted_ensemble.ipynb`

The inference notebooks read matrices directly from `Data/Raw/beach_all`; they
do not copy the source dataset.

## Remote Batch Execution

The launch scripts execute notebooks sequentially in a detached `nohup`
pipeline. From the repository root, run:

```bash
./run_remote_notebooks.sh all
./ss_run_remote_notebooks.sh all
```

Individual jobs can also be selected:

```bash
./run_remote_notebooks.sh convnext vit ensemble
./ss_run_remote_notebooks.sh training
```

Set `BEACH_PYTHON` when a specific interpreter is required:

```bash
BEACH_PYTHON=/path/to/python ./run_remote_notebooks.sh all
```

Each job writes a debug log to the repository root. Executed notebooks are
written to `Notebooks/executed_notebooks/`; this generated directory is ignored
by Git.

## Results

Model-specific directories under `Results/` contain the selected configuration,
grid-search summaries and traces, normalization statistics, validation and test
metrics, predictions, inference metadata, and the best loadable checkpoint.
Ensemble directories contain validation-derived weights, predictions,
classification reports, and confusion matrices.

The metrics-reporting notebooks calculate accuracy, macro precision, macro
recall, macro F1-score, Matthews correlation coefficient, exact confusion
matrices, and one-vs-rest precision-recall curves. They also print LaTeX tables
for aggregate metrics, per-class metrics, best backbones, and selected
grid-search configurations.

Model checkpoints (`*.pt`) are tracked through Git LFS. After cloning the
repository, install Git LFS and materialize the checkpoints with:

```bash
git lfs install
git lfs pull
```

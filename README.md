# Context-Aware Deep Learning Ensembles for Beach-Surface and Grain-Size Classification in Isbjørnhamna, Svalbard

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
|-- bs_run_remote_notebooks.sh
`-- gs_run_remote_notebooks.sh
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
the central cell, radius 1 produces a 3 x 3 cell context, radius 2 produces
a 5 x 5 context, and radius 3 produces a 7 x 7 context. All training grids test
`[0, 1, 2, 3]`. The native mosaic sides are 20, 60, 100, and 140 pixels;
the four pretrained models resize them to 224 x 224. Small CNN uses native sizes.

NaN values and the `-999` sentinel are treated as missing. Channel means and
standard deviations are calculated from the training split. Missing values,
including unavailable neighboring cells, are filled with the corresponding
training-channel means. Validation and test data do not contribute to these
statistics.

The beach-surface split notebook writes `bs_dataset_splits.csv`,
`bs_train_split.csv`, `bs_validation_split.csv`, `bs_test_split.csv`, and
`bs_split_provenance.json` to `Data/Preprocessed/`. All labelled cells from
surveys **20180812, 20180915, and 20220910** belong exclusively to the test set.
All labelled cells from **20230731 and 20240623** belong exclusively to the
validation set. Every other survey is training-only. No cells are randomly
assigned between these sets: training, validation, and testing are disjoint at
the survey level. All three sets must contain all ten classes. With the current
annotations this gives 5,386 training, 1,243 validation, and 1,493 test cells.
Because context is loaded from the central cell's survey, no same-survey input
observation can appear across these split boundaries. The same geographic
location can still be observed in different surveys; this is not a geographic
holdout across all dates.

Grain-size evaluation uses **five outer folds and three inner folds**, saved as
`gs_cv_samples.csv`, `gs_cv_inner_folds.csv`, and `gs_cv_manifest.json`.
Each of the 140 cells is in exactly one outer evaluation fold. Same-survey
cells with intersecting maximum-radius contexts are assigned to the same
connected spatial group using DBSCAN (Chebyshev distance, `min_samples=1`).
For maximum radius 3, centres separated by at most 6 cells are linked;
transitively linked cells remain in the same group, even if their labels differ.
StratifiedGroupKFold assigns intact groups to outer and inner folds while
requiring all three classes in every training and evaluation partition.

There are 129 groups in the current annotations; outer evaluation fold sizes
are 30, 27, 27, 29, and 27 cells. Inner folds are formed only from the corresponding
outer-training groups. Both levels apply a training buffer excluding centres
at distance <= 6 from a same-survey held-out centre and verify zero context
intersection. The connected-group construction already separates all current
groups by more than 6 cells, so the buffer checks do not discard any additional
current annotations. This common maximum-radius separation is used for all
tested radii `[0, 1, 2, 3]`; a larger grid radius requires regenerating the plan.

Configuration selection uses mean inner-validation macro F1. Inner validation loss controls early stopping;
the median best epoch of the selected configuration determines the fixed refit
length on the corresponding outer-training cells. Outer labels never control selection,
normalization, early stopping, or ensemble weights. The same folds serve every
model. This is **spatial-group CV with same-survey context isolation**, not
survey-held-out validation or geographic isolation across different dates.

For each ensemble outer fold, weights are calculated from the selected
backbones' inner out-of-fold probabilities on that fold's training cells.
Hard voting resolves ties using mean probabilities among tied classes.
The outer-held-out predictions evaluate all four ensemble rules independently.
Separate three-fold inner CV on all labelled cells selects deployment models
and deployment ensemble weights; these are not evaluated on their training data.
The old `gs_*_split.csv` files are retained only as historical metadata and are
not consumed by current training, evaluation, or grain-size violin notebooks.
No split/fold notebook copies matrix files.

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

Run the notebooks in this order for the beach-surface dataset:

1. `bs_create_dataset_splits.ipynb`
2. `bs_train_small_cnn_classifier.ipynb`
3. `bs_train_efficientnetv2_classifier.ipynb`
4. `bs_train_vit_tiny_classifier.ipynb`
5. `bs_train_swin_tiny_classifier.ipynb`
6. `bs_train_convnext_classifier.ipynb`
7. `bs_evaluate_ensemble.ipynb`
8. `bs_example_metrics_reporting.ipynb`

Beach-surface notebooks use the `bs_` prefix. The corresponding grain-size
notebooks use the `gs_` prefix, beginning with
`gs_create_dataset_splits.ipynb`.

Beach-surface training performs grid search with two validation surveys
and evaluates once on the three-survey test holdout. Grain-size training performs
nested CV and separately fits a final deployment model. Both save checkpoints
and inference metadata. Statistics are fitted independently on each fit's
training cells, using central-cell pixels only, and shared across context radii
within that fit. Missing neighbors always come from the same survey date and
use those training-channel means.

Rerun split generation and training after changing the evaluation protocol.
Saved split fingerprints and CV completion manifests reject stale results.
Historical notebook outputs and existing trained results are preserved; they
do not represent the new experiments until the notebooks are executed again.

Feature-distribution notebooks report channel statistics by class:

- `bs_plot_feature_distributions_by_class.ipynb`
- `gs_plot_feature_distributions_by_class.ipynb`

Full-dataset classification is performed by:

- `bs_classify_all_cells_convnext.ipynb`
- `bs_classify_all_cells_weighted_ensemble.ipynb`
- `gs_classify_beach_surfaces_weighted_ensemble.ipynb`

The inference notebooks read matrices directly from `Data/Raw/beach_all`; they
do not copy the source dataset.

## Remote Batch Execution

The launch scripts execute notebooks sequentially in a detached `nohup`
pipeline. From the repository root, run:

```bash
./bs_run_remote_notebooks.sh all
./gs_run_remote_notebooks.sh all
```

The `all` and `training` modes first execute the corresponding split/fold
notebook. Individual model jobs reuse the saved metadata. Grain-size nested CV
requires many more fits than a single split; grid configurations are distributed
over the dynamically detected CUDA devices. Each notebook can also be executed
directly with `jupyter nbconvert --execute` and unlimited execution timeout.

Individual jobs can also be selected:

```bash
./bs_run_remote_notebooks.sh convnext vit ensemble
./gs_run_remote_notebooks.sh training
```

Set `BEACH_PYTHON` when a specific interpreter is required:

```bash
BEACH_PYTHON=/path/to/python ./bs_run_remote_notebooks.sh all
```

Each job writes a debug log to the repository root. Executed notebooks are
written to `Notebooks/executed_notebooks/`; this generated directory is ignored
by Git.

## Results

All results use the same stage prefixes as the notebooks: `bs_` for beach-surface
classification and `gs_` for grain-size classification. For example:

```text
Results/
|-- bs_convnext_tiny/          # Other backbones use the same bs_ prefix
|-- bs_ensemble/
|-- bs_all_cells_classification/
|-- gs_convnext_tiny/          # Other backbones use the same gs_ prefix
|-- gs_ensemble/
`-- gs_all_cells_classification/
```

These folders are created when the corresponding notebooks run. Previous
generated results were removed from the working tree and remain in Git history;
historical model checkpoints are stored through Git LFS. Existing notebook
outputs are retained and should be regenerated for the new evaluation protocol.
An ignored local `Results/Archive/` folder can hold restored historical results;
it is not used for current training, evaluation, or inference.

Model-specific directories under `Results/` contain the selected configuration,
grid-search summaries and traces, normalization statistics, validation and test
metrics, predictions, inference metadata, and the best loadable checkpoint.
Ensemble directories contain validation-derived weights, predictions,
classification reports, and confusion matrices.

Grain-size evaluation artifacts are kept in each `Results/gs_<model>/`
directory under `cross_validation/fold_<n>/`. Each fold saves inner grid scores,
per-class and epoch traces, selected configuration, inner OOF probabilities,
outer predictions, checkpoint, normalization, metrics, and timing.
`training_cells.csv`, `evaluation_cells.csv`, and `inner_partition_summary.csv`
record the fit membership, fold sizes, and buffer exclusions for auditability.
`cross_validation/oof_predictions.csv` pools all outer predictions, with one
prediction per cell. `fold_metrics_and_timing.csv`, `cv_summary.csv` (mean and
sample standard deviation), `fold_per_class_metrics.csv`,
`cv_per_class_summary.csv`, and pooled reports provide complementary summaries.
The model directory root contains the separate all-data deployment checkpoint,
metadata, configuration search, inner OOF predictions, and deployment timings.
`Results/gs_ensemble/cross_validation/` stores the corresponding results for
all four ensembles; its root `ensemble_definition.json` contains deployment
weights compatible with the existing whole-dataset classification notebook.

Timing uses wall-clock seconds with accelerator synchronization. Beach-surface
models save `timing_summary.csv`; the ensemble also saves
`test_metrics_and_timing.csv`. Grain-size fold timing separates normalization,
inner search elapsed time, summed inner fit work, outer refit training, and
held-out prediction. Training includes forward/backward passes and inner
validation used by early stopping; prediction includes DataLoader I/O.
Ensemble prediction time is the sum of component prediction times plus voting;
ensemble training cost includes the shared backbone training and weight fitting.
These shared costs must not be added across all four ensemble methods as if
they required four separate sets of models. CPU voting is timed separately;
checkpoint loading is included in beach-surface ensemble inference, while
grain-size timing uses already loaded outer models and cached probabilities.
Model construction/downloads and writing result files are excluded from fit
and prediction intervals. Both reporting notebooks display timings and print
LaTeX alongside quality results. Grain-size plots/tables use outer OOF predictions.

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

## Funding

The study was funded by the National Science Centre of Poland through grants 2023/51/D/ST6/01609 (SONATA 19) and 2021/40/C/ST10/00146 (SONATINA 5), and by the Warsaw University of Technology Research University--Excellence Initiative through grant CPR-IDUB/288/Z01/POB3/2024.

## License

The source code, shell scripts, and notebook code and explanatory text are
licensed under the GNU
General Public License version 3.0 (`GPL-3.0-only`). You may use, modify, and
redistribute them, including for commercial purposes, provided that distributed
copies and derivative works remain under GPLv3 and their corresponding source
code is made available. See [LICENSE](LICENSE) for the complete terms.

Research data are published in a separate repository and are governed by the
licence stated in that repository. The software licence here does not relicense
those data, third-party datasets, pretrained weights, or other third-party
materials.

The [NCN Open Science policy](https://www.ncn.gov.pl/finansowanie-nauki/otwarta-nauka)
specifies licensing provisions for publications and research data, but does not
designate a preferred software licence. GPLv3 is this project's software
licence, not an NCN-mandated choice.

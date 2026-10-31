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
- EfficientNetV2-S, adapted to the selected input channels.
- ViT-Tiny (`vit_tiny_patch16_224`), adapted to the selected input channels.
- Swin-Tiny, adapted to the selected input channels.
- ConvNeXt-Tiny, adapted to the selected input channels.

The four pretrained backbones are also compared using hard voting, probability
voting, globally weighted probability voting, and model-and-class weighted
probability voting. The Small CNN is retained as a standalone baseline.

## Channel Ablation

Both stages compare four input setups: `RGB_only` (3 channels), `RGB_slope`
(4 channels), `RGB_elevation` (4 channels), and `RGB_slope_elevation` (5 channels).
`Source/channel_setups.py::get_channel_setup` reads `BEACH_CHANNEL_SETUP`;
when unset, an individual training or evaluation notebook defaults to
`RGB_slope_elevation`. Each notebook run processes just one selected setup.
The batch launchers instead default to running all four setups sequentially.

The ablation preserves the beach-surface survey holdout, grain-size five-fold single-validation
spatial-group CV, context-radius search, training-only normalization, model
selection, and ensemble-weight protocols described below. Splits and fold plans
are shared across setups and generated only once. Running all four setups means
four complete training/evaluation runs, approximately **4x the computational
cost** of one setup; actual time depends on channel count and early stopping.

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

Grain-size evaluation uses **five outer folds and one training/validation split
per fold**, saved as `gs_cv_samples.csv`, `gs_cv_validation_splits.csv`, and
`gs_cv_manifest.json`. The old `gs_cv_inner_folds.csv` is historical and unused.
Each of the 140 cells is in exactly one outer evaluation fold. Same-survey
cells with intersecting maximum-radius contexts are assigned to the same
connected spatial group using DBSCAN (Chebyshev distance, `min_samples=1`).
For maximum radius 3, centres separated by at most 6 cells are linked;
transitively linked cells remain in the same group, even if their labels differ.
StratifiedGroupKFold assigns intact groups to outer folds and validation holdouts while
requiring all three classes in every training and evaluation partition.

There are 129 groups in the current annotations; outer evaluation fold sizes
are 30, 27, 27, 29, and 27 cells. Each remaining outer-training partition is split
once into training and approximately 20% validation, retaining all classes.
These holdouts are not rotated as inner cross-validation. All boundaries apply
a training buffer excluding centres
at distance <= 6 from a same-survey held-out centre and verify zero context
intersection. The connected-group construction already separates all current
groups by more than 6 cells, so the buffer checks do not discard any additional
current annotations. This common maximum-radius separation is used for all
tested radii `[0, 1, 2, 3]`; a larger grid radius requires regenerating the plan.

Each of the 16 configurations is fitted once per fold and scored separately on
validation and outer test cells. Validation loss controls early stopping.
Mean validation macro F1 across five runs selects one global configuration;
its median best epoch determines the fixed all-data refit length. Test labels
never directly control selection, normalization, stopping or ensemble weights.
The same folds serve every
model. This is **spatial-group CV with same-survey context isolation**, not
survey-held-out validation or geographic isolation across different dates.

**Selection limitation:** this is not nested CV. The global configuration is
selected across all five validation holdouts, and another run's training or
validation can include a current run's test cell. Global-winner test summaries
are therefore not an independent nested-CV estimate. All configurations' fold
metrics and their means/sample standard deviations are saved, with validation
and test values in separate columns.

For each ensemble outer fold, weights are calculated from the selected
backbones' single-split validation probabilities for that fold.
Hard voting resolves ties using mean probabilities among tied classes.
The outer-held-out predictions evaluate all four ensemble rules independently.
The selected configuration is freshly trained on all labelled cells, without
another deployment search. No fold checkpoints are saved. Deployment ensemble
weights use pooled validation predictions of the selected configurations across
the five runs; cells can recur in that pool. They never use test labels directly.
With five backbones, four channel setups and 16 configurations, this is
`5 * 4 * (5 * 16 + 1) = 1,620` fits, instead of 5,880.
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
five-fold single-validation evaluation and refits the mean-validation winner on all data. Both save checkpoints
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
- `bs_classify_all_cells_weighted_per_class_ensemble.ipynb`
- `gs_classify_beach_surfaces_weighted_ensemble.ipynb`

The inference notebooks read matrices directly from `Data/Raw/beach_all`; they
do not copy the source dataset. Outputs are separated by the selected channel
setup. ConvNeXt-only beach-surface inference writes to
`Results/bs_all_cells_classification/<setup>/<date>/classification.csv`.
Global ensemble inference writes to
`Results/bs_all_cells_classification/weighted_ensemble/<setup>/<date>/classification.csv`;
grain-size inference writes to
`Results/gs_all_cells_classification/weighted_ensemble/<setup>/<date>/classification.csv`.

The beach-surface `weighted_per_class` notebook explicitly uses
model-and-class weighted probability voting with the saved class-specific
validation F1 weights. It reuses the existing best checkpoints and each model's
own context and normalization; no retraining is needed. Output is kept separate
at `Results/bs_all_cells_classification/weighted_ensemble_per_class/<setup>/<date>/classification.csv`.
The existing global-weighting notebook writes to `weighted_ensemble/<setup>/`.
The class-weighted notebook prints seconds per backbone and survey, ensemble
calculation time, and total survey time. Timings are saved to each survey's
`prediction_timing.csv` and the setup output root's `classification_summary.csv`, updated after
each survey. `run_timing.json` records completed-run wall time including model
loading. Backbone timings include input I/O and preprocessing; ensemble timings
exclude those operations. Survey totals include prediction CSV writing but not
timing-report writing.
The grain-size inference notebook reads the matching setup's global BS output path;
to run grain-size inference on the new class-weighted surface maps, set its
`FIRST_STAGE_CLASSIFICATION_ROOT` to
`Results/bs_all_cells_classification/weighted_ensemble_per_class/<setup>/`.

Run the class-weighted all-cell notebook as a detached batch job:

```bash
mkdir -p Notebooks/executed_notebooks
nohup python -u -m jupyter nbconvert \
  --debug --to notebook \
  --execute Notebooks/bs_classify_all_cells_weighted_per_class_ensemble.ipynb \
  --output bs_classify_all_cells_weighted_per_class_ensemble_executed.ipynb \
  --output-dir Notebooks/executed_notebooks \
  --ExecutePreprocessor.timeout=-1 \
  > bs_classify_all_cells_weighted_per_class_ensemble_debug.log 2>&1 &
```

## Remote Batch Execution

The launch scripts execute notebooks sequentially in a detached `nohup`
pipeline. From the repository root, run:

```bash
./bs_run_remote_notebooks.sh all
./gs_run_remote_notebooks.sh all
```

The `all` and `training` modes first execute the corresponding split/fold
notebook once, independently of channel selection. Then, for each setup, `all`
runs all five model jobs followed by ensemble evaluation; `training` omits the
ensemble. Explicit jobs run in the supplied order within each setup and reuse
the saved metadata unless `splits` is requested. An explicitly requested
`splits` job runs once before all setup-dependent jobs, even if repeated or
listed last. Selecting only `splits` does not run training or evaluation.
Grain-size five-fold grid search requires more fits than a single split;
grid configurations are distributed
over the dynamically detected CUDA devices. Each notebook can also be executed
directly with `jupyter nbconvert --execute` and unlimited execution timeout.

Individual jobs can also be selected:

```bash
./bs_run_remote_notebooks.sh convnext vit ensemble
./gs_run_remote_notebooks.sh training
```

To run a single setup, set `BEACH_CHANNEL_SETUP` for either launcher:

```bash
BEACH_CHANNEL_SETUP=RGB_only ./bs_run_remote_notebooks.sh all
BEACH_CHANNEL_SETUP=RGB_elevation ./gs_run_remote_notebooks.sh convnext ensemble
```

Only the four setup names listed above are accepted; invalid values (including
an explicitly empty value) are rejected before any detached worker launches.
Unset the variable to restore the four-setup batch default. For direct notebook
execution, set the variable in the notebook process/kernel environment; leaving
it unset selects only `RGB_slope_elevation`, not the full batch.

Set `BEACH_PYTHON` when a specific interpreter is required:

```bash
BEACH_PYTHON=/path/to/python ./bs_run_remote_notebooks.sh all
```

Each setup-dependent job writes a debug log to the repository root named
`<notebook_stem>_<setup>_debug.log`. Its executed notebook is written to
`Notebooks/executed_notebooks/<setup>/<notebook_stem>_executed.ipynb`.
The channel-independent split job retains `<notebook_stem>_debug.log` and
`Notebooks/executed_notebooks/<notebook_stem>_executed.ipynb`. This generated
notebook directory is ignored by Git. A pipeline log records the sequential
run; a failing notebook stops the pipeline. Rerunning a job for the same setup
replaces that job's debug log and executed notebook.

## Results

All results use the same stage prefixes as the notebooks: `bs_` for beach-surface
classification and `gs_` for grain-size classification. For example:

```text
Results/
|-- bs_convnext_tiny/          # Other backbones use the same bs_ prefix
|   |-- RGB_only/
|   |-- RGB_slope/
|   |-- RGB_elevation/
|   `-- RGB_slope_elevation/
|-- bs_ensemble/
|-- bs_all_cells_classification/
|-- gs_convnext_tiny/          # Other backbones use the same gs_ prefix
|-- gs_ensemble/
`-- gs_all_cells_classification/
```

Training and ensemble outputs are isolated under
`Results/{bs,gs}_<model>/<setup>/`, including
`Results/bs_ensemble/<setup>/` and `Results/gs_ensemble/<setup>/`.
All four setup names use the same layout; no setup overwrites another's
checkpoints, metrics, or ensemble weights. Historical files directly in a
model's root are not new setup-specific results.

BS training and ensemble evaluation leave `experiment_in_progress.json` in the
setup directory until all required outputs have been saved. Inference and
reporting reject a directory with this marker, including one left by an
interrupted rerun; rerun the corresponding notebook to completion. GS uses its
existing CV completion manifests for the same purpose. Grid-score ties retain
the original configuration order, so the first summary row agrees with the
saved winner.

New ensemble definitions fingerprint the exact component metadata. Loading or
reporting rejects changed backbones rather than reusing their old ensemble
weights, and checkpoint checksums are verified when recorded in metadata.
Legacy BS ensemble definitions without fingerprints emit a warning during
loading; rerun `bs_evaluate_ensemble.ipynb` for the relevant setup before
deployment to generate a fingerprinted definition. This does not require
retraining unchanged backbones.

These folders are created when the corresponding notebooks run. Existing results
for the five beach-surface models and their ensemble were moved into their
`RGB_slope_elevation/` subfolders, preserving scores and checkpoints. The baseline
ensemble's stored model paths are relative and point to the relocated setup
directories. This relocation does not retrain or reevaluate the models.
Historical model checkpoints are stored through Git LFS. Existing notebook
outputs are retained and should be regenerated for the new evaluation protocol.
An ignored local `Results/Archive/` folder can hold restored historical results;
it is not used for current training, evaluation, or inference.

Model-specific setup directories under `Results/` contain the selected
configuration, grid-search summaries and traces, normalization statistics, validation and test
metrics, predictions, inference metadata, and the best loadable checkpoint.
Ensemble directories contain validation-derived weights, predictions,
classification reports, and confusion matrices.

Grain-size evaluation artifacts are kept in each `Results/gs_<model>/<setup>/`
directory under `cross_validation/fold_<n>/`. Each fold saves configuration scores,
per-class and epoch traces, validation/test predictions for every configuration,
and metrics/timing. There are no fold model checkpoints.
`training_cells.csv`, `validation_cells.csv`, `evaluation_cells.csv`, and `partition_summary.csv`
record the fit membership, fold sizes, and buffer exclusions for auditability.
`cross_validation/oof_predictions.csv` pools all outer predictions, with one
prediction per cell. `fold_metrics_and_timing.csv`, `cv_summary.csv` (mean and
sample standard deviation), `fold_per_class_metrics.csv`,
`cv_per_class_summary.csv`, and pooled reports provide complementary summaries.
`configuration_fold_metrics.csv`, `configuration_summary.csv`, and
`configuration_per_class_summary.csv` report every configuration, with separate
validation/test metrics and mean/SD values. The setup directory root contains
the all-data deployment checkpoint, metadata, the same averaged configuration
summary, and deployment timings. `selection.json` records the winner and caveat.
`Results/gs_ensemble/<setup>/cross_validation/` stores the corresponding results for
all four ensembles; its root `ensemble_definition.json` contains deployment
weights compatible with the existing whole-dataset classification notebook.

Timing uses wall-clock seconds with accelerator synchronization. Beach-surface
models save `timing_summary.csv`; the ensemble also saves
`test_metrics_and_timing.csv`. Grain-size fold timing separates normalization,
single-split search elapsed time, summed configuration fit work, final all-data
refit training, and held-out prediction. Training includes forward/backward passes and
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

Appended channel-ablation comparison sections in
`bs_example_metrics_reporting.ipynb` and `gs_example_metrics_reporting.ipynb`
compare the available setup-specific quality metrics across backbones and
ensemble rules. These appended comparisons do not currently compare computational
timings; timing artifacts are still saved separately for each setup.
Beach-surface comparisons use the preserved
test holdout; grain-size comparisons use outer-fold/OOF evaluation, not deployment
training scores. Run these reporting notebooks after the setup batches finish.

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

# Context-Aware Deep Learning Ensembles for Beach-Surface and Grain-Size Classification in Isbjørnhamna, Svalbard

Training, ensemble evaluation, inference, and reporting for two-stage
classification of UAV-derived beach cells:

1. **Beach surface (`bs_`):** ten classes labelled in `Data/Raw/Descriptor_fs.csv`.
2. **Grain size (`gs_`):** classes 11–13 labelled in `Data/Raw/Descriptor_ss.csv`.
   Full-dataset inference processes only cells assigned to beach-surface
   class 1 (Dry sediment) or 2 (Wet sediment).

## Models and Data

Five backbones are evaluated: Small CNN, EfficientNetV2-S, ViT-Tiny, Swin-Tiny,
and ConvNeXt-Tiny. The four pretrained backbones also form ensembles using
hard voting, probability voting, globally weighted probability voting, and
model-and-class weighted probability voting. Small CNN is a standalone baseline.

Raw data must be provided separately at `Data/Raw/beach_all/<date>/`. Each cell
has five 20 x 20 matrices: `<cell_id>r.txt`, `<cell_id>g.txt`,
`<cell_id>b.txt`, `<cell_id>s.txt`, and `<cell_id>d.txt` (red, green,
blue, slope, elevation). Source matrices are read directly, not copied.

Experiments compare `RGB_only`, `RGB_slope`, `RGB_elevation`, and
`RGB_slope_elevation`. Context radii `[0, 1, 2, 3]` produce same-survey
mosaics with sides of 20, 60, 100, and 140 pixels. Pretrained models resize
them to 224 x 224; Small CNN uses native sizes.

NaN, `-999`, and unavailable neighbours are filled with per-channel means.
Normalization statistics use only the current training partition's central-cell
pixels, are shared across its context radii, and are saved for inference.

## Layout

- `Data/Raw/`: descriptors, class specifications, and survey matrices.
- `Data/Preprocessed/`: split CSVs, fold plans, and provenance metadata.
- `Source/`: shared implementation.
- `Notebooks/`: executable workflows prefixed `bs_` or `gs_`.
- `Results/`: checkpoints, predictions, metrics, timings, and ensemble weights.
- `Notebooks/executed_notebooks/`: batch outputs, ignored by Git.

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

For remote CUDA 12.8, use `requirements-remote-gpu-cu128.txt` instead. It pins
PyTorch 2.8.0, torchvision 0.23.0, and torchaudio 2.8.0. Check GPU availability:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Checkpoints use Git LFS. After installing Git LFS, retrieve them with:

```bash
git lfs install
git lfs pull
```

## Evaluation Protocols

**Beach surface:** survey-disjoint partitions contain 5,386 training, 1,243
validation, and 1,493 test cells. Test surveys are `20180812`, `20180915`,
and `20220910`; validation surveys are `20230731` and `20240623`.
Other surveys are training-only. All ten classes occur in each partition.
Same-survey contexts cannot overlap across partitions, although geographic
locations can recur on different dates.

**Grain size:** 140 labelled cells undergo five-fold stratified spatial-group
evaluation, with one approximately 80:20 training/validation split within each
fold's remaining data. Groups and buffer checks prevent overlapping same-survey
contexts at maximum radius 3 across partitions. All three classes occur in every
partition, and the same plan serves all models, channels, and radii. Current
plans are `gs_cv_samples.csv`, `gs_cv_validation_splits.csv`, and
`gs_cv_manifest.json`; historical `gs_*_split.csv` and
`gs_cv_inner_folds.csv` files are unused.

Each model/setup tests 16 configurations: two learning rates, two weight-decay
values with coupled dropout settings, and four radii. Validation loss controls
early stopping; validation macro-F1 selects configurations. For grain size,
mean validation macro-F1 across five runs selects one configuration, freshly
refitted on all labels using the median best epoch. No fold checkpoints are
saved. Five backbones across four channel setups require 1,620 grain-size fits,
including deployment refits.

**Selection limitation:** grain-size evaluation is not independent nested CV.
A cell held out in one run can enter another run's training or validation data,
which contribute to global configuration selection. Global-winner fold metrics
require this qualification. Ensemble weights use validation predictions only;
deployment weights pool selected configurations' validation predictions, where
cells can recur. Reporting each method's best channel by test/held-out F1 is
descriptive, not an independent channel-selection estimate.

## Run Experiments

For either stage, run `*_create_dataset_splits.ipynb`, the five
`*_train_*_classifier.ipynb` notebooks, `*_evaluate_ensemble.ipynb`, then
`*_example_metrics_reporting.ipynb`.
Feature-distribution notebooks are `*_plot_feature_distributions_by_class.ipynb`.

Remote launchers run sequential notebook jobs in a detached `nohup` pipeline,
allowing the terminal to close:

```bash
bash bs_run_remote_notebooks.sh all
bash gs_run_remote_notebooks.sh all
```

`all` generates splits once, then trains all five backbones and evaluates
ensembles for each channel setup. `training` omits ensembles. Select individual
jobs, channels, or an interpreter as follows:

```bash
bash bs_run_remote_notebooks.sh convnext vit ensemble
BEACH_CHANNEL_SETUP=RGB_elevation bash gs_run_remote_notebooks.sh convnext ensemble
BEACH_PYTHON=/path/to/python bash bs_run_remote_notebooks.sh all
```

Launchers default to all four channel setups. Direct notebooks default to
`RGB_slope_elevation`; set `BEACH_CHANNEL_SETUP` in the kernel/process
environment to select another setup. Grid-search work uses dynamically detected
CUDA devices. Avoid concurrent pipelines unless resources permit.

Logs go to the repository root as `<notebook_stem>_<setup>_debug.log`;
executed notebooks go to
`Notebooks/executed_notebooks/<setup>/<notebook_stem>_executed.ipynb`.
Split jobs omit the setup component. A failed job stops the pipeline; rerunning
overwrites that job's log and executed notebook.

## Full-Dataset Classification

Choose the notebook for the intended saved model or ensemble:

- `bs_classify_all_cells_convnext.ipynb`: ConvNeXt only.
- `bs_classify_all_cells_weighted_ensemble.ipynb`: global model weights.
- `bs_classify_all_cells_weighted_per_class_ensemble.ipynb`: model/class weights.
- `gs_classify_beach_surfaces_weighted_ensemble.ipynb`: grain-size inference.

Check channel and input/output settings before execution. In particular, set
`FIRST_STAGE_CLASSIFICATION_ROOT` in the grain-size notebook to the intended
beach-surface classifications. Components retain their saved contexts and
normalization; ensembles use components from one channel setup only.

Example detached inference job:

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

Each survey receives a `classification.csv` with `date`, `cell_id`, and
`predicted_class`. Global ensemble outputs use
`Results/{bs,gs}_all_cells_classification/weighted_ensemble/<setup>/<date>/`;
beach-surface model/class weighting uses
`Results/bs_all_cells_classification/weighted_ensemble_per_class/<setup>/<date>/`.
ConvNeXt-only outputs omit the ensemble-method directory. Timing records cover
backbones, ensemble calculation, and total survey processing.

## Results and Reporting

Training/evaluation outputs are isolated under `Results/{bs,gs}_<model>/<setup>/`
and `Results/{bs,gs}_ensemble/<setup>/`. They include selected configurations,
grid-search traces, normalization statistics, predictions, metrics, timings,
and deployment checkpoints with inference metadata. Grain-size
`cross_validation/` directories additionally retain fold membership, every
configuration's metrics, pooled held-out predictions, and mean/sample-SD
summaries. Pooled metrics differ from averages of fold metrics.

Provenance fingerprints and completion markers guard against stale/incomplete
artifacts. Finish interrupted experiments before inference/reporting. Regenerate
splits and retrain after changing the protocol or maximum radius. Retained
notebook outputs may reflect earlier runs; `Results/Archive/` is ignored and
unused by current workflows.

Reporting notebooks provide accuracy, macro precision/recall/F1, MCC,
per-class metrics, confusion matrices, precision-recall curves, configuration
tables, channel/context comparisons, and runtime plots, with LaTeX tables.
Training timings include epoch validation; prediction timings include data
loading. Do not count shared backbone costs separately for each ensemble rule.

Both reporting notebooks expose **`FIGURE_DIR` near the top**. Its current
default is `../Beach_classification_manscript/fig/`, relative to the project
root. Change it for another layout, or set it to `None` to disable PDF export.

## Funding

The study was funded by the National Science Centre of Poland through grants 2023/51/D/ST6/01609 (SONATA 19) and 2021/40/C/ST10/00146 (SONATINA 5), and by the Warsaw University of Technology Research University--Excellence Initiative through grant CPR-IDUB/288/Z01/POB3/2024.

## License

Code, scripts, and notebook code/text use **GPL-3.0-only**; see [LICENSE](LICENSE).
Research data are published separately under their own licence. This licence
does not relicense third-party datasets or pretrained weights. GPLv3 is the
project's software licence, not an NCN-mandated choice.

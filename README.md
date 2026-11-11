# Context-Aware Deep Learning Ensembles for Beach-Surface and Grain-Size Classification in Isbjørnhamna, Svalbard

Code and saved results for two-stage classification of UAV-derived beach cells using Small CNN, EfficientNetV2-S, ViT-Tiny, Swin-Tiny, ConvNeXt-Tiny, and four voting ensembles. RGB, slope, and elevation channel combinations and context radii 0–3 are compared. Town Neck Beach experiments evaluate RGB-only, site-specific fine-tuning of the Svalbard models.

## File Structure

| Path | Contents |
| --- | --- |
| `Data/Raw/` | Separately supplied survey matrices, label descriptors, and class mappings; not tracked by Git. |
| `Data/Preprocessed/` | Beach-surface splits, grain-size cross-validation plans, and provenance metadata. `bs_town_neck/` contains the Town Neck sampled spatial splits. |
| `Source/` | Shared data loading, models, training, grid search, ensembles, inference, timing, and reporting code. |
| `Notebooks/bs_*.ipynb` | Beach-surface split creation, training, ensemble evaluation, all-cell classification, and reporting. |
| `Notebooks/gs_*.ipynb` | Grain-size cross-validation, deployment training, ensemble evaluation, classification, and reporting. |
| `Notebooks/bs_town_neck_*.ipynb` | Town Neck preparation, sampled splits, transfer-learning grid search, ensemble evaluation, and reporting. |
| `Notebooks/report_dataset_sizes.ipynb` | Reports dataset storage sizes. |
| `Notebooks/executed_notebooks/` | Executed batch notebooks; ignored by Git. |
| `Results/` | Model checkpoints, selected configurations, predictions, metrics, timings, and ensemble definitions, separated by dataset, method, and channel setup. Grain-size `cross_validation/` directories contain fold-level results. |
| `bs_run_remote_notebooks.sh` | Detached batch launcher for beach-surface experiments. |
| `gs_run_remote_notebooks.sh` | Detached batch launcher for grain-size experiments. |
| `bs_town_neck_run_remote_notebooks.sh` | Detached batch launcher for Town Neck experiments; logs are in `logs/bs_town_neck/`. |
| `requirements.txt` | Packages for the standard Python environment. |
| `requirements-remote-gpu-cu128.txt` | Remote GPU dependencies, including PyTorch 2.8.0 with CUDA 12.8 wheels. |
| `LICENSE` | GPLv3 software licence text. |

Main source modules:

- `beach_dataset.py`, `channel_setups.py`: matrix loading, spatial contexts, missing values, normalization, and channel selection.
- `bs_splits.py`, `cross_validation.py`: survey-disjoint beach-surface splits and spatial-group grain-size evaluation.
- `models.py`, `training.py`, `grid_search.py`: backbone construction, training, validation, and parameter search.
- `ensemble.py`, `inference.py`, `timing.py`: voting methods, saved-model prediction, and runtime records.
- `metrics_reporting.py`: quality metrics, plots, and LaTeX tables.
- `bs_town_neck_*.py`: Town Neck preprocessing, spatial sampling, transfer learning, and reporting.

## Essential Information

- Prefixes: `bs_` means beach surface, `gs_` means grain size, and `bs_town_neck_` identifies the external-dataset experiments.
- Raw Svalbard cells belong in `Data/Raw/beach_all/<date>/`. Each cell has five 20 × 20 text matrices named `<cell_id>{r,g,b,s,d}.txt`. Labels are in `Descriptor_fs.csv` and `Descriptor_ss.csv`. Raw data must be obtained separately.
- Beach-surface evaluation uses survey-disjoint training, validation, and test subsets. Grain-size evaluation uses five spatial-group folds with one training/validation split per run; configuration selection averages validation scores across runs. It is not independent nested cross-validation. Best-channel comparisons based on test/held-out F1 are descriptive.
- Grain-size all-cell inference processes only cells assigned to beach-surface class 1 or 2. Model components retain their saved channels, context radii, and training-derived normalization statistics.
- Checkpoints (`*.pt`) use Git LFS. Install Git LFS and run `git lfs pull` after cloning to retrieve model files.
- Batch experiments: run `bash bs_run_remote_notebooks.sh all`, `bash gs_run_remote_notebooks.sh all`, or `bash bs_town_neck_run_remote_notebooks.sh all`. Launchers use `nohup`; all-cell classification is run separately through the corresponding notebook.
- Reporting notebooks expose `FIGURE_DIR` near the beginning. Change its manuscript-folder default for your environment, or set it to `None` to disable PDF export.

## Funding

The study was funded by the National Science Centre of Poland through grants 2023/51/D/ST6/01609 (SONATA 19) and 2021/40/C/ST10/00146 (SONATINA 5), and by the Warsaw University of Technology Research University--Excellence Initiative through grant CPR-IDUB/288/Z01/POB3/2024.

## Licence

Code, scripts, and notebook code/text use **GPL-3.0-only**; see [LICENSE](LICENSE). Research data are distributed separately under their own licences. This software licence does not relicense third-party datasets or pretrained weights.

# KineLearn
Pose-based behavior classification from DeepLabCut keypoints.

KineLearn is designed to bridge **pose estimation** and **behavior classification**.  
It builds on DeepLabCut keypoint data to extract interpretable kinematic features (e.g., angles, distances, relative coordinates) and align them with user-defined behavioral annotations.  

KineLearn now supports the full core workflow for pose-based behavior modeling: **feature extraction**, **dataset splitting**, **single-behavior model training**, **evaluation**, **inference on new videos**, and **basic output visualization**. More experiment-specific downstream analyses, such as stimulus-aligned summaries, are better handled by companion tools built on top of KineLearn outputs.

## Table of Contents
- [Installation](#installation)
- [🚀 Typical Workflows](#-typical-workflows)
- [🧩 Using KineLearn to Generate Features](#-using-kinelearn-to-generate-features)
- [🧭 Splitting Data into Train and Test Sets](#-splitting-data-into-train-and-test-sets)
- [🧠 Training a Behavior Classifier](#-training-a-behavior-classifier)
- [📚 Batch-Evaluating Split Sweeps](#-batch-evaluating-split-sweeps)
- [📊 Evaluating Predictions](#-evaluating-predictions)
- [🧮 Creating Ensembles](#-creating-ensembles)
- [🎯 Selecting Ensemble Members](#-selecting-ensemble-members)
- [🔮 Running Inference on New Videos](#-running-inference-on-new-videos)
- [🎨 Visualizing Behavioral Dynamics](#-visualizing-behavioral-dynamics)
- [🗄️ Archiving Result Directories](#archiving-result-directories)
- [🧰 Restoring Archived Run Artifacts](#-restoring-archived-run-artifacts)
- [🧪 Running Tests](#-running-tests)

---

## Installation

### 1. Prerequisites
- **Python:** 3.11 (recommended)
- **CUDA-enabled GPU (optional):**  
  If you plan to train models on GPU, make sure your system has:
  - NVIDIA driver ≥ 535
  - CUDA 12.x toolkit
  - cuDNN 9.x  
  These are installed automatically when using `tensorflow[and-cuda]` if the drivers are present.

> 💡 If you don’t have an NVIDIA GPU, TensorFlow will fall back to CPU automatically.


### 2. Create a virtual environment
You can use **Conda**, **venv**, or any other environment manager.

**Using Conda (recommended):**
```bash
conda create -n kinelearn python=3.11
conda activate kinelearn
```

**Using venv:**
```bash
python3.11 -m venv kinelearn
source kinelearn/bin/activate
```

### 3. Install dependencies
Install all required Python packages:
```bash
pip install -r requirements.txt
```

The main dependencies include:
- TensorFlow (with GPU support if available)
- Keras
- Scikit-learn
- OpenCV
- Matplotlib / Seaborn for visualization
- Pandas / PyArrow for data handling


### 4. Verify the environment
Check that TensorFlow detects your GPU (optional):

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Expected output (if GPU is available):
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### 5. Installing KineLearn itself

You can install KineLearn either as a **user package** or in **editable (developer) mode**.

**A. Standard installation (recommended for users):**
```bash
pip install .
```
This installs KineLearn normally, adding the CLI commands  
`kinelearn-calc`, `kinelearn-split`, `kinelearn-train`, `kinelearn-eval`, `kinelearn-predict`, `kinelearn-create-ensemble`, `kinelearn-select-ensemble`, `kinelearn-restore-run-artifacts`, `kinelearn-plot-timeline`, `kinelearn-split-variability`, `kinelearn-batch-eval-splits`, and `kinelearn-archive-results` to your PATH.

**B. Developer installation (for code modification):**
```bash
pip install -e .
```
The `-e` (editable) flag links the package directly to your working directory,  
so any changes you make to the source code are applied immediately—  
no need to reinstall unless you modify `pyproject.toml` or add new dependencies.

💡 *Why the difference?*  
- Use `pip install .` if you’re just **using** KineLearn (e.g., for feature extraction or model training).  
- Use `pip install -e .` if you’re **developing** KineLearn and want edits to take effect instantly.  

### 6. Verify command-line tools

After installation, check that the KineLearn CLI commands are recognized:

```bash
kinelearn-calc --help
kinelearn-split --help
kinelearn-train --help
kinelearn-eval --help
kinelearn-predict --help
kinelearn-create-ensemble --help
kinelearn-select-ensemble --help
kinelearn-restore-run-artifacts --help
kinelearn-plot-timeline --help
kinelearn-split-variability --help
kinelearn-batch-eval-splits --help
kinelearn-archive-results --help
```

If these commands run successfully, your environment is correctly configured.

---
## 🚀 Typical Workflows

### Train and evaluate models

1. Run `kinelearn-calc` on a labeled video set to generate `frame_features_*.parquet` and `frame_labels_*.parquet`.
2. Create a train/test split with `kinelearn-split`, or use `kinelearn-split-variability` when you want to generate repeated split experiments.
3. Train one model per behavior with `kinelearn-train`.
4. Evaluate saved runs with `kinelearn-eval` or `kinelearn-batch-eval-splits`.
5. Optionally choose ensemble members from validation-performing runs with `kinelearn-select-ensemble`, or group already-chosen members with `kinelearn-create-ensemble`.

### Run inference on new videos

1. Run `kinelearn-calc` on new videos using the existing scalers from your config.
2. Apply one or more trained models with `kinelearn-predict`.
3. Inspect outputs with `kinelearn-plot-timeline`.

---
## 🧩 Using KineLearn to Generate Features

KineLearn extracts pose-based behavioral features from DeepLabCut (DLC) keypoints and prepares them for model training.  
Before running the feature extraction step, ensure the following setup is complete.

---

### 1. Prerequisites

You should already have:

- A **DeepLabCut project and trained model** that you’ve used to generate pose predictions (`.csv` or `.h5`) for your videos.
- Behavior annotations for those videos created in **BORIS**.

> ⚠️ **Note:** Currently, KineLearn supports one animal per video scene.

---

### 2. Export behavior annotations

After annotating behaviors in BORIS:
1. Go to **Export → Tabular events**.
2. Save each file in the same directory as its corresponding video and DLC output.
3. Use the following filename convention:

```
{stem_of_video_filename}_ground_truth.tsv
```

Example:

```
video_20250704_ground_truth.tsv
```

---

### 3. Create a KineLearn configuration file

Use the provided examples in `configs/` as a starting point.  
`configs/example_config.yaml` is a minimal template, while `configs/drosophila_example.yaml` is a more realistic worked example.  
This YAML file defines:
- `dlc_config`: Path to your DeepLabCut project configuration file.
- `behaviors`: List of behavior names that appear in your BORIS annotations.
- `features`:  
  - `ref_pt`: The keypoint used as the origin for relative coordinates.  
  - `body_length_pts`: A pair of keypoints defining body length (used to normalize relative coordinates).  
  - `distances`: Pairs of keypoints for distance features.  
  - `angles`: Triplets of keypoints for angle features.
- `training`:
  - training hyperparameters such as `epochs`, `batch_size`, `learning_rate`, and focal-loss settings
  - whether raw absolute keypoint coordinates are included in model input via `include_absolute_coordinates`
  - optional Gaussian noise injected into training windows via `keypoint_noise_std`, either globally or per behavior
  - optional architecture selection via `training.model.variant` for either the original recurrent model or an opt-in convolutional front end
  - optional final zero-fill parity cleanup via `final_zero_fill`

Example:

```yaml
dlc_config: /path/to/your/dlc/config.yaml

behaviors:
- grooming
- wing_extension
- abdomen_bend

features:
ref_pt: thorax
body_length_pts:
 - thorax
 - abdomen_tip
distances:
 - [head, thorax]
 - [thorax, abdomen_tip]
angles:
 - [head, thorax, abdomen_tip]

training:
  batch_size: 8
  epochs: 10
  keypoint_noise_std:
    grooming: 0.0
    wing_extension: 0.01
    abdomen_bend: 0.0
  model:
    variant: bilstm
  final_zero_fill: false
```

Set `training.keypoint_noise_std` to either a single float or a per-behavior mapping to add Gaussian noise to keypoint inputs during training only. A value of `0.01` matches the always-on noise used in the older training codepath; validation and test windows remain noise-free.
Set `training.model.variant: bilstm` to use the original recurrent-only model, or `training.model.variant: conv_bilstm` to add a lightweight `Conv1D` front end before the BiLSTM stack. For `conv_bilstm`, you can optionally provide `training.model.conv_frontend.filters` and `training.model.conv_frontend.kernel_sizes`; the defaults are `[32, 32]` and `[5, 3]`.

For behaviors that may depend on short local motion motifs, you can opt into a convolutional front end in the config:

```yaml
training:
  model:
    variant: conv_bilstm
    conv_frontend:
      filters: [32, 32]
      kernel_sizes: [5, 3]
```

Set `training.final_zero_fill: true` to apply one final `fillna(0)` pass after loading the per-video feature files and before windowing, which mirrors the old validation/training pipeline's last-stage NaN cleanup.

---
### 4. Create a video list file
Create a YAML file containing the list of videos you want to process.
Each line should be a full path to a video file:
```yaml
- /data/videos/fly_001.mp4
- /data/videos/fly_002.mp4
- /data/videos/fly_003.mp4
```
Save this file somewhere convenient, e.g. `video_lists/train_videos.yaml`.

---

### 5. Run feature extraction

Use the `kinelearn-calc` command-line tool to extract and scale features.

```bash
kinelearn-calc \
  -v video_lists/train_videos.yaml \
  --kl-config configs/drosophila_example.yaml \
  --create-scalers
```

* The `--create-scalers` flag fits new normalization scalers (`StandardScaler`) on the data in your training/validation/test set.
* The fitted scalers are automatically saved under `scalers/` and their paths added to your KineLearn config file.
* When you later process new videos (for inference or testing), **omit** `--create-scalers` so the existing scalers will be loaded automatically.

Each processed video will produce:
* `frame_features_<video_stem>.parquet`— per-frame features
* `frame_labels_<video_stem>.parquet`— per-frame labels (from BORIS)

Both files are saved in your output directory (default: `features/`).

---

### ✅ Example output structure
```
features/
├── extracted_features_video_20250704.csv
├── frame_features_video_20250704.parquet
├── frame_labels_video_20250704.parquet
scalers/
├── scaler_drosophila_coordinates.pkl
├── scaler_drosophila_velocity.pkl
├── ...

```

---

### Tips
* Ensure all videos in a single run share the same frame rate (the script checks this automatically).
* For DeepLabCut `.h5` files, KineLearn will convert them to `.csv` automatically.
* Missing or incomplete feature data are mean-imputed during export.
* Behavior labels not found in BORIS exports default to zeros (no behavior active).

---
## 🧭 Splitting Data into Train and Test Sets

Once you have generated the feature and label files for your videos,
the next step is to divide them into **training** and **testing** sets.

KineLearn provides a command-line tool for this: `kinelearn-split`.

---

### 1. Create a split

```bash
kinelearn-split \
  -v video_lists/all_videos.yaml \
  --seed 42
```

This command:
* Reads the list of video paths from the YAML file (`-v`).
* Randomly splits the videos into train and test sets (80/20 by default).
* Saves the split information as a YAML file under `data_splits/`.

Example auto-generated filename:

```
data_splits/all_videos_split_20251023_153805.yaml
```

---

### 2. Customize the split

You can control the split behavior with optional flags:

| **Flag** | **Description** | **Default** |
| :--- | :--- | :--- |
| `--seed` | Sets a random seed for reproducible splits. | None |
| `--test-fraction` | Fraction of videos to reserve for testing. | 0.2 |
| `--out` | Custom path for the split output YAML file. | Automatically generated under `data_splits/` |

Example:

```
kinelearn-split \
  -v video_lists/july_sessions.yaml \
  --test-fraction 0.25 \
  --seed 123 \
  --out data_splits/july_sessions_split.yaml
```

---

### 3. Split file format
The generated YAML file lists which video stems belong to each set:

```yaml
seed: 42
test_fraction: 0.2
train:
  - fly_001
  - fly_002
  - fly_003
test:
  - fly_004
```

These stems correspond directly to the feature and label filenames, e.g., `frame_features_fly_001.parquet`, `frame_labels_fly_001.parquet`.

---

### ✅ Example output structure

```
data_splits/
├── all_videos_split_20251023_153805.yaml
features/
├── frame_features_fly_001.parquet
├── frame_labels_fly_001.parquet
scalers/
├── scaler_drosophila_velocity.pkl
├── scaler_drosophila_angles.pkl

```

---

### Tips

* The split uses only **filename stems** (not full paths), so it's flexible across systems.
* The split file can be reused for consistent training/testing across multiple runs.
* You can generate different splits (e.g., by session, condition, or date) and store them in `data_splits/`.

---
## 🧠 Training a Behavior Classifier

The `kinelearn-train` command trains a single-behavior classifier from precomputed keypoint features.

It performs:
1. **Loading and validating data** — reads feature and label `.parquet` files for each video.
2. **Splitting into train/val/test** — uses the split file from `kinelearn-split`, applying the validation fraction defined in your config unless you provide an explicit validation split.
3. **Windowing the data** — converts frame-level features and labels into overlapping windows stored as efficient `.memmap` arrays.
4. **Building generators and model** — creates memmap-backed Keras generators and a keypoints-only BiLSTM model for the selected behavior.
5. **Training with focal loss** — optimizes a per-timestep sigmoid classifier, checkpointing on `val_loss`, with optional early stopping.
6. **Evaluating on test data** — reloads the best checkpointed weights and reports test metrics.
7. **Recording outputs** — saves all artifacts into a run-specific directory under `results/<behavior>/<timestamp>/`, including a `train_manifest.yml` file summarizing dataset sizes, feature dimensions, training hyperparameters, artifact paths, and evaluation results.
   You can also provide `--out-dir` when you want to place a run in an explicit directory instead of using a timestamped one.

---

### Example command

```bash
kinelearn-train \
  --kl-config configs/drosophila_example.yaml \
  --split data_splits/2025_jul_aug_with_ground_truth_split_20260326_103723.yaml \
  --behavior genitalia_extension
```

This will:
- Prepare `train`, `val`, and `test` memmaps under a run directory in `results/`
- Train a single-behavior BiLSTM using focal loss
- Save the best model weights to `results/<behavior>/<timestamp>/best_model.weights.h5`
- Save per-epoch logs to `results/<behavior>/<timestamp>/train_history.csv`
- Write a manifest to `results/<behavior>/<timestamp>/train_manifest.yml` with hyperparameters, artifact paths, and test metrics

Safety notes:
- Training runs are isolated by behavior and timestamp, so repeated runs do not overwrite one another.
- If training is interrupted with `Ctrl-C`, KineLearn saves partial run artifacts but does not write `train_manifest.yml`, so interrupted runs remain distinguishable from completed runs.

Optional CLI overrides:
- `--features-dir` to read features from a directory other than `features/`
- `--val-split` to provide an explicit train/validation split file instead of deriving validation videos from `training.val_fraction`
- `--epochs` to override `training.epochs`
- `--batch-size` to override `training.batch_size`
- `--seed` to override `training.seed` for a specific run
- `--focal-alpha` to override the focal-loss alpha for a specific training run
- `--keypoint-noise-std` to override the training-time Gaussian noise std for a specific run
- `--out-dir` to force the run into a specific output directory

Training config note:
- Set `training.include_absolute_coordinates: false` to exclude raw absolute `*_x` / `*_y` keypoint columns from model input while still retaining derived motion and geometry features.
- Set `training.early_stopping: true` to stop early when `val_loss` stops improving; `training.early_stopping_patience` and `training.early_stopping_min_delta` control its sensitivity.
- Use `--seed` when you want to change the train/validation split for a run without editing the config file; the resolved seed used for that run is recorded in the training manifest.
- Use `--val-split` when you need a fixed, explicit train/validation partition. The resolved `split`, `val_split`, and train/val/test video stems are recorded in `train_manifest.yml` for traceability.
- Use `--focal-alpha` when you want to tune alpha per split without changing the project-wide default in your config file; the resolved alpha used for that run is still recorded in the run manifest.
- Use `--keypoint-noise-std` when you want to tune training-time noise per run without changing the project-wide default in your config file; the resolved noise value used for that run is still recorded in the run manifest.

### Tuning focal alpha in practice

For some behaviors, especially when an early model settles into a high-recall / low-precision regime, the most effective next step may be to tune focal-loss `alpha` on the current split rather than immediately changing the architecture.

Recommended workflow:
1. Keep the split fixed.
2. Train the same behavior multiple times with different `--focal-alpha` values.
3. Evaluate each run on the validation subset.
4. Compare validation precision, recall, and F1 for that behavior.
5. Choose the alpha that gives the best balance for your use case.
6. Evaluate the selected run once on the test subset.

Example:

```bash
kinelearn-train \
  --kl-config configs/drosophila_example.yaml \
  --split data_splits/2025_jul_aug_with_ground_truth_split_20260326_103723.yaml \
  --behavior genitalia_extension \
  --focal-alpha 0.60

kinelearn-train \
  --kl-config configs/drosophila_example.yaml \
  --split data_splits/2025_jul_aug_with_ground_truth_split_20260326_103723.yaml \
  --behavior genitalia_extension \
  --focal-alpha 0.67

kinelearn-train \
  --kl-config configs/drosophila_example.yaml \
  --split data_splits/2025_jul_aug_with_ground_truth_split_20260326_103723.yaml \
  --behavior genitalia_extension \
  --focal-alpha 0.75
```

Then evaluate each run on validation data:

```bash
kinelearn-eval \
  --manifest results/genitalia_extension/<timestamp>/train_manifest.yml \
  --subset val \
  --level frame
```

Practical notes:
- KineLearn stores each run in its own `results/<behavior>/<timestamp>/` directory, so alpha sweeps do not overwrite one another.
- The resolved alpha for that run is written into `train_manifest.yml`, which makes it easy to compare runs later.
- Use validation results to choose alpha. Do not use test-set performance for alpha selection.
- If frame-level precision/recall tradeoffs are the main issue, start with frame-level validation metrics first; if episode quality matters more, also inspect `--level episode` or `--level both`.
- Alpha tuning and decision-threshold tuning are different levers: `alpha` changes how the model is trained, while `kinelearn-eval --threshold` changes how predicted probabilities are binarized at evaluation time.

### Measuring split variability

When a behavior appears sensitive to split choice, use `kinelearn-split-variability` to generate reproducible train/test and train/validation split experiments and optionally run them in batch.

Two practical modes are supported:

1. Hold the test split fixed and vary only the train/validation split.
2. Generate multiple outer train/test splits and multiple inner train/validation splits within each outer split.

Example: fixed historical test split, multiple validation seeds

```bash
kinelearn-split-variability \
  --base-split data_splits/legacy/oct2025_train_test_split.txt \
  --inner-seeds 0 1 2 3 4 5 6 7 8 9 \
  --kl-config configs/drosophila.yaml \
  --behavior genitalia_extension \
  --features-dir features \
  --seed 0 \
  --focal-alpha 0.67 \
  --keypoint-noise-std 0.01 \
  --execute \
  --out-dir results/split_variability/ge_fixed_test
```

Example: multiple outer splits plus multiple validation seeds

```bash
kinelearn-split-variability \
  --video-list video_lists/2025_jul_aug_with_ground_truth.yaml \
  --outer-seeds 0 1 2 3 4 \
  --inner-seeds 0 1 2 3 \
  --test-fraction 0.2 \
  --kl-config configs/drosophila.yaml \
  --behavior genitalia_extension \
  --features-dir features \
  --seed 0 \
  --focal-alpha 0.67 \
  --keypoint-noise-std 0.01 \
  --execute \
  --out-dir results/split_variability/ge_nested
```

This command writes:
- `experiment_config.yml` with the sweep settings
- `experiment_plan.csv` with one row per planned run
- generated split files under `<sweep_dir>/splits/`
- per-run training outputs under `<sweep_dir>/runs/<outer_id>/inner_seed<seed>/`
- `results_summary.csv` with one row per completed or resumed training run and the captured test metrics

Practical notes:
- Omit `--execute` to do a dry run and inspect the planned commands first.
- If `--out-dir` is omitted, the default sweep directory is `results/split_variability/<timestamp>/`.
- To resume an interrupted sweep, point `--resume` at the existing sweep directory. A run is treated as complete only if its run directory contains `train_manifest.yml`; missing or partial run directories are treated as pending, and partial managed run directories are removed before rerunning.
- Resume mode reuses the existing `experiment_plan.csv` and split files instead of regenerating them.
- Use fixed training hyperparameters while measuring split sensitivity; otherwise hyperparameter changes and split effects get mixed together.
- For historical parity/reproduction work, the old October 2025 split definitions are preserved under `data_splits/legacy/`.

Example: inspect an interrupted sweep without launching training

```bash
kinelearn-split-variability \
  --resume results/split_variability/ge_nested
```

Example: resume only the missing or incomplete runs

```bash
kinelearn-split-variability \
  --resume results/split_variability/ge_nested \
  --execute
```

---
## 📚 Batch-Evaluating Split Sweeps

The `kinelearn-batch-eval-splits` command runs `kinelearn-eval` across the manifests produced by a split-variability sweep and aggregates the resulting metrics into one table.

It is intended for the common follow-up step after `kinelearn-split-variability --execute`, where you want to compare validation or test performance across many train/test and train/validation split choices.

It performs:
1. **Resolving the sweep source** — accepts a split-variability output directory, `results_summary.csv`, or `experiment_plan.csv`.
2. **Finding manifests for each run** — uses `manifest_path` when present, or infers manifests from saved split metadata.
3. **Running evaluation per run** — invokes `kinelearn-eval` with consistent subset, threshold, and reporting settings.
4. **Aggregating metrics** — writes one combined CSV summarizing metrics across the whole sweep.

Example command:

```bash
kinelearn-batch-eval-splits \
  results/split_variability/ge_fixed_test \
  --subset val \
  --threshold 0.6 \
  --level frame \
  --out-dir results/split_variability_evals/ge_fixed_test_val
```

Another example using a specific sweep table:

```bash
kinelearn-batch-eval-splits \
  results/split_variability/ge_nested/results_summary.csv \
  --subset test \
  --threshold 0.6 \
  --level both
```

Optional CLI arguments:
- `--subset train|val|test` to choose which subset to evaluate for each run (default: `val`)
- `--threshold` to change the frame-level decision threshold passed through to `kinelearn-eval`
- `--level frame|episode|both` to request frame-level metrics, episode-level metrics, or both
- `--episode-min-frames` to control the minimum predicted episode length
- `--episode-max-gap` to control the allowed internal gap inside a predicted episode
- `--episode-overlap-threshold` to control predicted/ground-truth episode matching
- `--batch-size` to override evaluation batch size
- `--out-dir` to choose the output directory

This will write:
- `results/split_variability_evals/<timestamp>/batch_eval_config.yml`
- `results/split_variability_evals/<timestamp>/batch_eval_summary.csv`
- per-run evaluation outputs under `results/split_variability_evals/<timestamp>/runs/<outer_id>/inner_seed<seed>/`

Practical notes:
- If a sweep already recorded `manifest_path` values, the command uses them directly.
- If manifests are missing from the sweep table, the command attempts to infer them by matching `split_path` and `val_split_path`.
- Failed or unresolved runs are kept in the summary CSV with an error field instead of being silently dropped.

---
## 📊 Evaluating Predictions

The `kinelearn-eval` command evaluates one or more trained single-behavior models from their `train_manifest.yml` files.

It performs:
1. **Loading manifests and weights** — reads one or more training manifests and resolves the saved model weights for each behavior.
2. **Loading windowed artifacts** — opens the chosen subset's memmaps and index arrays from the training run directory.
3. **Reconstructing frame-level predictions** — runs inference over windows and averages overlapping window probabilities back onto frames.
4. **Computing metrics** — reports frame-level metrics, episode-level metrics, or both, depending on the selected evaluation mode.
5. **Saving evaluation outputs** — writes an evaluation summary, per-behavior metrics table, frame-level predictions table, and episode error table when episode-level reporting is enabled.

### Example commands

Single behavior:

```bash
kinelearn-eval \
  --manifest results/genitalia_extension/20260326_131743/train_manifest.yml \
  --threshold 0.6
```

Multiple behaviors together:

```bash
kinelearn-eval \
  --manifest results/back_leg_together/20260326_140102/train_manifest.yml \
  --manifest results/genitalia_extension/20260326_131743/train_manifest.yml
```

Episode-level reporting:

```bash
kinelearn-eval \
  --manifest results/genitalia_extension/20260326_131743/train_manifest.yml \
  --threshold 0.6 \
  --level both
```

Optional CLI arguments:
- `--subset train|val|test` to choose which split to evaluate (default: `test`)
- `--threshold` to change the frame-level decision threshold (default: `0.5`)
- `--level frame|episode|both` to switch between frame-level and episode-level reporting
- `--episode-min-frames` to control the minimum predicted episode length (default: `16`)
- `--episode-max-gap` to control the allowed internal gap inside a predicted episode (default: `3`)
- `--episode-overlap-threshold` to control predicted/ground-truth episode matching (default: `0.2`)
- `--batch-size` to override the evaluation batch size
- `--out` to choose the evaluation output directory

This will write:
- `results/evaluations/<timestamp>/eval_summary.yml`
- `results/evaluations/<timestamp>/per_behavior_metrics.csv`
- `results/evaluations/<timestamp>/frame_predictions.parquet`
- `results/evaluations/<timestamp>/episode_errors.csv` when `--level` is `episode` or `both`

Current scope notes:
- Episode-level reporting uses thresholded frame predictions to build bouts with a minimum length and a small allowed internal gap.
- For multi-model evaluation, provide at most one manifest per behavior.
- Manifests in the same evaluation run must share the same project/split/window settings.

## 🧮 Creating Ensembles

When you already know exactly which member runs you want, use `kinelearn-create-ensemble` to write the ensemble definition directly.

The `kinelearn-create-ensemble` command writes a lightweight `ensemble_manifest.yml` that groups multiple compatible single-behavior training runs into one reusable inference source.

This first implementation is intentionally narrow:
- Ensembles are behavior-specific.
- Members must be compatible for inference: same behavior, feature columns, window settings, feature-selection signature, and inference-relevant preprocessing.
- Aggregation is mean probability only.
- Ensemble manifests are for inference today; `kinelearn-eval` still expects train manifests.

Example command:

```bash
kinelearn-create-ensemble \
  --manifest results/genitalia_extension/run_a/train_manifest.yml \
  --manifest results/genitalia_extension/run_b/train_manifest.yml \
  --manifest results/genitalia_extension/run_c/train_manifest.yml \
  --name ge_split_variability_mean \
  --out-dir results/ensembles/genitalia_extension/ge_split_variability_mean
```

This writes:
- `results/ensembles/<behavior>/<name_or_timestamp>/ensemble_manifest.yml`

Practical notes:
- The ensemble manifest records the shared inference signature plus the member `train_manifest.yml` paths.
- This works well with split-variability outputs because you can choose a subset of completed runs and make that selection explicit in one manifest.
- If member manifests become incompatible later, KineLearn will fail loudly and ask you to recreate the ensemble manifest from the current runs.

---
## 🎯 Selecting Ensemble Members

The `kinelearn-select-ensemble` command helps you choose ensemble members from compatible training runs using validation performance rather than test performance.

This is the intended workflow when you have many candidate runs from `kinelearn-split-variability` and want a reproducible way to identify a stable band of good models before writing a deployable ensemble manifest.

The first implementation is intentionally simple:
- Inputs can mix one or more split-variability sweep outputs and direct `train_manifest.yml` paths.
- All candidates must match in behavior, feature/window signature, and training recipe; split metadata may differ.
- Candidates are scored on the validation subset only.
- Selection keeps runs within a configurable score band of the best validation performer, then lightly prefers diversity across outer splits when that provenance exists.
- The result is a normal `ensemble_manifest.yml`, so it stays compatible with the current ensemble inference flow.

Example command:

```bash
kinelearn-select-ensemble \
  --source results/split_variability/ge_fixed_test \
  --source results/split_variability/ge_nested \
  --metric frame_f1 \
  --threshold 0.6 \
  --min-score 0.55 \
  --band-tolerance 0.03 \
  --max-members 5 \
  --name ge_validation_band \
  --out-dir results/ensembles/genitalia_extension/ge_validation_band
```

This writes:
- `results/ensembles/<behavior>/selection_<timestamp>/ensemble_manifest.yml`
- `results/ensembles/<behavior>/selection_<timestamp>/candidate_scores.csv`
- `results/ensembles/<behavior>/selection_<timestamp>/selection_summary.yml`

Practical notes:
- `--source` accepts a split-variability directory, `results_summary.csv`, or `experiment_plan.csv`.
- `--manifest` can be used to add hand-picked direct runs into the candidate pool.
- Use validation metrics for selection; keep test-set reporting separate.
- The current scoring metrics are `frame_f1`, `frame_precision`, `frame_recall`, `episode_f1`, `episode_precision`, and `episode_recall`.
- For archived runs whose memmaps were pruned during archiving, restore the run artifacts first with `kinelearn-restore-run-artifacts`.

---
## 🔮 Running Inference on New Videos

The `kinelearn-predict` command applies one or more trained single-behavior models to arbitrary `frame_features_*.parquet` files. Each `--manifest` can point either to a single-model `train_manifest.yml` or to an `ensemble_manifest.yml` created with `kinelearn-create-ensemble`. This is the standalone inference path for videos that were processed with `kinelearn-calc` but are not part of the original train/val/test splits saved inside a training manifest.

It performs:
1. **Loading manifests and weights** — reads one or more train or ensemble manifests and resolves the saved model weights for each behavior source.
2. **Loading arbitrary feature files** — reads `frame_features_*.parquet` from a features directory, selected either by full stem list or by a video-list YAML.
3. **Aligning feature columns** — reorders and filters the input features to match the exact feature columns recorded in each training manifest.
4. **Running model inference** — windows the frame-level features, runs the trained model, and reconstructs overlapping-window probabilities back onto frames.
5. **Saving prediction outputs** — writes frame-level probability tables, optional thresholded predictions, and optional predicted bouts when a threshold is supplied.

### Example commands

Single behavior:

```bash
kinelearn-predict \
  --manifest results/genitalia_extension/20260328_142539/train_manifest.yml \
  --features-dir features \
  --stems output_video_20250730_181758_cropped_wheel_20250730_181758 \
          output_video_20250708_163744_cropped_wheel_20250708_163744 \
  --threshold 0.6 \
  --write-csv \
  --out results/inference/smoke_test_ge
```

Mixed ensemble + single-model inference:

```bash
kinelearn-predict \
  --manifest results/ensembles/genitalia_extension/ge_split_variability_mean/ensemble_manifest.yml \
  --manifest results/back_leg_together/<timestamp>/train_manifest.yml \
  --features-dir features \
  --video-list video_lists/new_videos.yaml \
  --threshold 0.6 \
  --write-csv \
  --out results/inference/new_videos
```

Ensemble-only inference:

```bash
kinelearn-predict \
  --manifest results/ensembles/genitalia_extension/ge_split_variability_mean/ensemble_manifest.yml \
  --features-dir features \
  --video-list video_lists/new_videos.yaml \
  --threshold 0.6 \
  --out results/inference/ge_ensemble_new_videos
```

Multiple behaviors together:

```bash
kinelearn-predict \
  --manifest results/back_leg_together/<timestamp>/train_manifest.yml \
  --manifest results/genitalia_extension/<timestamp>/train_manifest.yml \
  --features-dir features \
  --video-list video_lists/new_videos.yaml \
  --threshold 0.6 \
  --write-csv \
  --out results/inference/new_videos
```

Per-video outputs:

```bash
kinelearn-predict \
  --manifest results/genitalia_extension/<timestamp>/train_manifest.yml \
  --features-dir features \
  --video-list video_lists/new_videos.yaml \
  --threshold 0.6 \
  --write-csv \
  --output-mode per-video \
  --out results/inference/new_videos_per_video
```

Optional CLI arguments:
- `--features-dir` to read features from a directory other than `features/`
- `--video-list` to select stems from a YAML list of video paths
- `--stems` to pass one or more feature stems directly
- `--threshold` to add thresholded frame predictions and predicted bouts
- `--episode-min-frames` to control the minimum predicted bout length (default: `16`)
- `--episode-max-gap` to control the allowed internal gap inside a predicted bout (default: `3`)
- `--batch-size` to override the windowed inference batch size
- `--write-csv` to export a CSV copy of the frame-level predictions in addition to Parquet
- `--output-mode merged|per-video|both` to choose whether outputs are written as one merged table, one directory per video, or both (default: `merged`)
- `--out` to choose the inference output directory

This will write:
- `results/inference/<timestamp>/frame_predictions.parquet`
- `results/inference/<timestamp>/frame_predictions.csv` when `--write-csv` is provided
- `results/inference/<timestamp>/predicted_bouts.csv` when `--threshold` is provided
- `results/inference/<timestamp>/predict_summary.yml`

When `--output-mode per-video` or `--output-mode both` is used, KineLearn also writes per-video outputs under:
- `results/inference/<timestamp>/videos/<stem>/frame_predictions.parquet`
- `results/inference/<timestamp>/videos/<stem>/frame_predictions.csv` when `--write-csv` is provided
- `results/inference/<timestamp>/videos/<stem>/predicted_bouts.csv` when `--threshold` is provided

Practical notes:
- `kinelearn-predict` expects that `kinelearn-calc` has already been run on the target videos so that `frame_features_*.parquet` files exist.
- Unlike `kinelearn-eval`, this command does not require the videos to belong to the train/val/test subsets stored inside the training manifest.
- For multi-behavior inference, provide at most one manifest per behavior, whether that source is a single train manifest or an ensemble manifest.
- The feature files must contain the columns expected by the chosen manifest(s); the command will fail loudly if required feature columns are missing.

---

## 🎨 Visualizing Behavioral Dynamics

The first built-in visualization utility in KineLearn is `kinelearn-plot-timeline`, which generates per-video timeline plots from frame-level prediction tables. This is meant for general model-output inspection rather than experiment-specific downstream analysis.

It works with:
- `frame_predictions.parquet` written by `kinelearn-eval`
- `frame_predictions.parquet` written by `kinelearn-predict`
- the corresponding CSV files if you prefer to work from CSV

It performs:
1. **Loading prediction tables** — reads a `frame_predictions.parquet`/CSV file directly, or resolves one from an inference/evaluation output directory.
2. **Selecting videos and behaviors** — optionally filters to chosen stems or behaviors.
3. **Plotting behavior timelines** — draws one subplot per behavior, with probability traces over frames or seconds.
4. **Showing prediction context** — overlays a threshold line when requested, shades predicted bouts when `pred_<behavior>` columns or a threshold are available, and shades true bouts when `true_<behavior>` columns are present.
5. **Writing one figure per video** — saves plot files under a per-video directory and records them in a plot summary YAML.

### Example commands

Plot inference results in frame units:

```bash
kinelearn-plot-timeline \
  results/inference/smoke_test_ge \
  --threshold 0.6 \
  --out results/plots/timeline/smoke_test_ge
```

Plot a subset of stems in seconds:

```bash
kinelearn-plot-timeline \
  results/evaluations/20260330_120000/frame_predictions.parquet \
  --stems output_video_20250730_181758_cropped_wheel_20250730_181758 \
          output_video_20250708_163744_cropped_wheel_20250708_163744 \
  --fps 60 \
  --threshold 0.6 \
  --format both \
  --out results/plots/timeline/example_eval
```

Optional CLI arguments:
- `--stems` to plot only selected video stems
- `--behaviors` to plot only selected behaviors
- `--fps` to convert the x-axis from frames to seconds
- `--threshold` to draw a threshold line and infer predicted bout shading when `pred_<behavior>` columns are absent
- `--format png|pdf|both` to control image format (default: `png`)
- `--width` to control figure width
- `--height-per-behavior` to control figure height per subplot
- `--out` to choose the output directory

This will write:
- `results/plots/timeline/<timestamp>/<stem>/timeline.png`
- `results/plots/timeline/<timestamp>/<stem>/timeline.pdf` when `--format` is `pdf` or `both`
- `results/plots/timeline/<timestamp>/plot_summary.yml`

Practical notes:
- `kinelearn-plot-timeline` is intended for general probability/bout inspection across videos.
- More experiment-specific visualization, such as stimulus-aligned PSTHs or cohort-level optogenetic summaries, is better handled in a downstream analysis package rather than inside KineLearn itself.

---
<a id="archiving-result-directories"></a>
## 🗄️ Archiving Result Directories

The `kinelearn-archive-results` command moves a KineLearn results subtree to long-term storage while intentionally pruning the largest cache files created during training.

It is designed for cases where you want to free local disk space but keep reproducibility-relevant artifacts such as:
- `train_manifest.yml`
- `best_model.weights.h5`
- `train_history.csv`
- evaluation outputs
- inference outputs
- split-variability plans and summaries
- timeline plot outputs
- `*_vids.npy`
- `*_starts.npy`

Files omitted from the archive:
- `*_features.fp32`
- `*_labels.u8`

These memmap files are deleted from the source during a real archive run and are not copied to the destination. The smaller `*_vids.npy` and `*_starts.npy` index arrays are preserved.

Conservative protections for incomplete work:
- Ordinary training run directories are treated as incomplete when they contain training artifacts such as checkpoints, memmaps, `train_history.csv`, `*_vids.npy`, or `*_starts.npy` but do not contain `train_manifest.yml`.
- Incomplete training run directories are skipped entirely. Their files are neither copied nor deleted.
- Split-variability sweep directories are inspected by comparing the number of planned runs in `experiment_plan.csv` with the number of completed runs recorded in `results_summary.csv`.
- If those counts do not match, the sweep is treated as unfinished and the whole sweep subtree is left in place. This is intentionally conservative so resume/debugging metadata is not partially removed.
- When there is ambiguity, `kinelearn-archive-results` prefers leaving files in the source over archiving or deleting them.

Dry-run and verbose output report:
- files that would be moved
- memmaps that would be omitted
- files and directories skipped because they appear incomplete or still in progress
- total bytes moved, omitted, and skipped

Example dry run:

```bash
kinelearn-archive-results \
  results \
  /mnt/lab_archive/kinelearn/results \
  --dry-run \
  --verbose
```

Example real archive:

```bash
kinelearn-archive-results \
  results/genitalia_extension/20260326_131743 \
  /mnt/lab_archive/kinelearn/results/genitalia_extension/20260326_131743
```

Practical notes:
- The command moves files rather than copying them.
- Destination directories are created as needed.
- Existing destination files are treated as collisions and cause the run to stop before any files are moved.
- Existing destination directories are allowed when there are no file collisions, which makes conservative resume/merge scenarios possible.
- Empty source directories are removed after a successful archive pass when possible.

---
## 🧰 Restoring Archived Run Artifacts

The `kinelearn-restore-run-artifacts` command rebuilds the cached windowed artifacts for completed runs from their saved `train_manifest.yml` files.

This is useful when archived runs kept their manifests, weights, and split metadata but intentionally omitted the large memmaps during `kinelearn-archive-results`, and you later want to:
- rerun `kinelearn-eval`
- score archived runs during `kinelearn-select-ensemble`
- inspect or reuse historical runs without retraining

Example: restore one archived run

```bash
kinelearn-restore-run-artifacts \
  --manifest /media/Synology4/Robert/kinelearn_experiments/genitalia_extension/20260328_142539/train_manifest.yml
```

Example: restore every run in an archived split-variability sweep

```bash
kinelearn-restore-run-artifacts \
  --source /media/Synology4/Robert/kinelearn_experiments/split_variability/ge_nested
```

Example: remove the restored memmaps again after evaluation/selection work is finished

```bash
kinelearn-restore-run-artifacts \
  --source /media/Synology4/Robert/kinelearn_experiments/split_variability/ge_nested \
  --teardown
```

Optional CLI arguments:
- `--features-dir` to override the feature directory recorded in each manifest
- `--subset train|val|test|all` to restore only selected subsets (default: `all`)
- `--overwrite` to rebuild artifacts even when restored files already exist next to the manifest
- `--teardown` to remove previously restored memmaps instead of rebuilding them
- `--remove-indexes` to also remove `*_vids.npy` and `*_starts.npy` when tearing down
- `--report-out` to choose where `restore_summary.yml` is written

Practical notes:
- Restored `*_features.fp32`, `*_labels.u8`, `*_vids.npy`, and `*_starts.npy` files are written alongside the manifest’s current directory using the recorded filenames.
- KineLearn now resolves stale absolute artifact and weight paths conservatively by falling back to files adjacent to the manifest when they exist, which helps relocated archived runs work again without rewriting manifests.
- This command rebuilds cached artifacts only; it does not retrain the model or change its saved weights.
- Teardown is conservative by default: it removes the bulky restored memmaps and keeps the smaller index arrays unless `--remove-indexes` is explicitly requested.

---
## 🧪 Running Tests

KineLearn includes a lightweight synthetic regression test suite covering split parsing, inference helper logic, per-video output writing, and timeline plotting.

Run the full suite with:

```bash
python3 -m unittest discover -s tests -v
```

These tests are designed to stay fast and avoid dependence on large real datasets, which makes them suitable for routine local checks during development.

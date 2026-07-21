# SFO-Software-Defect-Prediction
FSSFO-based computationally efficient wrapper feature selection for software defect prediction using NASA/PROMISE datasets and multiple classifiers.
[README.md](https://github.com/user-attachments/files/30240686/README.md)
# FSSFO for Software Defect Prediction

A computationally efficient wrapper-based feature selection framework based on the **Swift Flight Optimizer (SFO)** for software defect prediction.

This repository contains the implementation of **FSSFO**, the experimental pipeline, benchmark optimizers, machine-learning classifiers, NASA/PROMISE datasets, result-processing scripts, and convergence-analysis tools used in the study:

> **Computationally Efficient Wrapper-Based Feature Selection Model for Software Defect Prediction Using the Swift Flight Optimization Algorithm**

---

## Overview

Software defect prediction aims to identify fault-prone software modules before deployment so that testing and maintenance resources can be allocated more effectively. However, software defect datasets frequently contain redundant software metrics and strongly imbalanced class distributions.

FSSFO adapts the continuous Swift Flight Optimizer to the binary feature-selection problem. Each candidate solution is transformed into a binary feature mask using a sigmoid transfer function and a deterministic threshold. The selected feature subset is then evaluated using a wrapper classifier.

The repository supports experiments with:

- 12 NASA/PROMISE software defect datasets
- 4 machine-learning classifiers
- 14 feature-selection optimizers
- 10 independent runs
- A population size of 10
- 200 optimization iterations
- Accuracy, defective-class precision, recall, F1-score, RMSE, selected-feature count, computational time, ranking, and convergence analysis

---

## Proposed Method

The proposed method is called **FSSFO**, where:

- **FS** denotes feature selection.
- **SFO** denotes the Swift Flight Optimizer.

The original SFO operates in a continuous search space. In FSSFO, each continuous position is converted into a binary decision vector:

- `1` indicates that the corresponding software metric is selected.
- `0` indicates that the corresponding software metric is excluded.

A sigmoid transfer function followed by a threshold of `0.5` is used to generate the binary feature mask.

The wrapper fitness function is defined as:

```text
Fitness = 1 - Validation Accuracy
```

Lower fitness values indicate better candidate feature subsets.

---

## Compared Feature-Selection Algorithms

FSSFO is compared with 13 population-based metaheuristic optimizers:

| Abbreviation | Algorithm |
|---|---|
| FSSFO | Swift Flight Optimizer |
| FSPSO | Particle Swarm Optimization |
| FSGWO | Grey Wolf Optimizer |
| FSDE | Differential Evolution |
| FSGA | Genetic Algorithm |
| FSACOR | Ant Colony Optimization for Continuous Domains |
| FSWOA | Whale Optimization Algorithm |
| FSHHO | Harris Hawks Optimization |
| FSHBA | Honey Badger Algorithm |
| FSAGTO | Artificial Gorilla Troops Optimizer |
| FSMGO | Mountain Gazelle Optimizer |
| FSSeaHO | Sea-Horse Optimizer |
| FSCoatiOA | Coati Optimization Algorithm |
| FSSCSO | Sand Cat Swarm Optimization |

All algorithms are evaluated under the same population size, iteration budget, data partitions, preprocessing procedure, and random seeds.

---

## Classifiers

The wrapper evaluation uses four machine-learning classifiers:

- Gaussian Naive Bayes
- K-Nearest Neighbors
- Decision Tree
- Linear Discriminant Analysis

The defective class is treated as the positive class when computing precision, recall, and F1-score.

---

## Datasets

The experiments use the following NASA/PROMISE software defect datasets:

| Dataset | Modules | Predictors | Defective Modules | Defect Ratio |
|---|---:|---:|---:|---:|
| CM1 | 344 | 37 | 42 | 12.21% |
| JM1 | 9,593 | 21 | 1,759 | 18.34% |
| KC1 | 2,096 | 21 | 325 | 15.51% |
| KC3 | 200 | 39 | 36 | 18.00% |
| KC4 | 125 | 40 | 61 | 48.80% |
| MC1 | 9,277 | 38 | 68 | 0.73% |
| MC2 | 127 | 39 | 44 | 34.65% |
| PC1 | 759 | 37 | 61 | 8.04% |
| PC2 | 1,585 | 36 | 16 | 1.01% |
| PC3 | 1,125 | 37 | 140 | 12.44% |
| PC4 | 1,399 | 37 | 178 | 12.72% |
| PC5 | 17,186 | 39 | 516 | 3.00% |

> The datasets are publicly available through the NASA/PROMISE software engineering repositories. Users should cite the original dataset sources when using them.

---

## Repository Structure

```text
FSSFO-Software-Defect-Prediction/
│
├── algorithms/
│   ├── base_optimizer.py
│   ├── fitness.py
│   ├── registry.py
│   ├── sfo.py
│   └── transfer.py
│
├── classifiers/
│   └── models.py
│
├── datasets/
│   ├── CM1.csv
│   ├── JM1.csv
│   ├── KC1.csv
│   ├── KC3.csv
│   ├── KC4.csv
│   ├── MC1.csv
│   ├── MC2.csv
│   ├── PC1.csv
│   ├── PC2.csv
│   ├── PC3.csv
│   ├── PC4.csv
│   └── PC5.csv
│
├── experiments/
│   ├── run_all.py
│   ├── run_single_dataset.py
│   └── generate_convergence_plots.py
│
├── results/
│   ├── convergence/
│   ├── plots/
│   ├── Accuracy.csv
│   ├── Metrics.csv
│   ├── Friedman.csv
│   └── selected_features.csv
│
├── README.md
└── requirements.txt
```

The exact file names may be adjusted according to the final repository organization.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/USERNAME/FSSFO-Software-Defect-Prediction.git
cd FSSFO-Software-Defect-Prediction
```

### 2. Create a virtual environment

Using `venv`:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The experiments were developed using Python 3.11 and MEALPY 3.0.3.

---

## Running the Experiments

### Run the complete benchmark

```bash
python experiments/run_all.py
```

This command runs all combinations of:

- 12 datasets
- 4 classifiers
- 14 feature-selection algorithms
- 10 independent runs

The complete benchmark may require substantial computational time.

### Run a single dataset

```bash
python experiments/run_single_dataset.py --dataset PC2
```

### Run a specific classifier

```bash
python experiments/run_single_dataset.py --dataset PC2 --classifier KNN
```

### Run FSSFO only

```bash
python experiments/run_single_dataset.py --dataset PC2 --classifier KNN --algorithm SFO
```

The command-line options above should be adjusted if the final scripts use different argument names.

---

## Experimental Protocol

The main experimental settings are:

| Parameter | Value |
|---|---|
| Population size | 10 |
| Maximum iterations | 200 |
| Independent runs | 10 |
| Search bounds | `[-6, 6]` |
| Transfer function | Sigmoid |
| Binary threshold | `0.5` |
| Outer split | 70% training / 30% testing |
| Inner split | 80% optimization training / 20% validation |
| Stratified splitting | Yes |
| Optimization objective | Minimize validation error |

The test partition is not used during feature selection. After the best feature subset is selected, the classifier is retrained using the complete outer-training partition and evaluated on the held-out test partition.

---

## Evaluation Metrics

The repository reports:

- Classification accuracy
- Defective-class precision
- Defective-class recall
- Defective-class F1-score
- Root Mean Square Error
- Number of selected features
- Computational time
- Accuracy-based ranking
- Mean best-so-far convergence

Because several datasets are highly imbalanced, accuracy should not be interpreted alone. Defective-class recall and F1-score are especially important for evaluating the ability to identify fault-prone modules.

---

## Main Results

Across all 48 dataset-classifier combinations, FSSFO obtained:

| Metric | FSSFO Result |
|---|---:|
| Average accuracy | 84.66% |
| Defective-class precision | 43.98% |
| Defective-class recall | 31.40% |
| Defective-class F1-score | 34.23% |
| RMSE | 36.05% |
| Average selected features | 16.51 |
| Average computational time | 10.17 s |
| Mean accuracy rank | 8.08 |

FSSFO was:

- Second-fastest among the 14 evaluated methods
- Third in defective-class recall
- Fourth in defective-class F1-score
- Within 0.61 percentage points of the highest aggregate accuracy

The results indicate that FSSFO provides a competitive balance between predictive performance, minority-class recognition, feature reduction, and computational efficiency.

---

## Convergence Analysis

The convergence scripts calculate the mean best-so-far fitness over the 10 independent runs for each:

```text
Dataset + Classifier + Algorithm + Iteration
```

To generate the convergence figures:

```bash
python experiments/generate_convergence_plots.py
```

The generated figures are saved in:

```text
results/plots/
```

Lower fitness values indicate better feature subsets.

---

## Reproducibility

To reproduce the reported results:

1. Use the same dataset files.
2. Use Python 3.11.
3. Install the package versions listed in `requirements.txt`.
4. Use the experimental parameters reported above.
5. Preserve the random seeds used for the 10 independent runs.
6. Use stratified training, validation, and test splits.
7. Apply preprocessing using training data only.
8. Do not use the held-out test set during feature selection.

Generated result files should be retained so that the reported tables and convergence figures can be reconstructed without rerunning all experiments.

---


## Author

**Abbas Aqeel Kareem**  
Department of Cybersecurity Engineering Techniques  
Technical Engineering College of Artificial Intelligence  
Middle Technical University  
Baghdad, Iraq

---

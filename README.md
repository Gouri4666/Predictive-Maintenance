# Predictive Maintenance for CNC Machines Using Machine Learning

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4+-red.svg)](https://xgboost.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

A complete, production-grade **MCA Machine Learning Project** designed to predict catastrophic failures in Computer Numerical Control (CNC) machines within a 24-hour operational window.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description & CNC Filtering](#dataset-description--cnc-filtering)
3. [Selected Features & Target Variable](#selected-features--target-variable)
4. [Project Folder Structure](#project-folder-structure)
5. [End-to-End Workflow](#end-to-end-workflow)
6. [Notebook Descriptions](#notebook-descriptions)
7. [Preprocessing & Leakage Prevention](#preprocessing--leakage-prevention)
8. [Machine Learning Algorithms](#machine-learning-algorithms)
9. [Evaluation Metrics & Actual Results](#evaluation-metrics--actual-results)
10. [Model Selection & Industrial Engineering Insights](#model-selection--industrial-engineering-insights)
11. [How to Reproduce the Project](#how-to-reproduce-the-project)

---

## 1. Project Overview

Predictive maintenance in modern Industry 4.0 leverages industrial Internet-of-Things (IoT) sensor streams to anticipate equipment breakdowns before catastrophic failure occurs. In Computer Numerical Control (CNC) machining centers, unexpected breakdowns result in:
- Ruined raw workpieces and broken cutting tools
- Severe damage to precision spindle motor bearings
- Unplanned factory production downtime costing thousands of dollars per hour

### Project Objective:
Build a binary classification machine learning system that ingests CNC condition monitoring telemetry and predicts:
$$\text{failure\_within\_24h} \in \{0, 1\}$$
- **0 = Normal Operation**: No failure anticipated within the next 24 hours.
- **1 = Failure Imminent**: A failure event will occur within the next 24 hours.

---

## 2. Dataset Description & CNC Filtering

- **Raw Dataset**: `predictive_maintenance_v3.csv` (24,042 records across multi-machine fleet)
- **Fleet Composition**:
  - Centrifugal Pumps: 6,114 records (Excluded)
  - Robotic Arms: 6,003 records (Excluded)
  - Compressors: 5,988 records (Excluded)
  - **CNC Machines: 5,937 records (Retained for analysis)**

### Strict Isolation Rule:
Because each machine type operates under distinct mechanical dynamics and degradation physics, only CNC records are included:
```python
df_cnc = df[df["machine_type"] == "CNC"].copy()
```
The `machine_type` column is used strictly for filtering and is **never** used as an ML input feature.

---

## 3. Selected Features & Target Variable

After filtering CNC machine records, exactly **five operational sensor features** are used:

```python
FEATURES = [
    "vibration_rms",      # Spindle/structural vibration RMS (mm/s)
    "temperature_motor",  # Motor core temperature (°C)
    "current_phase_avg",  # Average electrical phase current (A)
    "pressure_level",     # Hydraulic/pneumatic fluid pressure (bar)
    "rpm"                 # Spindle rotational speed (RPM)
]

TARGET = "failure_within_24h" # Binary label (0 = Normal, 1 = Failure Imminent)
```

### Excluded Columns:
`machine_type`, `product_type`, `machine_id`, `timestamp`, `operating_mode`, `hours_since_maintenance`, `ambient_temp`, `rul_hours`, `failure_type`, `estimated_repair_cost`.

---

## 4. Project Folder Structure

The project is structured professionally and portably using relative paths:

```text
CNC_Predictive_Maintenance/
│
├── dataset/
│   ├── raw/
│   │   └── predictive_maintenance_v3.csv       # Original dataset (unchanged)
│   │
│   └── processed/
│       ├── cnc_selected_features.csv           # CNC records with 6 selected columns
│       ├── X_train.csv                         # Median-imputed training features
│       ├── X_test.csv                          # Median-imputed testing features
│       ├── X_train_scaled.csv                  # Standardized training features (for SVM)
│       ├── X_test_scaled.csv                   # Standardized testing features (for SVM)
│       ├── y_train.csv                         # Training ground truth labels
│       └── y_test.csv                          # Testing ground truth labels
│
├── notebooks/
│   ├── 01_data_visualization.ipynb             # Exploratory Data Analysis & EDA plots
│   ├── 02_preprocessing.ipynb                  # Validation, splitting, imputation & scaling
│   ├── 03_algorithm_comparison.ipynb           # RF, SVM, XGBoost baseline benchmark
│   └── 04_smote_algorithm_comparison.ipynb     # SMOTE class balancing & re-evaluation benchmark
│
├── reports/
│   ├── EDA_Report/
│   │   └── EDA_Report.md                       # Comprehensive EDA findings
│   ├── Preprocessing_Report/
│   │   └── Preprocessing_Report.md             # Preprocessing & leakage prevention report
│   └── Model_Comparison_Report/
│       ├── Model_Comparison_Report.md          # Baseline benchmark & engineering tradeoffs
│       └── SMOTE_Balancing_Report.md           # SMOTE class balancing analysis & results
│
├── images/
│   ├── eda/                                    # Target dist, histograms, boxplots, heatmaps
│   ├── preprocessing/                          # Train/test split, imputation, scaling plots
│   └── model_comparison/                       # Confusion matrix heatmaps, metric comparison & importance
│
├── results/
│   ├── metrics/
│   │   ├── model_comparison.csv                # Baseline model benchmark metrics
│   │   ├── smote_model_comparison.csv          # SMOTE model benchmark metrics
│   │   └── baseline_vs_smote_comparison.csv    # Comprehensive head-to-head comparison
│   ├── confusion_matrices/
│   │   ├── random_forest_confusion_matrix.csv
│   │   ├── svm_confusion_matrix.csv
│   │   ├── xgboost_confusion_matrix.csv
│   │   ├── rf_smote_confusion_matrix.csv
│   │   ├── svm_smote_confusion_matrix.csv
│   │   └── xgb_smote_confusion_matrix.csv
│   └── feature_importance/
│       ├── rf_feature_importance.csv           # Random Forest Gini feature importances
│       └── xgb_feature_importance.csv          # XGBoost split gain feature importances
│
├── models/
│   ├── imputer.pkl                             # SimpleImputer fitted on X_train
│   ├── scaler.pkl                              # StandardScaler fitted on X_train
│   ├── best_model/                             # Production deployment bundle
│   │   ├── champion_model.pkl                  # Champion XGBoost (SMOTE) model
│   │   ├── imputer.pkl                         # Production imputer
│   │   └── model_metadata.json                 # Model hyperparameters & benchmark metadata
│   ├── random_forest/
│   │   ├── random_forest_model.pkl             # Serialized baseline RF model
│   │   └── random_forest_smote_model.pkl       # Serialized SMOTE RF model
│   ├── svm/
│   │   ├── svm_model.pkl                       # Serialized baseline SVM model
│   │   └── svm_smote_model.pkl                 # Serialized SMOTE SVM model
│   └── xgboost/
│       ├── xgboost_model.pkl                   # Serialized baseline XGBoost model
│       └── xgboost_smote_model.pkl             # Serialized SMOTE XGBoost model
│
├── web_app/                                    # Full-Stack Flask Web Application
│   ├── app.py                                  # Application entrypoint & factory
│   ├── config.py                               # MySQL & ML model configuration
│   ├── requirements.txt                        # Web app dependencies
│   ├── models/
│   │   └── model_loader.py                     # Leak-free model & imputer loader
│   ├── database/
│   │   ├── db.py                               # PyMySQL connection manager
│   │   └── schema.sql                          # MySQL tables & initial seed data
│   ├── routes/
│   │   ├── auth.py                             # Operator authentication & session
│   │   ├── dashboard.py                        # KPI dashboard & summary metrics
│   │   ├── machines.py                         # CNC Machine Asset CRUD
│   │   ├── prediction.py                       # 5-sensor inference execution
│   │   ├── history.py                          # Searchable historical audit logs
│   │   ├── analytics.py                        # Chart.js telemetry visualization
│   │   └── reports.py                          # CSV export & printable PDF reports
│   ├── templates/                              # Jinja2 HTML5 Bootstrap 5 templates
│   └── static/                                 # CSS styles & JS interactions
│
├── predict.py                                  # Real-time CLI & batch inference pipeline
└── README.md                                   # Master project documentation
```

---

## 5. End-to-End Workflow

```text
predictive_maintenance_v3.csv
            ↓
     Filter CNC Machines (5,937 records)
            ↓
       Data Visualization (Notebook 01)
            ↓
      Feature Extraction (5 sensors + target)
            ↓
      Data Validation (0 duplicates, valid binary target)
            ↓
       Stratified Train/Test Split (80% / 20%, random_state=42)
            ↓
      Median Imputation (Fitted ONLY on X_train)
            ↓
    Standard Scaling (Fitted ONLY on X_train, for SVM only)
            ↓
 ┌──────────────────┬──────────────────┬──────────────────┐
 │  Random Forest   │       SVM        │     XGBoost      │
 │ (100 estimators) │  (RBF Kernel)    │ (Depth 4, LR .05)│
 └──────────────────┴──────────────────┴──────────────────┘
            ↓
       Model Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
            ↓
    Confusion Matrices (TN, FP, FN, TP)
            ↓
    Feature Importance (RF Gini, XGBoost Gain)
            ↓
     Model Comparison Table & Visualizations
            ↓
   Serialized Artifacts + Markdown Reports
```

---

## 6. Notebook Descriptions

The project contains **exactly three notebooks**:

### Notebook 1: `01_data_visualization.ipynb`
- Inspects the raw dataset and filters 5,937 CNC records.
- Analyzes class distribution (5,103 Normal [85.95%] vs 834 Failure [14.05%]).
- Quantifies missing values across the 5 operational features (< 4.1% per feature).
- Generates histograms with KDE and boxplots (retaining physical outliers).
- Analyzes sensor shifts between normal and failure operating states.
- Generates Pearson correlation heatmap.
- Verifies zero duplicate rows.

### Notebook 2: `02_preprocessing.ipynb`
- Filters CNC records and isolates strictly the 6 designated columns.
- Performs data validation checks.
- Executes stratified 80/20 train-test split (`random_state=42`, `stratify=y`).
- Fits `SimpleImputer(strategy='median')` strictly on `X_train` to prevent data leakage.
- Fits `StandardScaler()` strictly on `X_train_imputed` for distance-based SVM.
- Persists all processed CSV splits and preprocessor objects (`imputer.pkl`, `scaler.pkl`).

### Notebook 3: `03_algorithm_comparison.ipynb`
- Implements and trains all three algorithms on the identical CNC dataset and test partition.
- Evaluates Random Forest, SVM (on scaled features), and XGBoost.
- Computes Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices.
- Extracts feature importance for Random Forest and XGBoost (with scientific note on SVM).
- Investigates cost-sensitive `class_weight='balanced'`.
- Compares models objectively and exports trained models using `joblib`.

### Notebook 4: `04_smote_algorithm_comparison.ipynb`
- Implements **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the training distribution (1:1 class ratio).
- Enforces strict leakage prevention: SMOTE is fitted **strictly on training partitions**; test set remains authentic and untouched.
- Retrains Random Forest, SVM (on standardized SMOTE features), and XGBoost.
- Generates simple correlation-style heatmap confusion matrices for all SMOTE models.
- Builds a direct head-to-head comparison between **Baseline** and **SMOTE** strategies.
- Evaluates industrial engineering trade-offs (drastically slashing False Negatives at a modest cost in False Positives).
- Persists SMOTE-trained models into `models/`.

---

## 7. Preprocessing & Leakage Prevention

### Critical Leakage Safeguard:
Information from the test set must **never** inform the training pipeline. 
```python
# Imputer: fit ONLY on training data
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Scaler: fit ONLY on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)
```
- **Imputation Strategy**: Median imputation is chosen because median is robust to extreme sensor spikes.
- **Scaling Requirement**: SVM requires standardized features ($\mu = 0, \sigma = 1$) because its RBF kernel computes geometric Euclidean distances. Random Forest and XGBoost do not require scaling as tree splits are invariant to monotonic scaling.

---

## 8. Machine Learning Algorithms

All three algorithms were trained on the identical 4,749 training records and evaluated on the identical 1,188 test records:

1. **Random Forest Classifier**:
   ```python
   RandomForestClassifier(n_estimators=100, random_state=42)
   ```
2. **Support Vector Machine (SVM)**:
   ```python
   SVC(probability=True, random_state=42) # Evaluated on standardized features
   ```
3. **XGBoost Classifier**:
   ```python
   XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss")
   ```

---

## 9. Evaluation Metrics & Actual Results

### The Cost of False Negatives in Predictive Maintenance:
- **False Negative (FN)**: Model predicts No Failure (`0`), but machine fails within 24 hours. Result: Catastrophic tool breakage, damaged spindle bearings, ruined workpiece, unscheduled downtime ($5,000–$50,000+).
- **False Positive (FP)**: Model predicts Failure (`1`), but machine is healthy. Result: Technician performs 15-minute diagnostic inspection.

Therefore, **Recall on Class 1 (Failure)** and **F1-Score** are the primary decision metrics.

### Baseline Benchmark Results Table (Original Distribution):

| Metric | Random Forest | Support Vector Machine (SVM) | XGBoost Classifier |
|---|---|---|---|
| **Train Accuracy** | **1.0000** (100.0%) | 0.9511 (95.11%) | 0.9714 (97.14%) |
| **Train Recall** | **1.0000** (100.0%) | 0.7766 (77.66%) | 0.8966 (89.66%) |
| **Train F1-Score** | **1.0000** (100.0%) | 0.8170 | 0.8979 |
| **Train ROC-AUC** | **1.0000** (100.0%) | 0.9833 | 0.9933 |
| **Test Accuracy** | **0.9478** (94.78%) | 0.9352 (93.52%) | 0.9444 (94.44%) |
| **Test Precision** | **0.8144** (81.44%) | 0.8000 (80.00%) | 0.8061 (80.61%) |
| **Test Recall** | **0.8144** (81.44%) | 0.7186 (71.86%) | 0.7964 (79.64%) |
| **Test F1-Score** | **0.8144** | 0.7571 | 0.8012 |
| **Test ROC-AUC** | **0.9836** | 0.9692 | 0.9826 |
| **Training Time** | 1.47 s | 1.26 s | **0.25 s** (Fastest) |
| **Testing Time (1,188 samples)** | 36.77 ms | 112.12 ms | **5.00 ms** (Ultra-fast) |
| **Inference Latency** | 0.0310 ms / sample | 0.0944 ms / sample | **0.0042 ms / sample** |
| **Missed Failures (FN)** | **31** (Lowest) | 47 | 34 |

*Generated directly from model predictions on test partition (`results/metrics/model_comparison.csv`).*

### Head-to-Head Benchmark: Baseline vs. SMOTE (with Training Metrics & Timing):

| Model | Strategy | Train Acc | Train Rec | Train F1 | Test Acc | Test Prec | Test Rec | Test F1 | Test AUC | Train Time | Test Time | Latency / Sample | Missed FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Random Forest** | Baseline | 1.0000 | 1.0000 | 1.0000 | **0.9478** | **0.8144** | 0.8144 | **0.8144** | **0.9836** | 1.47 s | 36.77 ms | 0.0310 ms | 31 |
| **Random Forest** | **SMOTE** | 1.0000 | 1.0000 | 1.0000 | 0.9436 | 0.7717 | **0.8503** | 0.8091 | 0.9822 | 2.33 s | 31.43 ms | 0.0265 ms | **25** |
| **SVM (RBF)** | Baseline | 0.9511 | 0.7766 | 0.8170 | 0.9352 | **0.8000** | 0.7186 | 0.7571 | 0.9692 | 1.26 s | 112.12 ms | 0.0944 ms | 47 |
| **SVM (RBF)** | **SMOTE** | 0.9483 | 0.9635 | 0.9491 | 0.9251 | 0.6773 | **0.8922** | 0.7700 | 0.9653 | 5.14 s | 258.57 ms | 0.2177 ms | **18** |
| **XGBoost** | Baseline | 0.9714 | 0.8966 | 0.8979 | **0.9444** | **0.8061** | 0.7964 | 0.8012 | 0.9826 | **0.25 s** | **5.00 ms** | **0.0042 ms** | 34 |
| **XGBoost** | **SMOTE** | 0.9722 | 0.9838 | 0.9725 | 0.9419 | 0.7450 | **0.8922** | **0.8120** | 0.9826 | **0.30 s** | **4.01 ms** | **0.0034 ms** | **18** |

*Source: `results/metrics/baseline_vs_smote_comparison.csv`*

### Confusion Matrix Breakdown (Test Set: 1,188 samples):

| Model | Strategy | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Detection Rate (Recall) |
|---|---|---:|---:|---:|---:|---:|
| **Random Forest** | Baseline | 990 | 31 | 31 | 136 | 81.44% |
| **Random Forest** | **SMOTE** | 979 | 42 | **25** | 142 | **85.03%** |
| **SVM** | Baseline | 991 | 30 | 47 | 120 | 71.86% |
| **SVM** | **SMOTE** | 950 | 71 | **18** | 149 | **89.22%** |
| **XGBoost** | Baseline | 989 | 32 | 34 | 133 | 79.64% |
| **XGBoost** | **SMOTE** | 970 | 51 | **18** | 149 | **89.22%** |

### Feature Importance Ranking:

| Rank | Feature | RF Importance (Gini) | XGBoost Importance (Gain) | Physical Failure Mechanism |
|---:|---|---:|---:|---|
| 1 | `rpm` | **31.46%** | **32.10%** | Spindle speed variance and drive slip |
| 2 | `temperature_motor` | **26.16%** | **41.57%** | Motor thermal buildup and coil overheating |
| 3 | `vibration_rms` | **24.14%** | **17.10%** | Spindle bearing wear and machining chatter |
| 4 | `current_phase_avg` | **9.71%** | **5.14%** | Electrical phase current draw under binding torque |
| 5 | `pressure_level` | **8.52%** | **4.10%** | Hydraulic clamping / pneumatic pressure loss |

*Scientific Note on SVM: Standard RBF SVM does not produce linear weights or tree-style feature importance scores. In accordance with strict scientific honesty, feature importances are not fabricated for SVM.*

---

## 10. Model Selection & Production Deployment

### Champion Production Model: **XGBoost Classifier (Trained with SMOTE)**
- **Failure Class Recall**: **89.22%** — Catches **149 out of 167** breakdowns; slashes missed breakdowns to only 18 (-47.1%).
- **F1-Score**: **0.8120** — Superior harmonic balance between breakdown detection and false alarms.
- **Inference Latency**: **0.0034 ms / sample** (3.4 microseconds per sample; ~290,000 predictions/second).
- **Production Artifact**: Saved in [`models/best_model/champion_model.pkl`](models/best_model/champion_model.pkl).

---

## 11. Real-Time Production Inference (`predict.py`)

A standalone, production-ready inference pipeline is provided in [`predict.py`](predict.py):

```bash
# 1. Predict from terminal with real-time sensor parameters:
python predict.py --vibration 4.2 --temperature 78.5 --current 21.0 --pressure 36.0 --rpm 1750

# 2. Batch predict on an industrial sensor CSV dataset:
python predict.py --file dataset/processed/X_test.csv --output test_predictions.csv

# 3. Interactive self-test mode (tests both normal and degraded machine states):
python predict.py
```

---

## 12. How to Reproduce the Project

### Prerequisites:
- Python 3.10+ (tested on Python 3.12)
- Required packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `matplotlib`, `seaborn`, `joblib`

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn joblib
```

### Execution Steps:
1. Navigate to the `notebooks/` folder:
   ```bash
   cd CNC_Predictive_Maintenance/notebooks
   ```
2. Execute the notebooks sequentially:
   - `01_data_visualization.ipynb` (Generates all EDA plots into `../images/eda/`)
   - `02_preprocessing.ipynb` (Generates processed datasets into `../dataset/processed/`)
   - `03_algorithm_comparison.ipynb` (Trains baseline models into `../models/`)
   - `04_smote_algorithm_comparison.ipynb` (Applies SMOTE, re-evaluates models, and benchmarks tradeoffs)

All notebooks use portable relative paths and will run seamlessly on any operating system.

---

## 13. Web Application — Full-Stack Industrial Dashboard

A complete, production-grade web application is provided in [`web_app/`](web_app/). It integrates the trained machine learning pipeline with a MySQL database and an industrial dashboard interface.

### Key Capabilities:
1. **Operator Authentication**: Secure password-hashed login system with session management.
2. **Operations Dashboard**: Real-time KPI summary cards, failure distribution donut chart, risk breakdown chart, and recent telemetry feed.
3. **Machine Asset Management**: Add, view, edit, and inspect CNC machines across workshop bays.
4. **Failure Prediction Engine**: Enter the exact 5 sensor inputs (`vibration_rms`, `temperature_motor`, `current_phase_avg`, `pressure_level`, `rpm`), select algorithm (XGBoost, Random Forest, SVM), and obtain real-time failure probability and recommendations.
5. **Prediction History & Audit**: Searchable, filterable (by risk, outcome, machine), and paginated historical logs.
6. **Analytics & Fleet Telemetry**: Time-series trend lines, machine-wise breakdown charts, and fleet-wide sensor averages.
7. **Reports & Exports**: One-click raw CSV export and printable shift audit report.

### Quick Start: Running the Web Application

#### 1. Start MySQL / XAMPP:
Ensure MySQL is running (default host: `localhost`, port: `3306`, user: `root`, password: `""`).

#### 2. Install Web Application Dependencies:
```bash
cd CNC_Predictive_Maintenance/web_app
pip install -r requirements.txt
```

#### 3. Launch Flask Server:
```bash
python app.py
```
The database `cnc_maintenance_db` and all tables (`users`, `machines`, `predictions`) are **automatically created and seeded** on initial startup.

#### 4. Access in Web Browser:
Open [http://127.0.0.1:5000](http://127.0.0.1:5000)

**Default Demo Credentials:**
- **Username:** `admin` (or `admin@cnc.com`)
- **Password:** `admin123`


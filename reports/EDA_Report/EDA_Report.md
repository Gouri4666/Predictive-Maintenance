# Exploratory Data Analysis (EDA) Report
## Predictive Maintenance for CNC Machines Using Machine Learning

---

### 1. Executive Summary
This report documents the Exploratory Data Analysis (EDA) conducted on industrial predictive maintenance telemetry to predict impending failures in **Computer Numerical Control (CNC)** machines. 

The primary objective is to classify whether a CNC machine will experience a failure within the subsequent 24 hours (`failure_within_24h`):
- **Class 0**: Normal Operation (No failure within 24 hours)
- **Class 1**: Failure Imminent (Failure occurs within 24 hours)

The analysis is performed on historical sensor logs comprising rotational speed, thermal signatures, electrical draw, fluid pressures, and mechanical vibrations.

---

### 2. Dataset Overview & Equipment Filtering

The raw dataset (`predictive_maintenance_v3.csv`) captures multi-machine industrial telemetry across four equipment categories:

| Machine Category | Total Records | Share of Raw Fleet | Scope Decision |
|---|---:|---:|---|
| **Pump** | 6,114 | 25.43% | Excluded |
| **Robotic Arm** | 6,003 | 24.97% | Excluded |
| **Compressor** | 5,988 | 24.91% | Excluded |
| **CNC Machine** | **5,937** | **24.69%** | **Retained (In-Scope)** |
| **Total** | **24,042** | **100.00%** | |

#### Domain Justification for Equipment Filtering
Industrial equipment archetypes operate under fundamentally distinct physics, duty cycles, and degradation trajectories:
- A centrifugal pump suffers cavitation and seal failure.
- A robotic arm experiences kinematic joint backlash and servo overload.
- A compressor exhibits valve cycling fatigue and gas chamber blow-by.
- A **CNC milling machine** undergoes high-speed spindle bearing wear, tool chatter vibration, cutting thermal buildup, and hydraulic clamping pressure drops.

Mixing heterogeneous machine records would inject confounding noise and degrade predictive validity. Consequently, **only CNC records (5,937 observations)** were retained. The `machine_type` attribute was used strictly as an isolation filter and excluded from input feature spaces.

---

### 3. Target Variable Analysis (`failure_within_24h`)

The target variable is binary and exhibits moderate class imbalance typical of industrial manufacturing environments:

| Target Class | Description | Record Count | Proportion (%) |
|---|---|---:|---:|
| **0** | Normal Operation | 5,103 | 85.95% |
| **1** | Failure Imminent (< 24h) | 834 | 14.05% |
| **Total** | All CNC Records | **5,937** | **100.00%** |

- **Class Imbalance Ratio**: Approximately **6.12 to 1** (Normal to Failure).
- **Practical Implication**: If a trivial dummy model always predicted "Normal" (Class 0), it would yield an 85.95% raw accuracy while completely failing to detect every machine breakdown. Therefore, evaluation must focus on **Recall**, **Precision**, and **F1-Score** for Class 1.

![Target Distribution](../../images/eda/target_distribution.png)

---

### 4. Operational Feature Selection & Missing Value Audit

In accordance with project constraints, five operational sensor features were selected for condition monitoring:

| Feature Name | Engineering Metric | Physical Unit | Expected Sensor Dynamics |
|---|---|---|---|
| `vibration_rms` | Root Mean Square Vibration | mm/s | Structural chatter, bearing race spalling, spindle imbalance |
| `temperature_motor` | Motor Core Temperature | °C | Electrical coil overheating, cooling circuit breakdown |
| `current_phase_avg` | Average Phase Current | Amperes (A) | Spindle motor electrical load, mechanical binding resistance |
| `pressure_level` | System Pressure | bar | Pneumatic tool release, hydraulic chuck clamping |
| `rpm` | Spindle Rotational Speed | RPM | Drive speed stability, slip, or abnormal drag |

#### Missing Value Audit in CNC Telemetry:

| Feature | Missing Rows | Missing Rate (%) | Available Records | Missingness Status |
|---|---:|---:|---:|---|
| `vibration_rms` | 240 | 4.04% | 5,697 | Low (< 5%) |
| `pressure_level` | 212 | 3.57% | 5,725 | Low (< 5%) |
| `temperature_motor` | 189 | 3.18% | 5,748 | Low (< 5%) |
| `current_phase_avg` | 154 | 2.59% | 5,783 | Low (< 5%) |
| `rpm` | 124 | 2.09% | 5,813 | Low (< 5%) |
| `failure_within_24h` | 0 | 0.00% | 5,937 | Complete |

![Missing Values](../../images/eda/missing_values.png)

Sensor dropouts represent transient field telemetry latency or routine maintenance recalibrations. The missingness rate is modest (< 4.1% per sensor), making **median imputation** an optimal, leak-free strategy.

---

### 5. Feature Distributions & Statistical Profiles

Descriptive statistics across the 5,937 CNC records:

| Feature | Mean | Std Dev | Min | 25% | Median | 75% | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `vibration_rms` | 0.942 | 0.384 | 0.350 | 0.690 | 0.840 | 1.080 | 3.420 |
| `temperature_motor` | 56.49 | 11.23 | 32.10 | 48.20 | 54.80 | 63.10 | 108.50 |
| `current_phase_avg` | 12.38 | 2.85 | 5.20 | 10.40 | 12.10 | 14.10 | 24.80 |
| `pressure_level` | 5.12 | 1.15 | 2.10 | 4.30 | 5.00 | 5.80 | 9.40 |
| `rpm` | 915.20 | 142.60 | 480.00 | 820.00 | 902.00 | 995.00 | 1650.00 |

![Feature Histograms](../../images/eda/feature_histograms.png)
![Feature Boxplots](../../images/eda/feature_boxplots.png)

#### Outlier Policy Rationale
Boxplot visualization indicates right-skewed extreme values in `temperature_motor` (> 90°C), `vibration_rms` (> 2.0 mm/s), and `rpm`. In standard business analytics, extreme values are often trimmed. In **predictive maintenance**, however, extreme readings represent the physical reality of impending breakdown (overheating bearings, severe mechanical resonance). **Outliers were deliberately retained** to preserve true failure indicators.

---

### 6. Failure State vs. Normal State Analysis

Comparing sensor distributions between normal (`0`) and failure-imminent (`1`) operation reveals significant separation:

| Feature | Normal Mean (Class 0) | Failure Mean (Class 1) | Delta (%) | Physical Indication |
|---|---:|---:|---:|---|
| `vibration_rms` | 0.86 mm/s | 1.44 mm/s | **+67.4%** | Pronounced mechanical resonance and bearing wear |
| `temperature_motor` | 53.8°C | 72.9°C | **+35.5%** | Severe thermal dissipation degradation |
| `rpm` | 895.4 RPM | 1,036.1 RPM | **+15.7%** | Spindle speed instability / overspeed conditions |
| `current_phase_avg` | 12.1 A | 14.1 A | **+16.5%** | Increased motor resistance and torque demand |
| `pressure_level` | 5.08 bar | 5.36 bar | **+5.5%** | Hydraulic pressure fluctuations |

![Failure vs Features](../../images/eda/failure_vs_features.png)

Noticeable divergence in `vibration_rms` and `temperature_motor` establishes that failure conditions exhibit strong physical sensor signatures, proving high machine learnability.

---

### 7. Correlation Analysis

Pearson correlation matrix between operational features and the target variable `failure_within_24h`:

![Correlation Heatmap](../../images/eda/correlation_heatmap.png)

| Ranked Feature | Correlation with Target | Physical Association |
|---|---:|---|
| `temperature_motor` | **+0.521** | Primary thermal buildup indicator preceding breakdown |
| `vibration_rms` | **+0.468** | Mechanical vibration / bearing deterioration signature |
| `rpm` | **+0.312** | Spindle motor anomalies and excessive rotational velocity |
| `current_phase_avg` | **+0.218** | Electrical draw increases under mechanical strain |
| `pressure_level` | **+0.076** | Minor direct correlation with electrical/mechanical failure |

Multicollinearity between sensor features is modest (highest inter-feature correlation between `vibration_rms` and `temperature_motor` is ~0.38), indicating each sensor supplies complementary predictive signal.

---

### 8. Data Integrity & Duplicate Check

- **Full-Row Duplicate Records**: Exactly **0** duplicate rows found.
- **Feature-Only Duplicate Sets**: Exactly **0** identical feature vectors found.
- **Data Integrity Status**: Verified. No records required deduplication or removal.

---

### 9. EDA Conclusions & Next Steps

1. **Target Imbalance**: 14.05% failure rate necessitates stratified partitioning and evaluation metrics centered on Recall, Precision, and F1-score rather than raw accuracy.
2. **Missing Values**: Under 4.1% across all 5 features. Leak-free median imputation fitted strictly on the training partition will eliminate missingness.
3. **Signal Quality**: Motor temperature, mechanical vibration, and spindle RPM provide strong discriminative power for predicting failure within 24 hours.
4. **Readiness**: Proceed directly to Preprocessing (`02_preprocessing.ipynb`).

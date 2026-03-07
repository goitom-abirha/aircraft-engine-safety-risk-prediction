# Aircraft Engine Safety Risk Prediction

Predictive maintenance and aviation safety analytics system using deep learning and the **NASA C-MAPSS turbofan engine dataset**.

This project predicts **Remaining Useful Life (RUL)** of aircraft engines and converts predictions into **risk scores, alerts, and fleet-level maintenance priorities** to support operational decision-making.

---

# Key Features

- LSTM and GRU-based RUL prediction  
- Engine-level operational risk scoring  
- Early warning alert generation  
- Fleet risk ranking for maintenance prioritization  
- Aviation-style evaluation using the latest window per engine  

---

# Project Objectives

- Predict aircraft engine **Remaining Useful Life (RUL)**  
- Convert RUL predictions into **operational risk scores**  
- Classify engines into **High / Medium / Low risk levels**  
- Generate **early warning alerts** for critical engines  
- Rank engines by **fleet safety risk priority**

---

# Dataset

This project uses the **NASA C-MAPSS Turbofan Engine Degradation Simulation dataset**.

### Dataset Characteristics

- Multivariate time-series sensor data  
- Multiple operating conditions  
- Engine degradation simulation  

### Dataset Splits Used

| Dataset | Purpose |
|-------|-------|
| FD001 | Model training |
| FD004 | Multi-condition training |
| FD004 Test | Evaluation and fleet monitoring |

Each engine contains:

- **21 sensor measurements**
- **3 operational settings**
- **Time-series degradation cycles**

---

# Machine Learning Models

Two deep learning architectures were implemented.

### LSTM (Long Short-Term Memory)

Designed to capture long-term temporal dependencies in sensor data.

### GRU (Gated Recurrent Unit)

A lighter recurrent architecture with fewer parameters while maintaining strong performance.

Both models were trained using **sliding window time-series sequences**.

---

# Model Evaluation

Models were evaluated using two approaches.

### Window-Level Evaluation

Measures prediction accuracy across **all sliding windows**.

Metrics used:

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**

### Engine-Level Evaluation (Operational Scenario)

Evaluates predictions using the **latest window per engine**, reflecting how airlines make maintenance decisions.

---
# Project Pipeline

Aircraft Sensor Data
↓
Data Preprocessing
↓
Feature Engineering
↓
Sliding Window Sequence Generation
↓
LSTM / GRU Deep Learning Models
↓
Remaining Useful Life Prediction
↓
Risk Scoring
↓
Fleet Risk Ranking
↓
Maintenance Decision Supp

---

# System Architecture
NASA C-MAPSS Sensor Data
↓
Data Preparation
(cleaning, RUL labeling, EDA)
↓
Feature Engineering
(scaling, rolling features, sequence generation)
↓
Deep Learning Models
(LSTM / GRU)
↓
RUL Prediction
↓
Risk Scoring Engine
(High / Medium / Low risk)
↓
Early Warning Alerts
↓
Fleet Risk Ranking
↓
Maintenance Decision Support


---

# Notebooks

| Notebook | Description |
|--------|-------------|
| Notebook 1 | Data loading and exploratory data analysis |
| Notebook 2 | Feature engineering and sequence generation |
| Notebook 3 | Model training, evaluation, and visualization |
| Notebook 4 | Risk scoring and fleet-level safety analysis |

---

# Risk Scoring System

Predicted RUL values are converted into operational risk categories.

| Risk Level | Condition |
|-----------|-----------|
| High Risk | RUL < 20 cycles |
| Medium Risk | 20 ≤ RUL < 50 |
| Low Risk | RUL ≥ 50 |

Risk scores are also normalized between **0 and 1** to support fleet ranking.

---

# Outputs

The system produces several operational outputs:

- Engine **RUL predictions**
- **Risk scores**
- **Risk levels**
- **Early warning alerts**
- **Fleet risk ranking**

### Example Output Files

---

# Example Use Cases

This system can support:

- Predictive aircraft maintenance
- Fleet health monitoring
- Maintenance prioritization
- Safety risk analysis
- Early failure detection

---

# Technologies Used

- Python  
- NumPy  
- Pandas  
- TensorFlow / Keras  
- Scikit-learn  
- Matplotlib  
- Jupyter Notebook  

---

# Future Improvements

Possible extensions include:

- Real-time sensor data ingestion  
- Streamlit fleet monitoring dashboard  
- Model explainability (**SHAP**)  
- Integration with maintenance planning systems  

---

# Author

**Goitom Abirha**


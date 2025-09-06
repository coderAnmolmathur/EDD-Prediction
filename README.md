# 📦 EDD-Prediction for E-commerce Shipments

Predicting **Estimated Delivery Date (EDD)** for e-commerce shipments using Machine Learning.

Accurately predicting EDD helps logistics companies:  
✔️ Improve operational efficiency  
✔️ Provide reliable delivery estimates to customers  
✔️ Reduce delays and penalties  
✔️ Optimize courier allocation and route planning  

---

## 🔎 Problem Statement

Logistics is a booming sector with millions of daily shipments. Predicting the exact delivery time of an order is challenging yet crucial.  

**Target Variable:**  
`predicted_exact_sla` → number of days between shipment date and actual delivery date  

**Objective:**  
Predict EDD for the test dataset containing 3 weeks of shipment data.

---

## 📂 Dataset

Three files:  

| File           | Description                            | Shape               |
|----------------|----------------------------------------|------------------|
| train.csv      | Training dataset                        | (1,115,514, 29)  |
| test.csv       | Test dataset for prediction             | (285,317, 29)    |
| pincodes.csv   | Pincode-level geographic details       | Varies           |

**Key Fields:**  

| Field Name             | Description |
|------------------------|------------|
| id                     | Unique identifier for a shipment |
| order_placed_date      | Date when the order was placed |
| order_shipped_date     | Date when the order was shipped |
| order_delivered_date   | Date when the order was delivered |
| courier_partner_id     | Courier partner identification code |
| account_type_id        | Courier account type code |
| pickup_pin_code        | Pickup location pincode |
| drop_pin_code          | Drop location pincode |
| quantity               | Number of items in shipment |
| account_mode           | Mode used to ship the package |
| order_delivery_sla     | Number of days taken to deliver |

---

## 🛠️ Project Pipeline

**Visual Suggestion:** Insert a diagram showing:  
**Raw Data → Data Cleaning → Feature Engineering → Models → Hybrid Mini-Stack → Submission**

### 1️⃣ Data Cleaning
- Imputed missing numerical values (median/mean) and categorical values (mode/"Unknown")  
- Removed duplicates & ensured `order_delivered_date` ≥ `order_shipped_date`  
- Capped extreme SLA or distance outliers  
- Created flags: `zero_distance` (identical pickup/drop locations), `drop_missing_flag`  

### 2️⃣ Feature Engineering
- **Distance Features:** distance_km, distance_km_capped, distance_per_item, dist_div_by_courier_median  
- **Frequency Features:** pickup_pin_code_freq, drop_pin_code_freq, route_freq  
- **Categorical Encoding:** Label encoding for courier_partner_id, account_type_id, account_mode  
- **Cluster/Geospatial Features:** Optional pickup_cluster_id to group nearby locations  
- **Model-specific predictions:** catboost_only included for ensemble stacking  

### 3️⃣ Exploratory Data Analysis (EDA)
- SLA distribution: right-skewed (most shipments 3–5 days)  
- Positive correlation: distance ↗ SLA  
- Express shipments → faster SLA  
- Enterprise accounts generally receive faster delivery  
- Visualizations: histograms, boxplots, scatter plots, correlation heatmaps  

### 4️⃣ Model Development

- **Base Models:** Random Forest, LightGBM, XGBoost, CatBoost  
- **Stacking Model:** Meta-model (Ridge regression) combining base models, optimized for overall SLA prediction  
- **Hybrid Mini-Stack Ensemble:**  

  Designed to improve predictions for **long-tail SLA cases** (>7 days).  

  **How it works:**
  1. For normal SLA shipments (≤7 days), the **stacking model** predictions are used.  
  2. For long SLA shipments (>7 days), a **mini-ensemble** predicts SLA using:
     - **LightGBM** (n_estimators=500, learning_rate=0.05, num_leaves=64)  
     - **XGBoost** (num_boost_round=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8)  
     - **CatBoost** (iterations=500, learning_rate=0.05, depth=6)  
  3. Predictions from LGB, XGB, and CatBoost are **averaged** to form the final long-SLA prediction  

- Hyperparameter tuning was applied to all models for optimal performance.  
- The **Hybrid Mini-Stack** outperforms both individual base models and standard stacking, especially for long-tail SLA shipments.  

---

## 📊 Model Performance Summary

| Model                   | RMSE       | MAE       |
|-------------------------|------------|-----------|
| Linear Regression       | 1.3520     | 0.9621    |
| Random Forest           | 1.212975   | 0.772943  |
| XGBoost                 | 1.117944   | 0.740035  |
| LightGBM                | 1.089829   | 0.744950  |
| CatBoost                | 1.078713   | 0.731397  |
| Stacking (Meta-Model)   | 0.996205   | 0.642579  |
| Hybrid Mini-Stack       | 0.856455   | 0.607697  |

✅ **Hybrid Mini-Stack outperforms all models**, achieving **RMSE < 1 day** and **MAE ~0.608**  

**Visual Suggestion:** Include a bar chart of RMSE/MAE for all models in decreasing RMSE order  

---

## 🔍 Model Explainability
**Top Features:**  
- `distance_km`, `account_mode`, `route_freq`, `courier_partner_id`  

**SHAP Analysis:**  
- Long distances & rare routes ↑ SLA  
- Express & frequent routes ↓ SLA  

**Partial Dependence Plots:**  
- Showed effect of distance & route frequency on predicted SLA  

---

## 🏁 Final Submission
- Predictions saved as `submission.csv` (columns: `id`, `predicted_exact_sla`)  
- Google Colab notebook `EDD_Prediction.ipynb` contains end-to-end workflow  
- Trained models saved in `models/` folder  

---

## ⚡ Tech Stack
- **Programming:** Python (pandas, numpy, scikit-learn, matplotlib, seaborn)  
- **ML Libraries:** LightGBM, XGBoost, CatBoost, RandomForest  
- **Environment:** Google Colab / Jupyter Notebook  
- **Model Persistence:** joblib  

---

## 🚀 Key Outcomes
- Built full ML pipeline: data cleaning → feature engineering → model evaluation → submission  
- Developed **Hybrid Mini-Stack ensemble** outperforming traditional stacking and single models  
- Demonstrated model explainability using SHAP and feature importance  
- Business impact: Accurate EDD predictions improve logistics efficiency and customer satisfaction  

---

# EDD-Prediction
Predicting Estimated Delivery Date (EDD) for e-commerce shipments using machine learning.

# EDD-Prediction

## Project Overview
This project predicts the **Estimated Delivery Date (EDD)** for e-commerce shipments using machine learning. Accurate EDD prediction helps logistics companies:

- Improve operational efficiency  
- Provide reliable delivery estimates to customers  
- Reduce delays and penalties  
- Optimize courier allocation and route planning  

---

## Problem Statement
Logistics is a booming sector with millions of daily shipments. Predicting the exact delivery time of an order is challenging yet crucial.  

- **Target variable:** `predicted_exact_sla` – number of days between shipment date and actual delivery date.  
- **Objective:** Predict EDD for the test dataset (3 weeks’ shipment data).  

---

## Dataset
The project uses three datasets:  

1. `train.csv` – Data to train the model  
2. `test.csv` – Data for which predictions are required  
3. `pincodes.csv` – Geographic details of pincodes  

**Key features include:**  
- Shipment dates (`order_placed_date`, `order_shipped_date`)  
- Courier and account information (`courier_partner_id`, `account_type_id`, `account_mode`)  
- Quantity and route information  
- Geolocation features derived from pincodes  

---

## Project Structure



---

## 📦 Deliverables (Checklist)

### **1. Data Cleaning**
- Imputed missing numerical values with median or mean.  
- Filled missing categorical values with mode or "Unknown" category.  
- Removed duplicates and ensured `order_delivered_date >= order_shipped_date`.  
- Capped extreme SLA or distance outliers.  
- Created flags: `zero_distance` for identical pickup/drop locations, `drop_missing_flag` for missing drop geocoordinates.  

---

### **2. Feature Engineering**
- **Distance Features:** `distance_km`, `distance_km_capped`, `distance_per_item`, `dist_div_by_courier_median`.  
- **Frequency Features:** `pickup_pin_code_freq`, `drop_pin_code_freq`, `route_freq`.  
- **Categorical Encoding:** Label encoding for `courier_partner_id`, `account_type_id`, `account_mode`.  
- **Cluster/Geospatial Features:** Optional `pickup_cluster_id` to group nearby locations.  
- **Model-specific predictions:** `catboost_only` included for ensemble stacking.  

---

### **3. Exploratory Data Analysis (EDA)**
- SLA distribution is right-skewed (most shipments 3–5 days).  
- Positive correlation between distance and SLA.  
- Frequent routes and Express shipments reduce SLA.  
- Enterprise accounts generally receive faster delivery.  
- Visualizations include histograms, boxplots, scatter plots, and correlation heatmaps.  

---

### **4. Model Development**
- **Base Models:** Random Forest, LightGBM, XGBoost, CatBoost.  
- **Ensemble Model:** Stacking (Meta-Model using Ridge regression).  
- Hyperparameter tuning applied to improve performance.  

---

### **5. Model Evaluation**
- **Metrics:** RMSE (Root Mean Squared Error), MAE (Mean Absolute Error).  
- **Performance Summary:**  

| Model                    | RMSE      | MAE      |
|--------------------------|-----------|----------|
| Random Forest            | 1.212975  | 0.772943 |
| LightGBM                 | 1.089829  | 0.744950 |
| XGBoost                  | 1.117944  | 0.740035 |
| CatBoost                  | 1.078713  | 0.731397 |
| **Stacking (Meta-Model)**| 0.996205  | 0.642579 |

- Stacking ensemble outperforms individual models, achieving RMSE below 1 day and MAE around 0.64.  

---

### **6. Model Explainability**
- **Feature Importance:** Key features – `distance_km`, `account_mode`, `route_freq`, `courier_partner_id`, `pickup_pin_code_freq`.  
- **SHAP Analysis:** Confirms distance and low-frequency routes increase SLA; Express mode and frequent routes reduce SLA.  
- **Partial Dependence Plots:** Showed effect of distance and route frequency on predicted SLA.  

---

### **7. Final Submission**
- **Predicted SLA values:** `submission.csv` containing `id` and `predicted_exact_sla`.  


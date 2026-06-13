# Bank Customer Churn Prediction - ANN

## Overview
A binary classification model using an Artificial Neural Network (ANN) to predict whether a bank customer will churn (leave the bank), based on demographic and account information.

## Dataset
- **File:** `bank_customer_churn_prediction.csv`
- **Rows:** 10,000
- **Features (12 → 12 after preprocessing):** `credit_score`, `gender`, `age`, `tenure`, `balance`, `products_number`, `credit_card`, `active_member`, `estimated_salary`, `country_France`, `country_Germany`, `country_Spain`
- **Target:** `churn` (0 = stayed, 1 = churned)
- **Class balance:** ~79.6% stayed, ~20.4% churned (imbalanced)

## Data Preprocessing
- No missing values, no duplicates.
- `customer_id` dropped (non-predictive identifier).
- Categorical encoding:
  - `gender`: Male = 1, Female = 0
  - `country`: one-hot encoded into `country_France`, `country_Germany`, `country_Spain`
- Feature scaling: `StandardScaler` applied to all input features (cast to `float32`).
- Train-test split: 80/20.

## Model Architecture
```
Input(shape=(12,))
Dense(64, activation='relu')
Dropout(0.3)
Dense(32, activation='relu')
Dropout(0.3)
Dense(16, activation='relu')
Dropout(0.2)
Dense(1, activation='sigmoid')
```
- **Total Parameters:** 3,457
- **Optimizer:** Adam (learning_rate = 0.01)
- **Loss:** Binary Crossentropy
- **Metrics:** Accuracy

## Training
- **Epochs:** Up to 100 (with Early Stopping)
- **Batch Size:** 32
- **Early Stopping:** Monitors `val_loss`, patience = 15, restores best weights

## Evaluation Results
| Metric | Value |
|---|---|
| Training Loss | 0.3298 |
| Training Accuracy | 0.8680 |
| Test Loss | 0.3364 |
| Test Accuracy | **0.8660** |

### Classification Report (Test Set)
| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 (Stayed) | 0.87 | 0.98 | 0.92 | 1607 |
| 1 (Churned) | 0.82 | 0.40 | 0.54 | 393 |
| **Accuracy** | | | **0.87** | 2000 |
| Macro avg | 0.85 | 0.69 | 0.73 | 2000 |
| Weighted avg | 0.86 | 0.87 | 0.85 | 2000 |


## Requirements
```
pandas
numpy
matplotlib
scikit-learn
tensorflow
```

## How to Run
1. Place `bank_customer_churn_prediction.csv` in the same directory as the notebook.
2. Run all cells sequentially in Jupyter.

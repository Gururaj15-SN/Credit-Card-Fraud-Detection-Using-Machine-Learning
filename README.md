# Credit Card Fraud Detection Using Machine Learning

## Project Overview
This project uses machine learning to identify potentially fraudulent credit card transactions. It compares Logistic Regression and Random Forest classifiers using an imbalanced credit card transaction dataset.

## Objectives
- Detect fraudulent credit card transactions.
- Preprocess transaction data for machine learning.
- Address class imbalance using SMOTE on training data.
- Compare classification models using precision, recall, F1-score, and confusion matrices.
- Save the trained model for reuse.

## Dataset
- **Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records:** 284,807 transactions
- **Features:** 30 input features and 1 target column (`Class`)
- **Fraud cases:** 492
- **Class imbalance:** Approximately 0.173% of transactions are fraudulent.

The dataset file is not included in this repository. Download it from Kaggle and follow the notebook instructions to load it.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib and Seaborn
- Joblib
- Google Colab

## Methodology
1. Load and explore the dataset.
2. Check missing values, duplicates, and class distribution.
3. Split data into training and testing sets using stratification.
4. Standardize the `Time` and `Amount` features.
5. Apply SMOTE to the training data only.
6. Train Logistic Regression and Random Forest models.
7. Evaluate both models using classification reports and confusion matrices.
8. Analyze detected and missed fraud transactions.
9. Save the trained model, scaler, and feature list.
10. Test the saved model on unseen test transactions.

## Model Results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 97% | Approximately 100% |
| Fraud Precision | 0.06 | 0.85 |
| Fraud Recall | 0.92 | 0.84 |
| Fraud F1-score | 0.11 | 0.84 |

Random Forest achieved a better balance between fraud detection and false alarms in this experiment. Accuracy is rounded; the classification report provides the more informative metrics for this imbalanced dataset.

## Random Forest Business Impact
- Fraud transactions detected: 82 out of 98
- Fraud transactions missed: 16
- Legitimate transactions incorrectly flagged: 15
- Fraud detection rate (recall): 83.67%
- Value of detected fraud transactions: 6,813.48 dataset currency units
- Value of missed fraud transactions: 3,831.45 dataset currency units

These transaction values are illustrative exposure estimates, not guaranteed savings or losses.

## Saved Model Files
- `credit_card_fraud_model.pkl` — trained Random Forest model
- `credit_card_fraud_scaler.pkl` — fitted scaler for `Time` and `Amount`
- `credit_card_fraud_features.pkl` — input feature names and order

## Limitations
- The dataset is highly imbalanced.
- SMOTE and model parameters may affect performance.
- Evaluation uses a single stratified train/test split.
- Duplicate rows were identified but not removed in this experiment.
- The sample prediction test demonstrates model loading and inference, not production readiness.
- The model is an educational prototype and should not be used to make real financial decisions without further validation.

## Future Improvements
- Tune classification thresholds and model hyperparameters.
- Evaluate precision-recall curves and average precision.
- Validate on additional data and test for data drift.
- Develop a simple interface for transaction-risk predictions.

## Disclaimer
This project is for educational purposes only. It is not a production-grade fraud prevention system.

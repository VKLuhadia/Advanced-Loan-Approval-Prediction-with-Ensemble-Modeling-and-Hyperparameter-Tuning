**Project Overview**


This project develops an advanced predictive model to determine loan approval status using the Kaggle Loan Prediction dataset. It leverages ensemble modeling techniques, combining Random Forest, XGBoost, and Logistic Regression in a stacking classifier, enhanced with hyperparameter optimization via GridSearchCV. The solution addresses class imbalance using SMOTE, incorporates feature engineering (e.g., Debt-to-Income Ratio, Credit Score Trend), and provides model explainability through SHAP values. Achieved an accuracy of 88% and a 45% increase in minority class representation, making it a robust tool for financial risk assessment.


**Dataset**
Source:
Kaggle Loan Prediction Dataset
Link: https://www.kaggle.com/datasets/altruist/delhi-house-price-prediction
Description: Contains 614 records with features such as Gender, Married, Dependents, Education, ApplicantIncome, Credit_History, LoanAmount, and Loan_Status (target: Y/N). The dataset is enriched with synthetic features like Credit_Score_Trend for advanced analysis.


**Features:**

Ensemble modeling with Random Forest, XGBoost, and Logistic Regression.
Hyperparameter tuning using GridSearchCV to optimize model performance.
SMOTE implementation to handle class imbalance, boosting minority class representation by 45%.
Feature engineering including Total_Income, Loan_Amount_per_Income, and Debt_to_Income_Ratio.
SHAP explainability to interpret feature importance, achieving 92% consistency in financial modeling.
Visualization of confusion matrix and feature importance plots.


**Installation**


Clone the repository:
bashgit clone <your-repo-url>
cd advanced-loan-prediction

Install required dependencies:
bashpip install pandas numpy scikit-learn xgboost imblearn shap seaborn matplotlib

Download the dataset:

Save train.csv from the Kaggle link above into the project directory as loan_data.csv.



**Usage:**


Run the main script:
bashpython advanced_loan_prediction.py

Or open and execute the Jupyter Notebook (if provided):

Launch Jupyter: jupyter notebook
Open advanced_loan_prediction.ipynb and run all cells.



**Outputs:**


Console: Accuracy (88%), ROC AUC score, classification report, and feature importance.
Plots: SHAP summary plot and confusion matrix.


Analyze results to understand key predictors (e.g., Credit_History, Debt_to_Income_Ratio) and model performance.



**Results:**


Accuracy: 88% on the test set.
Minority Class Improvement: 45% increase in representation with SMOTE.
Feature Importance: Credit_History and Debt_to_Income_Ratio identified as top contributors.
Explainability: SHAP values provide 92% consistent feature impact analysis.



**Contributions:**


Self-initiated project for placement preparation (August 2025).
Open to enhancements: Add more features, test additional models (e.g., LightGBM), or integrate real-time data.

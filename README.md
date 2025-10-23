# Gradient_Boosting_Classifier-for-UCI_Credit_Approval

Overview

This repository contains a comprehensive, beginner-friendly IPython notebook that implements a Gradient Boosting Classifier for a binary classification task using the UCI Credit Approval dataset (ID=27). The project demonstrates data preprocessing, model training, evaluation, and visualization, comparing the Gradient Boosting Classifier against baseline models (Logistic Regression and Dummy Classifier). The notebook is designed for educational purposes, showcasing best practices in machine learning with clear documentation and robust error handling.
The Gradient Boosting Classifier is an ensemble algorithm that builds a strong predictive model by combining multiple weak learners (shallow decision trees). It optimizes a loss function (log loss) using gradient descent, capturing complex, non-linear patterns in the data. This project applies it to predict credit approval outcomes, providing insights into model performance, feature importance, and hyperparameter configurations.

Purpose
The project aims to:

Provide a clear, reproducible implementation of the Gradient Boosting Classifier.
Demonstrate data preprocessing, model training, evaluation, and interpretation using a real-world dataset.
Compare Gradient Boosting against simpler models (Logistic Regression, Dummy Classifier) to highlight its effectiveness.
Visualize model performance and feature importance for transparency.
Serve as an educational resource for students and machine learning enthusiasts.

Dataset
The UCI Credit Approval dataset (ID=27) is used, sourced from the UCI Machine Learning Repository. It contains:

Features: 15 attributes (mix of numerical and categorical, anonymized as A1–A15).
Target: Binary classification (approved: +, not approved: -).
Size: ~690 instances, with some missing values (handled via imputation).
Characteristics: Mild class imbalance, anonymized features to protect privacy.
Source: Automatically fetched using the ucimlrepo library.

Usage
To run the notebook, follow these steps:

Environment Setup:

Use Google Colab (recommended) or a local Jupyter environment.
Install required libraries (see Requirements below).
Clone this repository:git clone https://github.com/your-username/your-repo-name.git




Open the Notebook:

Upload credit_approval_classifier.ipynb to Google Colab or open it in Jupyter.
Ensure the dataset is accessible (loaded automatically via ucimlrepo).


Run the Notebook:

Execute cells sequentially to:
Install dependencies and import libraries.
Load and clean the UCI Credit Approval dataset.
Explore data (missing values, class balance, etc.).
Preprocess features (imputation, scaling, encoding).
Train and evaluate models.
Generate visualizations and parameter tables.




Customization:

Modify hyperparameters in Section 8.1 (e.g., n_estimators, learning_rate for Gradient Boosting).
Use a different dataset by updating Section 2 (e.g., UCI Iris or Kaggle Telco Churn).
Add metrics or visualizations (e.g., precision-recall curves) as needed.



Outputs
The notebook produces the following outputs:

Data Exploration:
Tables: Head, tail, missing values, duplicates, numerical statistics.
Plots: Class distribution, numerical feature boxplots.


Model Performance:
Metrics: Accuracy, precision, recall, F1-score, ROC-AUC for Gradient Boosting, Logistic Regression, and Dummy Classifier.
Summary Plot: Grouped bar plot comparing accuracy and ROC-AUC across models.
Confusion matrices for each model.


Cross-Validation: Mean and standard deviation of accuracy and ROC-AUC for Gradient Boosting across 5 folds.
Feature Importance:
Table and bar plot of the top 10 features for Gradient Boosting.


Model Parameters:
Tables listing hyperparameters for all models and preprocessing steps.


Additional Visualizations:
ROC Curves: Show discriminative power of each model (Gradient Boosting typically has highest AUC: ~0.87–0.92).
Prediction Probability Histograms: Show class separation (Gradient Boosting and Logistic Regression show two peaks, Dummy shows one).



Note on Visualizations: Residuals and Q-Q plots are not included, as they are regression-specific and not suitable for this classification task. Instead, ROC curves and probability histograms are used to evaluate model performance, as they are standard for assessing class separation in binary classification.
Requirements
The notebook requires the following Python libraries:
pip install ucimlrepo scikit-learn pandas numpy matplotlib seaborn


Python: 3.6 or higher.
Libraries:
ucimlrepo: To fetch the UCI dataset.
scikit-learn: For machine learning models and metrics.
pandas, numpy: For data manipulation.
matplotlib, seaborn: For visualizations.


Environment: Google Colab (recommended) or local Jupyter Notebook.

In Google Colab, the notebook installs ucimlrepo automatically. For local environments, install dependencies manually.
Repository Structure
your-repo-name/
├── credit_approval_classifier.ipynb  # Main notebook with all code
├── README.md                        # This file
└── LICENSE                          # (Optional) License file (e.g., MIT)

Notes

Error Handling: The notebook includes robust error handling for issues like missing values, single-class test sets, or ROC-AUC computation failures.
Why No Residuals/Q-Q Plots: Residuals and Q-Q plots are designed for regression tasks to analyze prediction errors or normality. For this classification task, ROC curves, probability histograms, and the summary plot (accuracy, ROC-AUC) are more appropriate, as they directly assess class separation and model performance.
Extensibility: To use a different dataset (e.g., Kaggle Telco Churn), modify Section 2:import kagglehub
from kagglehub import KaggleDatasetAdapter
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "blastchar/telco-customer-churn",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
target_col = "Churn"
X_raw = df.drop(columns=[target_col])
y_raw = df[target_col]


Improvements: Consider adding precision-recall curves for imbalanced datasets or tuning hyperparameters (e.g., n_estimators=200, learning_rate=0.05) for better performance.

License
This project is licensed under the MIT License (see LICENSE file, if included).
Contact
For questions or contributions, open an issue or contact [your-username] on GitHub.

Generated on October 24, 2025

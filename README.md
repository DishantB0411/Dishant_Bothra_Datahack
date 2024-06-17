# Dishant_Bothra_Datahack

Certainly! Here's a README.md template that you can use for your GitHub repository based on the provided code:

---

# Vaccine Prediction Model

This repository contains code for predicting the likelihood of individuals getting vaccinated against XYZ and seasonal influenza based on demographic, behavioral, and health-related features. The prediction is performed using machine learning models, primarily logistic regression, after extensive data preprocessing and feature engineering.

## Dataset

The dataset consists of two main files:
- `training_set_features.csv`: Contains features (independent variables) used for training.
- `training_set_labels.csv`: Contains labels (dependent variables - vaccination status) for training.

## Overview

The project involves several key steps:

1. **Data Preprocessing and EDA**
   - Loading and merging datasets.
   - Handling missing values.
   - Exploring correlations between features and target variables.
   - Creating new features from existing ones.

2. **Model Selection**
   - Evaluating various machine learning models:
     - Logistic Regression
     - Random Forest Classifier
     - Support Vector Machines (SVM)

3. **Model Evaluation and Hyperparameter Tuning**
   - Training models on the training set.
   - Evaluating performance using metrics like ROC-AUC score, confusion matrix, and classification report.
   - Hyperparameter tuning using RandomizedSearchCV to optimize model performance.

4. **Final Model Training**
   - Selecting the best performing model (Logistic Regression) after hyperparameter tuning.
   - Training the selected model on the entire training dataset.

5. **Prediction and Submission**
   - Loading the test dataset.
   - Preprocessing the test data using the same transformations as the training data.
   - Predicting probabilities of vaccination for the test data using the trained models.
   - Generating a submission file (`final_csv`) containing respondent IDs and predicted probabilities.

## Files Included

- `README.md`: Overview of the project and instructions.
- `dataset and all/`: Directory containing training and test datasets.
- `vaccine_prediction.ipynb`: Jupyter notebook containing the complete code for data preprocessing, model training, and prediction.
- `final_csv`: Submission file containing predicted probabilities for the test dataset.

## Requirements

- Python 3.x
- Libraries:
  - numpy
  - pandas
  - seaborn
  - matplotlib
  - scikit-learn

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/vaccine-prediction.git
   cd vaccine-prediction
   ```

2. Install dependencies (if not already installed):
   ```
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook `vaccine_prediction.ipynb` to execute the code step-by-step or view the results.

4. Modify parameters or models as needed for further experimentation.

## Authors

- [Dishant Bothra](https://github.com/DishantB0411)



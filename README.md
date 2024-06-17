# Vaccine Prediction Model

This repository contains the code for predicting the likelihood of individuals getting vaccinated against XYZ and seasonal influenza. The prediction is performed using machine learning models, primarily logistic regression, after extensive data preprocessing and feature engineering.

## Dataset

The dataset consists of two main files:
- `training_set_features.csv`: Contains features (independent variables) used for training.
- `training_set_labels.csv`: Contains labels (dependent variables - vaccination status) for training.

## Project Overview

### 1. Exploratory Data Analysis (EDA)
- **Loading and Merging Data**: The features and labels are loaded and merged into a single DataFrame for easier manipulation.
- **Initial Examination**: The data is examined to understand its structure, including the number of rows, columns, data types, and summary statistics.
- **Correlation Analysis**: Correlation between features and target variables (`xyz_vaccine` and `seasonal_vaccine`) is calculated and visualized to understand relationships.

### 2. Data Preprocessing
- **Handling Missing Values**: Missing values are identified and columns with more than 20% missing data are removed.
- **Feature Separation**: Numerical and categorical features are separated for individual preprocessing.
- **Imputation and Encoding**: 
  - Numerical features are imputed using the median strategy.
  - Categorical features are imputed using the most frequent strategy and then one-hot encoded.
- **Feature Engineering**: New features are created by combining existing ones to reduce dimensionality and improve model performance. For example:
  - `behavioral_precautions`: Sum of various behavioral features.
  - `household_members`: Sum of household children and adults.

### 3. Model Training and Evaluation
- **Model Selection**: Three models are evaluated for their performance:
  - Logistic Regression
  - Random Forest Classifier
  - Support Vector Machines (SVM)
- **Train-Test Split**: The data is split into training and testing sets (80%-20% split).
- **Model Training**: Each model is trained on the training data.
- **Performance Evaluation**:
  - Models are evaluated using the ROC-AUC score.
  - ROC curves are plotted for visual comparison.
  - Confusion matrix and classification report are generated to understand model performance in detail.

### 4. Hyperparameter Tuning
- **Randomized Search**: Hyperparameters of the Logistic Regression model are tuned using RandomizedSearchCV to find the best parameters.
- **Final Model Training**: The Logistic Regression model with the best parameters is retrained on the entire training dataset.

### 5. Prediction and Submission
- **Preprocessing Test Data**: The test dataset is preprocessed using the same steps as the training data.
- **Prediction**: Probabilities of vaccination for the test data are predicted using the trained models.
- **Submission File**: A submission file (`final_csv`) is generated containing respondent IDs and predicted probabilities.

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

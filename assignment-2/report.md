# 1. Project Introduction

In this project, we utilize the Lung Cancer Survey dataset from Kaggle to develop a classification model that predicts the likelihood of an individual having lung cancer. The model is based on various factors including gender, age, smoking habits, presence of yellow fingers, anxiety levels, peer pressure, chronic diseases, fatigue, allergies, wheezing, alcohol consumption, coughing, shortness of breath, swallowing difficulty, and chest pain.

# 2. Data Pre-Processing

We carry out the following data preprocessing steps to clean and validate the dataset:

- Identify and handle any missing values.
- Detect and remove duplicate entries.
- Convert all categorical variables into numerical representations.
- Standardize the column names for consistency.

All columns except (`GENDER`, `SMOKING`) are used as features for the model, as the correlation matrix (see Table 1) indicates that each feature has a significant relationship with the target variable.

| Feature                 | Correlation |
| :---------------------- | :---------- |
| GENDER                  | -0.053666   |
| AGE                     | 0.106305    |
| SMOKING                 | 0.034878    |
| YELLOW_FINGERS          | 0.189192    |
| ANXIETY                 | 0.144322    |
| PEER_PRESSURE           | 0.195086    |
| CHRONIC_DISEASE         | 0.143692    |
| FATIGUE                 | 0.160078    |
| ALLERGY                 | 0.333552    |
| WHEEZING                | 0.249054    |
| ALCOHOL_CONSUMING       | 0.294422    |
| COUGHING                | 0.253027    |
| SHORTNESS_OF_BREATH     | 0.064407    |
| SWALLOWING_DIFFICULTY   | 0.268940    |
| CHEST_PAIN              | 0.194856    |
*Table 1: Correlation between each feature and the target variable, LUNG_CANCER.*


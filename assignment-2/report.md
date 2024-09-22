# 1. Project Introduction

In this project, we utilize the Lung Cancer Survey dataset from Kaggle to develop a classification model that predicts the likelihood of an individual having lung cancer. The model is based on various factors including gender, age, smoking habits, presence of yellow fingers, anxiety levels, peer pressure, chronic diseases, fatigue, allergies, wheezing, alcohol consumption, coughing, shortness of breath, swallowing difficulty, and chest pain.

# 2. Data Pre-Processing

We carry out the following data preprocessing steps to clean and validate the dataset:

- Identify and handle any missing values.
- Detect and remove duplicate entries.
- Convert all categorical variables into numerical representations.
- Standardize the column names for consistency.

The following heatmap shows the correlation between all features:

![Correlation Heatmap](https://github.com/uy-seng/cs4372/blob/main/assignment-2/image/correlation_heatmap.png?raw=true)

From the heatmap we extract only the relationships between the features and the target variable (`LUNG_CANCER`).

All columns except (`GENDER`, `SMOKING`, `SHORTNESS_OF_BREATH`) are used as features for the model, as the correlation matrix (see Table 1) indicates that each feature has a significant relationship with the target variable. We remove features that are lower than *10%* as it does not contribute much to the model and increases the complexity size of the model.

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

# 3. Tree Model Building

We will employ four different models to evaluate and compare their classification performance and accuracy. Additionally, we will analyze the impact of various parameters and dataset splitting methods on each model's accuracy. The models we will use are as follows:

- Decision Tree Classifier
- Random Forest Classifier
- AdaBoost Classifier
- XGBoost Classifier

## 3.1. Decision Tree Classifier

We are evaluating the performance of different combinations of the hyperparameters `max_depth` and `criterion` for our model.

- Max Depth: The maximum depth of the tree that yields the highest accuracy.
- Criterion: The criterion used to measure the quality of a split that results in the best model performance.

The table below summarizes the accuracy achieved for each set of hyperparameter values.

| Max Depth | Criterion | Accuracy                |
| :-------- | :-------- | :---------------------- |
| 1         | gini      | 0.8554216867469879      |
| 1         | entropy   | 0.8554216867469879      |
| 2         | gini      | 0.8554216867469879      |
| 2         | entropy   | 0.8554216867469879      |
| 3         | gini      | 0.8313253012048193      |
| 3         | entropy   | 0.8433734939759037      |
| 4         | gini      | 0.8554216867469879      |
| 4         | entropy   | 0.8313253012048193      |
| 5         | gini      | 0.8554216867469879      |
| 5         | entropy   | 0.8433734939759037      |
| 6         | gini      | 0.8554216867469879      |
| 6         | entropy   | 0.8313253012048193      |
| 7         | gini      | 0.8554216867469879      |
| 7         | entropy   | 0.8313253012048193      |
| 8         | gini      | 0.8554216867469879      |
| 8         | entropy   | 0.8313253012048193      |
| 9         | gini      | 0.8554216867469879      |
| 9         | entropy   | 0.8313253012048193      |
| 10        | gini      | 0.8554216867469879      |
| 10        | entropy   | 0.8313253012048193      |

Based on the table, we can make some of the following observations:

- We notice that as the depth increases, the accuracy plateus meaning that depth 1 is enough for us to make predictions with enough accuracy. 
- The low number of depth indicates to us that the data is linearly separable.
- We notice that the combinations of different `criterions` and `max_depth` does not yield us any improved results. We will omit out the criterions for our parameter search in the subsequent experiments.

By systematically testing various combinations of these parameters, we aim to identify the optimal settings that maximize the classification accuracy of our model. The optimal settings for the hyperparameters is as follow:

```
Best accuracy: 0.8554216867469879
Best max_depth: 1
Best model report:
              precision    recall  f1-score   support

           0       0.00      0.00      0.00        12
           1       0.86      1.00      0.92        71

    accuracy                           0.86        83
   macro avg       0.43      0.50      0.46        83
weighted avg       0.73      0.86      0.79        83
```

The results for the decision tree trained with the optimal parameters are presented below:
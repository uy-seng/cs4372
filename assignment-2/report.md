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

*Figure 1: Correlation Heatmap between all features*

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

Another observation that we noticed about the dataset is that the distribution of lung cancer patient and healthy patient is severely imbalanced. (As seen in *Figure 2*)

![Data Distribution](https://github.com/uy-seng/cs4372/blob/main/assignment-2/image/dataset_distribution.png?raw=true)

*Figure 2: Dataset Distribution for each features*

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
| 1         | gini      | 0.5542168674698795      |
| 1         | entropy   | 0.5542168674698795      |
| 2         | gini      | 0.7228915662650602      |
| 2         | entropy   | 0.7228915662650602      |
| 3         | gini      | 0.6746987951807228      |
| 3         | entropy   | 0.6746987951807228      |
| 4         | gini      | 0.8072289156626506      |
| 4         | entropy   | 0.8072289156626506      |
| 5         | gini      | 0.8433734939759037      |
| 5         | entropy   | 0.8072289156626506      |
| 6         | gini      | 0.8433734939759037      |
| 6         | entropy   | 0.8072289156626506      |
| 7         | gini      | 0.8072289156626506      |
| 7         | entropy   | 0.8313253012048193      |
| 8         | gini      | 0.8072289156626506      |
| 8         | entropy   | 0.8433734939759037      |
| 9         | gini      | 0.8072289156626506      |
| 9         | entropy   | 0.8313253012048193      |
| 10        | gini      | 0.8072289156626506      |
| 10        | entropy   | 0.8313253012048193      |

*Table 2: The combinations of different max_depth and criterion and the accuracy we obtained.*

Based on the table, we can make some of the following observations:

- We notice that the combinations of different `criterions` and `max_depth` does not yield us any improved results. We will omit out the criterions for our parameter search in the subsequent experiments.

By systematically testing various combinations of these parameters, we aim to identify the optimal settings that maximize the classification accuracy of our model. The optimal settings for the hyperparameters is as follow:

```
Best accuracy: 0.8433734939759037
Best max_depth: 5
Best model report:
              precision    recall  f1-score   support

           0       0.47      0.75      0.58        12
           1       0.95      0.86      0.90        71

    accuracy                           0.84        83
   macro avg       0.71      0.80      0.74        83
weighted avg       0.88      0.84      0.86        83
```

The results for the decision tree trained with the optimal parameters are presented below:

![Decision Tree Visualization](https://raw.githubusercontent.com/uy-seng/cs4372/ca0e66f6d54dd6443cbb7910983864dacfaab541/assignment-2/image/decision_tree.svg)

*Figure 3: Visualization of Decision Tree*

From this model, we obtained an accuracy of `84%`. We further tried the same model on the same dataset, however instead of using the `holdout` method from the previous experiment, we used `cross-validation` method. The following results is obtained:

```
Best accuracy: 0.8733766233766234
Best max_depth: 7
Best model report:
              precision    recall  f1-score   support

           0       0.44      0.58      0.50        12
           1       0.93      0.87      0.90        71

    accuracy                           0.83        83
   macro avg       0.68      0.73      0.70        83
weighted avg       0.85      0.83      0.84        83
```

We noticed that the model generalizes method using the `cross-validation` method as it has increased in accuracy by `3%`.

<!--TODO: insert ROC curve here-->
*Figure 4: Decision Tree ROC*


The ROC curve shows that the model has an AUC of 0.80, indicating good overall performance in distinguishing between the positive and negative classes. The curve rises sharply initially, reaching a high true positive rate with a low false positive rate, but flattens out as the false positive rate increases. This suggests the model is effective at identifying true positives early on, but performance decreases as the threshold lowers. While the model performs better than random guessing (represented by the diagonal line), there is still room for improvement to reduce false positives.

<!-- TODO: insert precision recall curve here -->
*Figure 5: Decision Tree Precision-Recall curve*

The Precision-Recall curve shows that the model maintains high precision and recall across most thresholds, with a slight decrease in precision as recall increases. It exhibits excellent performance overall, as indicated by the high AUC-PR score of 0.97. However, there is a sharp drop in precision near maximum recall, suggesting that while the model can capture almost all positive cases, it does so at the cost of many false positives. This model is well-suited for scenarios where both precision and recall are important, but caution is needed when aiming for very high recall.

## 3.2. Random Forest Classifier

The plain decision tree has the drawback of being prone to overfitting and limited generalization capabilities, as it often captures noise in the data. In contrast, Random Forest is a more robust model as it aggregates the results of multiple decision trees, reducing overfitting and improving overall accuracy and stability. To leverage these advantages, we will use the same experimental parameters as the plain decision tree for the Random Forest classifier. Instead of using the `criterion` hyperparameter, we will experiment with the `max_leaf_nodes` parameter to find the optimal configuration for enhanced model performance.

The table below summarizes the accuracy achieved for each set of hyperparameter values.

| Max Depth | Max Leaf Nodes | Accuracy                |
| :-------- | :------------- | :---------------------- |
| 1         | 2              | 0.7831325301204819      |
| 1         | 3              | 0.7469879518072289      |
| 1         | 4              | 0.7710843373493976      |
| 1         | 5              | 0.7951807228915663      |
| 1         | 6              | 0.8072289156626506      |
| 1         | 7              | 0.7228915662650602      |
| 1         | 8              | 0.7951807228915663      |
| 1         | 9              | 0.8192771084337349      |
| 1         | 10             | 0.8072289156626506      |
| 2         | 2              | 0.7831325301204819      |
| 2         | 3              | 0.8433734939759037      |
| 2         | 4              | 0.8433734939759037      |
| 2         | 5              | 0.7951807228915663      |
| 2         | 6              | 0.8313253012048193      |
| 2         | 7              | 0.8313253012048193      |
| 2         | 8              | 0.8072289156626506      |
| 2         | 9              | 0.8554216867469879      |
| 2         | 10             | 0.8674698795180723      |
| 3         | 2              | 0.7831325301204819      |
| 3         | 3              | 0.8072289156626506      |
| 3         | 4              | 0.8313253012048193      |
| 3         | 5              | 0.8313253012048193      |
| 3         | 6              | 0.8674698795180723      |
| 3         | 7              | 0.8674698795180723      |
| 3         | 8              | 0.8554216867469879      |
| 3         | 9              | 0.8433734939759037      |
| 3         | 10             | 0.8433734939759037      |
| 4         | 2              | 0.7831325301204819      |
| 4         | 3              | 0.8433734939759037      |
| 4         | 4              | 0.8674698795180723      |
| 4         | 5              | 0.8433734939759037      |
| 4         | 6              | 0.8674698795180723      |
| 4         | 7              | 0.8674698795180723      |
| 4         | 8              | 0.8554216867469879      |
| 4         | 9              | 0.8795180722891566      |
| 4         | 10             | 0.8554216867469879      |
| 5         | 2              | 0.7831325301204819      |
| 5         | 3              | 0.8674698795180723      |
| 5         | 4              | 0.8554216867469879      |
| 5         | 5              | 0.8674698795180723      |
| 5         | 6              | 0.8554216867469879      |
| 5         | 7              | 0.8554216867469879      |
| 5         | 8              | 0.8674698795180723      |
| 5         | 9              | 0.8554216867469879      |
| 5         | 10             | 0.8554216867469879      |
| 6         | 2              | 0.7951807228915663      |
| 6         | 3              | 0.8554216867469879      |
| 6         | 4              | 0.8433734939759037      |
| 6         | 5              | 0.8674698795180723      |
| 6         | 6              | 0.8795180722891566      |
| 6         | 7              | 0.8795180722891566      |
| 6         | 8              | 0.8795180722891566      |
| 6         | 9              | 0.8795180722891566      |
| 6         | 10             | 0.8674698795180723      |
| 7         | 2              | 0.7349397590361446      |
| 7         | 3              | 0.8313253012048193      |
| 7         | 4              | 0.8554216867469879      |
| 7         | 5              | 0.8554216867469879      |
| 7         | 6              | 0.8554216867469879      |
| 7         | 7              | 0.8554216867469879      |
| 7         | 8              | 0.8554216867469879      |
| 7         | 9              | 0.8554216867469879      |
| 7         | 10             | 0.8674698795180723      |
| 8         | 2              | 0.7710843373493976      |
| 8         | 3              | 0.8433734939759037      |
| 8         | 4              | 0.8554216867469879      |
| 8         | 5              | 0.8674698795180723      |
| 8         | 6              | 0.8795180722891566      |
| 8         | 7              | 0.8795180722891566      |
| 8         | 8              | 0.8554216867469879      |
| 8         | 9              | 0.8674698795180723      |
| 8         | 10             | 0.8554216867469879      |
| 9         | 2              | 0.7951807228915663      |
| 9         | 3              | 0.8433734939759037      |
| 9         | 4              | 0.8313253012048193      |
| 9         | 5              | 0.8313253012048193      |
| 9         | 6              | 0.8674698795180723      |
| 9         | 7              | 0.8674698795180723      |
| 9         | 8              | 0.8674698795180723      |
| 9         | 9              | 0.8674698795180723      |
| 9         | 10             | 0.8554216867469879      |
| 10        | 2              | 0.7831325301204819      |
| 10        | 3              | 0.8433734939759037      |
| 10        | 4              | 0.8554216867469879      |
| 10        | 5              | 0.8554216867469879      |
| 10        | 6              | 0.8674698795180723      |
| 10        | 7              | 0.8554216867469879      |
| 10        | 8              | 0.8674698795180723      |
| 10        | 9              | 0.8795180722891566      |
| 10        | 10             | 0.8795180722891566      |

*Table 3: The combinations of different max_depth and max_leaf_nodes and the accuracy we obtained.*

By systematically testing various combinations of these parameters, we aim to identify the optimal settings that maximize the classification accuracy of our model. The optimal settings for the hyperparameters is as follow:

```
Best accuracy: 0.8795180722891566
Best max_depth: 4
Best max_leaf_node: 9
Best model report:
              precision    recall  f1-score   support

           0       0.56      0.75      0.64        12
           1       0.96      0.90      0.93        71

    accuracy                           0.88        83
   macro avg       0.76      0.83      0.79        83
weighted avg       0.90      0.88      0.89        83
```

From the output result, we notice that the model perform exceptionally well on `holdout` method comparable to when we use `cross-validation` with plain decision tree.

To further improve the model, we used `cross-validation` method on the dataset and obtained the following results:

```
Best accuracy: 0.916948051948052
Best max_depth: 8
Best max_leaf_nodes: 10
Best model report:
              precision    recall  f1-score   support

           0       0.50      0.58      0.54        12
           1       0.93      0.90      0.91        71

    accuracy                           0.86        83
   macro avg       0.71      0.74      0.73        83
weighted avg       0.87      0.86      0.86        83
```

Similar to the observation from the previous experiment, we noticed tha tthe accuracy jumped again by around `3%` when we use `cross-validation` method.

The results for the first three decision tree in random forest trained with the optimal parameters are presented below:
<!-- TODO: Insert random forest tree here -->
*Figure 6: Visualization of Decision Tree (1)*
*Figure 7: Visualization of Decision Tree (2)*
*Figure 8: Visualization of Decision Tree (3)*



<!--TODO: insert ROC curve here-->
*Figure 9: Decision Tree ROC*
The ROC curve for the Random Forest classifier, with an AUC of 0.89, indicates strong performance in distinguishing between positive and negative classes. The model achieves a high true positive rate early on, with a relatively low false positive rate, and maintains good performance across various thresholds. While it significantly outperforms random guessing, there is still some room for improvement in minimizing false positives. Overall, the model is highly effective for this classification task.

<!-- TODO: insert precision recall curve here -->
*Figure 10: Decision Tree Precision-Recall curve*
The Precision-Recall curve for the Random Forest classifier, with an AUC of 0.98, indicates excellent model performance in identifying positive cases. The curve remains close to a precision of 1.0 across a broad range of recall values, suggesting that the model makes highly accurate positive predictions with minimal false positives for most thresholds. However, as recall approaches 1.0, precision starts to decline sharply, indicating that capturing all true positives comes at the cost of an increasing number of false positives. Overall, this high AUC-PR score demonstrates the model's strong capability in maintaining both high precision and recall, making it very effective for scenarios where false positives need to be minimized.

## 3.3. AdaBoost Classfier

Decision trees, while powerful, have the drawback of being prone to overfitting and failing to generalize well, especially when the tree depth is not properly constrained. To overcome this limitation, we introduce boosting, a technique that combines the predictions of multiple weak learners to build a strong model. Boosting sequentially corrects the errors of previous models, resulting in improved accuracy and robustness. In this experiment, we will explore various combinations of the following hyperparameters: `n_estimators` and `learning_rate`.


The table below summarizes the accuracy achieved for each set of hyperparameter values.

| n_estimators | learning_rate | Accuracy                |
| :----------- | :------------ | :---------------------- |
| 50           | 0.01          | 0.8554216867469879      |
| 50           | 0.05          | 0.8554216867469879      |
| 50           | 0.1           | 0.8313253012048193      |
| 50           | 0.5           | 0.8795180722891566      |
| 50           | 1             | 0.8795180722891566      |
| 100          | 0.01          | 0.8554216867469879      |
| 100          | 0.05          | 0.8433734939759037      |
| 100          | 0.1           | 0.8433734939759037      |
| 100          | 0.5           | 0.8795180722891566      |
| 100          | 1             | 0.891566265060241       |
| 150          | 0.01          | 0.8554216867469879      |
| 150          | 0.05          | 0.8433734939759037      |
| 150          | 0.1           | 0.8674698795180723      |
| 150          | 0.5           | 0.891566265060241       |
| 150          | 1             | 0.8795180722891566      |
| 200          | 0.01          | 0.8554216867469879      |
| 200          | 0.05          | 0.8433734939759037      |
| 200          | 0.1           | 0.8795180722891566      |
| 200          | 0.5           | 0.891566265060241       |
| 200          | 1             | 0.8795180722891566      |
| 250          | 0.01          | 0.8554216867469879      |
| 250          | 0.05          | 0.8433734939759037      |
| 250          | 0.1           | 0.8795180722891566      |
| 250          | 0.5           | 0.891566265060241       |
| 250          | 1             | 0.891566265060241       |
| 300          | 0.01          | 0.8554216867469879      |
| 300          | 0.05          | 0.8674698795180723      |
| 300          | 0.1           | 0.8795180722891566      |
| 300          | 0.5           | 0.8795180722891566      |
| 300          | 1             | 0.891566265060241       |

*Table 4: The combinations of different n_estimators and learning_rate and the accuracy we obtained.*


By systematically testing various combinations of these parameters, we aim to identify the optimal settings that maximize the classification accuracy of our model. The optimal settings for the hyperparameters is as follow:

```
Best accuracy: 0.891566265060241
Best n_estimator: 100
Best learning_rate: 1
Best model report:
              precision    recall  f1-score   support

           0       0.60      0.75      0.67        12
           1       0.96      0.92      0.94        71

    accuracy                           0.89        83
   macro avg       0.78      0.83      0.80        83
weighted avg       0.90      0.89      0.90        83
```

To further optimize the model, we used `cross-validation` method on the dataset and obtained the following results:

```
Best accuracy: 0.8985064935064935
Best n_estimator: 50
Best learning_rate: 0.5
Best model report:
              precision    recall  f1-score   support

           0       0.57      0.67      0.62        12
           1       0.94      0.92      0.93        71

    accuracy                           0.88        83
   macro avg       0.76      0.79      0.77        83
weighted avg       0.89      0.88      0.88        83
```

Unlike trees, we noticed that `cross-validation` does not help much in boosting models. Cross-validation can be less effective for boosting models because they are highly sensitive to small changes in training data, leading to instability and potential overfitting across different folds. Boosting models build sequentially, correcting errors from previous models, making them more prone to variations in performance compared to decision trees. Additionally, their longer training times and reliance on early stopping complicate the use of standard cross-validation, making it less reliable for performance estimation.

Compare to trees, we notice that AdaBoost falls behind on Random Forest Tree model with cross-validation.


<!--TODO: insert ROC curve here-->
*Figure 11: Decision Tree ROC*
The ROC curve for the AdaBoost classifier shows an AUC (Area Under the Curve) of 0.87, indicating good performance in distinguishing between the positive and negative classes. The curve rises sharply initially, reaching a high true positive rate with a low false positive rate, suggesting that the model effectively identifies most true positives while keeping false positives minimal at early thresholds. As the false positive rate increases, the curve becomes less steep, indicating some trade-off between true and false positives. Overall, the model performs significantly better than random guessing (represented by the diagonal line) and demonstrates strong classification capabilities, though there is room for improvement in minimizing false positives at higher thresholds.

<!-- TODO: insert precision recall curve here -->
*Figure 12: Decision Tree Precision-Recall curve*
The Precision-Recall curve for the AdaBoost classifier, with an AUC of 0.97, indicates excellent performance in distinguishing positive instances. The curve remains close to a precision of 1.0 across a wide range of recall values, showing that the model makes highly accurate positive predictions with very few false positives. As recall increases beyond 0.6, precision starts to decline gradually, and it drops sharply near maximum recall. This pattern suggests that while the model can maintain high precision and recall initially, capturing all true positives (high recall) comes at the cost of more false positives, reducing precision. Overall, the model demonstrates strong performance, particularly in scenarios where maintaining high precision is crucial.

Random Forests handle imbalanced datasets better than AdaBoost because they build multiple uncorrelated trees that reduce bias towards the majority class, providing more robust and stable performance. In contrast, AdaBoost assigns higher weights to misclassified instances, which can lead to overfitting, especially with imbalanced data, making its performance more variable during cross-validation.

## 3.4. XGBoost Classifier


The table below summarizes the accuracy achieved for each set of hyperparameter values.

| n_estimator | learning_rate | max_depth | Accuracy             |
|-------------|---------------|-----------|----------------------|
| 50          | 0.01          | 1         | 0.8554216867469879   |
| 50          | 0.01          | 2         | 0.8554216867469879   |
| 50          | 0.01          | 3         | 0.8554216867469879   |
| 50          | 0.01          | 4         | 0.8554216867469879   |
| 50          | 0.01          | 5         | 0.8554216867469879   |
| 50          | 0.01          | 6         | 0.8554216867469879   |
| 50          | 0.01          | 7         | 0.8554216867469879   |
| 50          | 0.01          | 8         | 0.8554216867469879   |
| 50          | 0.01          | 9         | 0.8554216867469879   |
| 50          | 0.01          | 10        | 0.8554216867469879   |
| 50          | 0.05          | 1         | 0.8554216867469879   |
| 50          | 0.05          | 2         | 0.8433734939759037   |
| 50          | 0.05          | 3         | 0.8433734939759037   |
| 50          | 0.05          | 4         | 0.8674698795180723   |
| 50          | 0.05          | 5         | 0.8554216867469879   |
| 50          | 0.05          | 6         | 0.8554216867469879   |
| 50          | 0.05          | 7         | 0.8554216867469879   |
| 50          | 0.05          | 8         | 0.8554216867469879   |
| 50          | 0.05          | 9         | 0.8554216867469879   |
| 50          | 0.05          | 10        | 0.8554216867469879   |
| 50          | 0.1           | 1         | 0.8433734939759037   |
| 50          | 0.1           | 2         | 0.8674698795180723   |
| 50          | 0.1           | 3         | 0.8674698795180723   |
| 50          | 0.1           | 4         | 0.8554216867469879   |
| 50          | 0.1           | 5         | 0.8674698795180723   |
| 50          | 0.1           | 6         | 0.8674698795180723   |
| 50          | 0.1           | 7         | 0.8674698795180723   |
| 50          | 0.1           | 8         | 0.8674698795180723   |
| 50          | 0.1           | 9         | 0.8674698795180723   |
| 50          | 0.1           | 10        | 0.8674698795180723   |
| 50          | 0.5           | 1         | 0.8554216867469879   |
| 50          | 0.5           | 2         | 0.891566265060241    |
| 50          | 0.5           | 3         | 0.8795180722891566   |
| 50          | 0.5           | 4         | 0.8795180722891566   |
| 50          | 0.5           | 5         | 0.891566265060241    |
| 50          | 0.5           | 6         | 0.9036144578313253   |
| 50          | 0.5           | 7         | 0.9036144578313253   |
| 50          | 0.5           | 8         | 0.9036144578313253   |
| 50          | 0.5           | 9         | 0.9036144578313253   |
| 50          | 0.5           | 10        | 0.9036144578313253   |
| 50          | 1             | 1         | 0.8795180722891566   |
| 50          | 1             | 2         | 0.9036144578313253   |
| 50          | 1             | 3         | 0.9036144578313253   |
| 50          | 1             | 4         | 0.9036144578313253   |
| 50          | 1             | 5         | 0.891566265060241    |
| 50          | 1             | 6         | 0.891566265060241    |
| 50          | 1             | 7         | 0.891566265060241    |
| 50          | 1             | 8         | 0.891566265060241    |
| 50          | 1             | 9         | 0.891566265060241    |
| 50          | 1             | 10        | 0.891566265060241    |
| 100         | 0.01          | 1         | 0.8554216867469879   |
| 100         | 0.01          | 2         | 0.8554216867469879   |
| 100         | 0.01          | 3         | 0.8554216867469879   |
| 100         | 0.01          | 4         | 0.8554216867469879   |
| 100         | 0.01          | 5         | 0.8554216867469879   |
| 100         | 0.01          | 6         | 0.8554216867469879   |
| 100         | 0.01          | 7         | 0.8554216867469879   |
| 100         | 0.01          | 8         | 0.8554216867469879   |
| 100         | 0.01          | 9         | 0.8554216867469879   |
| 100         | 0.01          | 10        | 0.8554216867469879   |
| 100         | 0.05          | 1         | 0.8433734939759037   |
| 100         | 0.05          | 2         | 0.8674698795180723   |
| 100         | 0.05          | 3         | 0.8674698795180723   |
| 100         | 0.05          | 4         | 0.8554216867469879   |
| 100         | 0.05          | 5         | 0.8674698795180723   |
| 100         | 0.05          | 6         | 0.8674698795180723   |
| 100         | 0.05          | 7         | 0.8674698795180723   |
| 100         | 0.05          | 8         | 0.8674698795180723   |
| 100         | 0.05          | 9         | 0.8674698795180723   |
| 100         | 0.05          | 10        | 0.8674698795180723   |
| 100         | 0.1           | 1         | 0.8313253012048193   |
| 100         | 0.1           | 2         | 0.891566265060241    |
| 100         | 0.1           | 3         | 0.8554216867469879   |
| 100         | 0.1           | 4         | 0.8674698795180723   |
| 100         | 0.1           | 5         | 0.8795180722891566   |
| 100         | 0.1           | 6         | 0.8795180722891566   |
| 100         | 0.1           | 7         | 0.8795180722891566   |
| 100         | 0.1           | 8         | 0.8795180722891566   |
| 100         | 0.1           | 9         | 0.8795180722891566   |
| 100         | 0.1           | 10        | 0.8795180722891566   |
| 100         | 0.5           | 1         | 0.8795180722891566   |
| 100         | 0.5           | 2         | 0.9036144578313253   |
| 100         | 0.5           | 3         | 0.9036144578313253   |
| 100         | 0.5           | 4         | 0.891566265060241    |
| 100         | 0.5           | 5         | 0.891566265060241    |
| 100         | 0.5           | 6         | 0.9036144578313253   |
| 100         | 0.5           | 7         | 0.9036144578313253   |
| 100         | 0.5           | 8         | 0.9036144578313253   |
| 100         | 0.5           | 9         | 0.9036144578313253   |
| 100         | 0.5           | 10        | 0.9036144578313253   |
| 100         | 1             | 1         | 0.8795180722891566   |
| 100         | 1             | 2         | 0.9156626506024096   |
| 100         | 1             | 3         | 0.891566265060241    |
| 100         | 1             | 4         | 0.891566265060241    |
| 100         | 1             | 5         | 0.9036144578313253   |
| 100         | 1             | 6         | 0.9036144578313253   |
| 100         | 1             | 7         | 0.9036144578313253   |
| 100         | 1             | 8         | 0.9036144578313253   |
| 100         | 1             | 9         | 0.9036144578313253   |
| 100         | 1             | 10        | 0.9036144578313253   |
| 150         | 0.01          | 1         | 0.8554216867469879   |
| 150         | 0.01          | 2         | 0.8554216867469879   |
| 150         | 0.01          | 3         | 0.8433734939759037   |
| 150         | 0.01          | 4         | 0.8554216867469879   |
| 150         | 0.01          | 5         | 0.8433734939759037   |
| 150         | 0.01          | 6         | 0.8433734939759037   |
| 150         | 0.01          | 7         | 0.8433734939759037   |
| 150         | 0.01          | 8         | 0.8433734939759037   |
| 150         | 0.01          | 9         | 0.8433734939759037   |
| 150         | 0.01          | 10        | 0.8433734939759037   |
| 150         | 0.05          | 1         | 0.8433734939759037   |
| 150         | 0.05          | 2         | 0.8795180722891566   |
| 150         | 0.05          | 3         | 0.8674698795180723   |
| 150         | 0.05          | 4         | 0.8554216867469879   |
| 150         | 0.05          | 5         | 0.8554216867469879   |
| 150         | 0.05          | 6         | 0.8554216867469879   |
| 150         | 0.05          | 7         | 0.8554216867469879   |
| 150         | 0.05          | 8         | 0.8554216867469879   |
| 150         | 0.05          | 9         | 0.8554216867469879   |
| 150         | 0.05          | 10        | 0.8554216867469879   |
| 150         | 0.1           | 1         | 0.8433734939759037   |
| 150         | 0.1           | 2         | 0.9036144578313253   |
| 150         | 0.1           | 3         | 0.8795180722891566   |
| 150         | 0.1           | 4         | 0.8674698795180723   |
| 150         | 0.1           | 5         | 0.8795180722891566   |
| 150         | 0.1           | 6         | 0.8674698795180723   |
| 150         | 0.1           | 7         | 0.8674698795180723   |
| 150         | 0.1           | 8         | 0.8674698795180723   |
| 150         | 0.1           | 9         | 0.8674698795180723   |
| 150         | 0.1           | 10        | 0.8674698795180723   |
| 150         | 0.5           | 1         | 0.8795180722891566   |
| 150         | 0.5           | 2         | 0.9036144578313253   |
| 150         | 0.5           | 3         | 0.9036144578313253   |
| 150         | 0.5           | 4         | 0.891566265060241    |
| 150         | 0.5           | 5         | 0.891566265060241    |
| 150         | 0.5           | 6         | 0.9036144578313253   |
| 150         | 0.5           | 7         | 0.9036144578313253   |
| 150         | 0.5           | 8         | 0.9036144578313253   |
| 150         | 0.5           | 9         | 0.9036144578313253   |
| 150         | 0.5           | 10        | 0.9036144578313253   |
| 150         | 1             | 1         | 0.8795180722891566   |
| 150         | 1             | 2         | 0.9156626506024096   |
| 150         | 1             | 3         | 0.891566265060241    |
| 150         | 1             | 4         | 0.891566265060241    |
| 150         | 1             | 5         | 0.9036144578313253   |
| 150         | 1             | 6         | 0.9036144578313253   |
| 150         | 1             | 7         | 0.9036144578313253   |
| 150         | 1             | 8         | 0.9036144578313253   |
| 150         | 1             | 9         | 0.9036144578313253   |
| 150         | 1             | 10        | 0.9036144578313253   |
| 200         | 0.01          | 1         | 0.8554216867469879   |
| 200         | 0.01          | 2         | 0.8433734939759037   |
| 200         | 0.01          | 3         | 0.8433734939759037   |
| 200         | 0.01          | 4         | 0.8554216867469879   |
| 200         | 0.01          | 5         | 0.8554216867469879   |
| 200         | 0.01          | 6         | 0.8554216867469879   |
| 200         | 0.01          | 7         | 0.8554216867469879   |
| 200         | 0.01          | 8         | 0.8554216867469879   |
| 200         | 0.01          | 9         | 0.8554216867469879   |
| 200         | 0.01          | 10        | 0.8554216867469879   |
| 200         | 0.05          | 1         | 0.8433734939759037   |
| 200         | 0.05          | 2         | 0.891566265060241    |
| 200         | 0.05          | 3         | 0.8674698795180723   |
| 200         | 0.05          | 4         | 0.8674698795180723   |
| 200         | 0.05          | 5         | 0.8554216867469879   |
| 200         | 0.05          | 6         | 0.8674698795180723   |
| 200         | 0.05          | 7         | 0.8674698795180723   |
| 200         | 0.05          | 8         | 0.8674698795180723   |
| 200         | 0.05          | 9         | 0.8674698795180723   |
| 200         | 0.05          | 10        | 0.8674698795180723   |
| 200         | 0.1           | 1         | 0.8433734939759037   |
| 200         | 0.1           | 2         | 0.9036144578313253   |
| 200         | 0.1           | 3         | 0.891566265060241    |
| 200         | 0.1           | 4         | 0.891566265060241    |
| 200         | 0.1           | 5         | 0.891566265060241    |
| 200         | 0.1           | 6         | 0.8795180722891566   |
| 200         | 0.1           | 7         | 0.8795180722891566   |
| 200         | 0.1           | 8         | 0.8795180722891566   |
| 200         | 0.1           | 9         | 0.8795180722891566   |
| 200         | 0.1           | 10        | 0.8795180722891566   |
| 200         | 0.5           | 1         | 0.8795180722891566   |
| 200         | 0.5           | 2         | 0.9036144578313253   |
| 200         | 0.5           | 3         | 0.891566265060241    |
| 200         | 0.5           | 4         | 0.891566265060241    |
| 200         | 0.5           | 5         | 0.9036144578313253   |
| 200         | 0.5           | 6         | 0.891566265060241    |
| 200         | 0.5           | 7         | 0.891566265060241    |
| 200         | 0.5           | 8         | 0.891566265060241    |
| 200         | 0.5           | 9         | 0.891566265060241    |
| 200         | 0.5           | 10        | 0.891566265060241    |
| 200         | 1             | 1         | 0.8795180722891566   |
| 200         | 1             | 2         | 0.9156626506024096   |
| 200         | 1             | 3         | 0.891566265060241    |
| 200         | 1             | 4         | 0.891566265060241    |
| 200         | 1             | 5         | 0.891566265060241    |
| 200         | 1             | 6         | 0.891566265060241    |
| 200         | 1             | 7         | 0.891566265060241    |
| 200         | 1             | 8         | 0.891566265060241    |
| 200         | 1             | 9         | 0.891566265060241    |
| 200         | 1             | 10        | 0.891566265060241    |
| 250         | 0.01          | 1         | 0.8554216867469879   |
| 250         | 0.01          | 2         | 0.8433734939759037   |
| 250         | 0.01          | 3         | 0.8433734939759037   |
| 250         | 0.01          | 4         | 0.8674698795180723   |
| 250         | 0.01          | 5         | 0.8554216867469879   |
| 250         | 0.01          | 6         | 0.8554216867469879   |
| 250         | 0.01          | 7         | 0.8554216867469879   |
| 250         | 0.01          | 8         | 0.8554216867469879   |
| 250         | 0.01          | 9         | 0.8554216867469879   |
| 250         | 0.01          | 10        | 0.8554216867469879   |
| 250         | 0.05          | 1         | 0.8433734939759037   |
| 250         | 0.05          | 2         | 0.891566265060241    |
| 250         | 0.05          | 3         | 0.8795180722891566   |
| 250         | 0.05          | 4         | 0.8674698795180723   |
| 250         | 0.05          | 5         | 0.8795180722891566   |
| 250         | 0.05          | 6         | 0.8795180722891566   |
| 250         | 0.05          | 7         | 0.8795180722891566   |
| 250         | 0.05          | 8         | 0.8795180722891566   |
| 250         | 0.05          | 9         | 0.8795180722891566   |
| 250         | 0.05          | 10        | 0.8795180722891566   |
| 250         | 0.1           | 1         | 0.8433734939759037   |
| 250         | 0.1           | 2         | 0.9036144578313253   |
| 250         | 0.1           | 3         | 0.891566265060241    |
| 250         | 0.1           | 4         | 0.891566265060241    |
| 250         | 0.1           | 5         | 0.891566265060241    |
| 250         | 0.1           | 6         | 0.891566265060241    |
| 250         | 0.1           | 7         | 0.891566265060241    |
| 250         | 0.1           | 8         | 0.891566265060241    |
| 250         | 0.1           | 9         | 0.891566265060241    |
| 250         | 0.1           | 10        | 0.891566265060241    |
| 250         | 0.5           | 1         | 0.8795180722891566   |
| 250         | 0.5           | 2         | 0.9036144578313253   |
| 250         | 0.5           | 3         | 0.9036144578313253   |
| 250         | 0.5           | 4         | 0.891566265060241    |
| 250         | 0.5           | 5         | 0.891566265060241    |
| 250         | 0.5           | 6         | 0.891566265060241    |
| 250         | 0.5           | 7         | 0.891566265060241    |
| 250         | 0.5           | 8         | 0.891566265060241    |
| 250         | 0.5           | 9         | 0.891566265060241    |
| 250         | 0.5           | 10        | 0.891566265060241    |
| 250         | 1             | 1         | 0.8795180722891566   |
| 250         | 1             | 2         | 0.9156626506024096   |
| 250         | 1             | 3         | 0.891566265060241    |
| 250         | 1             | 4         | 0.891566265060241    |
| 250         | 1             | 5         | 0.891566265060241    |
| 250         | 1             | 6         | 0.891566265060241    |
| 250         | 1             | 7         | 0.891566265060241    |
| 250         | 1             | 8         | 0.891566265060241    |
| 250         | 1             | 9         | 0.891566265060241    |
| 250         | 1             | 10        | 0.891566265060241    |
| 300         | 0.01          | 1         | 0.8554216867469879   |
| 300         | 0.01          | 2         | 0.8433734939759037   |
| 300         | 0.01          | 3         | 0.8433734939759037   |
| 300         | 0.01          | 4         | 0.8674698795180723   |
| 300         | 0.01          | 5         | 0.8674698795180723   |
| 300         | 0.01          | 6         | 0.8674698795180723   |
| 300         | 0.01          | 7         | 0.8674698795180723   |
| 300         | 0.01          | 8         | 0.8674698795180723   |
| 300         | 0.01          | 9         | 0.8674698795180723   |
| 300         | 0.01          | 10        | 0.8674698795180723   |
| 300         | 0.05          | 1         | 0.8433734939759037   |
| 300         | 0.05          | 2         | 0.891566265060241    |
| 300         | 0.05          | 3         | 0.8795180722891566   |
| 300         | 0.05          | 4         | 0.8674698795180723   |
| 300         | 0.05          | 5         | 0.8795180722891566   |
| 300         | 0.05          | 6         | 0.8674698795180723   |
| 300         | 0.05          | 7         | 0.8674698795180723   |
| 300         | 0.05          | 8         | 0.8674698795180723   |
| 300         | 0.05          | 9         | 0.8674698795180723   |
| 300         | 0.05          | 10        | 0.8674698795180723   |
| 300         | 0.1           | 1         | 0.8554216867469879   |
| 300         | 0.1           | 2         | 0.9036144578313253   |
| 300         | 0.1           | 3         | 0.891566265060241    |
| 300         | 0.1           | 4         | 0.891566265060241    |
| 300         | 0.1           | 5         | 0.891566265060241    |
| 300         | 0.1           | 6         | 0.8795180722891566   |
| 300         | 0.1           | 7         | 0.8795180722891566   |
| 300         | 0.1           | 8         | 0.8795180722891566   |
| 300         | 0.1           | 9         | 0.8795180722891566   |
| 300         | 0.1           | 10        | 0.8795180722891566   |
| 300         | 0.5           | 1         | 0.8795180722891566   |
| 300         | 0.5           | 2         | 0.9036144578313253   |
| 300         | 0.5           | 3         | 0.9036144578313253   |
| 300         | 0.5           | 4         | 0.891566265060241    |
| 300         | 0.5           | 5         | 0.891566265060241    |
| 300         | 0.5           | 6         | 0.9036144578313253   |
| 300         | 0.5           | 7         | 0.9036144578313253   |
| 300         | 0.5           | 8         | 0.9036144578313253   |
| 300         | 0.5           | 9         | 0.9036144578313253   |
| 300         | 0.5           | 10        | 0.9036144578313253   |
| 300         | 1             | 1         | 0.8795180722891566   |
| 300         | 1             | 2         | 0.9156626506024096   |
| 300         | 1             | 3         | 0.891566265060241    |
| 300         | 1             | 4         | 0.891566265060241    |
| 300         | 1             | 5         | 0.891566265060241    |
| 300         | 1             | 6         | 0.891566265060241    |
| 300         | 1             | 7         | 0.891566265060241    |
| 300         | 1             | 8         | 0.891566265060241    |
| 300         | 1             | 9         | 0.891566265060241    |
| 300         | 1             | 10        | 0.891566265060241    |


*Table 5: The combinations of different n_estimators, learning_rate, max_depth and the accuracy we obtained.*


By systematically testing various combinations of these parameters, we aim to identify the optimal settings that maximize the classification accuracy of our model. The optimal settings for the hyperparameters is as follow:

```
Best accuracy: 0.9156626506024096
Best max_depth: 2
Best n_estimator: 100
Best learning_rate: 1
Best model report:
              precision    recall  f1-score   support

           0       0.67      0.83      0.74        12
           1       0.97      0.93      0.95        71

    accuracy                           0.92        83
   macro avg       0.82      0.88      0.85        83
weighted avg       0.93      0.92      0.92        83
```

To further optimize the model, we used `cross-validation` method on the dataset and obtained the following results:

```
Best accuracy: 0.8985064935064935
Best n_estimator: 50
Best learning_rate: 1
Best max_depth: 1
Best model report:
              precision    recall  f1-score   support

           0       0.56      0.75      0.64        12
           1       0.96      0.90      0.93        71

    accuracy                           0.88        83
   macro avg       0.76      0.83      0.79        83
weighted avg       0.90      0.88      0.89        83
```

Similar to Adaboost, the `cross-validation` did not help anything. In contrast, the `holdout` method performs better for boosting model.

<!--TODO: insert ROC curve here-->
*Figure 13: Decision Tree ROC*
The ROC curve for the XGBoost classifier shows the trade-off between the true positive rate (TPR) and the false positive rate (FPR) across different threshold values. The AUC (Area Under the Curve) is 0.89, indicating that the model performs well in distinguishing between classes, as an AUC value closer to 1 represents better classification performance. The ROC curve is significantly above the diagonal (representing random guessing), indicating that the XGBoost model has a strong ability to separate the positive and negative classes in the dataset.

<!-- TODO: insert precision recall curve here -->
*Figure 14: Decision Tree Precision-Recall curve*
The Precision-Recall curve for the XGBoost classifier shows the relationship between precision and recall at different thresholds. The AUC (Area Under the Curve) of 0.98 indicates a very strong performance, suggesting that the model effectively maintains high precision across a wide range of recall values. The curve starts with a precision close to 1 for lower recall values, indicating that the model is very accurate when it makes predictions for positive cases. However, as recall increases, precision gradually decreases, showing a trade-off as more positive samples are correctly identified, but at the cost of increasing false positives. The high AUC value demonstrates that the XGBoost classifier is highly effective in identifying true positives while keeping false positives low.

# 4. Summary

<!-- TODO: insert model comparison -->
*Figure 15: Model Comparison*

From *Figure 15*, we notice that the best models are XGBoost using holdout method and Random Forest using Cross-Validation method, both achieving an accuracy of around 91%. This experiment highlights that while cross-validation effectively mitigates overfitting in tree-based models by providing a more robust evaluation across multiple folds, it may not capture the full potential of boosting models like XGBoost. This discrepancy arises due to the inherent complexity and sensitivity of boosting models to slight changes in the training data, which can result in fluctuating performance across folds in cross-validation. On the other hand, the holdout method provides a more stable estimate for boosting models by evaluating them on a single, consistent test set. Consequently, for complex models like XGBoost, using the holdout method can yield better performance consistency, whereas simpler models like decision trees benefit more from the multiple evaluations offered by cross-validation. This underlines the importance of selecting the appropriate evaluation strategy based on the model's complexity and the nature of the data.
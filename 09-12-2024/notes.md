recap about overfitting and underfitting of trees

recap about ensemble method

problem of bagging and random forest:
    model not knowing what the previous model did

talk about boosting:
    sequential training

additive training:
    start with constant prediction (usually start at 0)
    at every round we add something

evaluation metric:
    regression:
        r2
        r2 adj
        coeff
        t-statistic
        p-value
        f-statistic
        mse (mean squared error)

    classification:
        ** confusion matrix **
        error
        accuracy
        precision
        recall
        f-statistic
        ROC curve
        AuROC curve (area under ROC)
        P-R curve (precision recall curve)

    confusion:
        false positive: model falsely predict input as positive
        faslse negative: model falsely predict input as negative
        true positive
        true negative

        accuracy: (tp + tn) // total; tp, tn = true positive, true negative
        error: (fp + fn) // total; fp, fn = false positive, false negative

        weakness of confusion matrix:
            does not care about class distribution (class imbalance)

        NOTE: take account into class imbalance

        to tackle class imbalance:
            cost matrix
                need confusion matrix and cost matrix
                negative cost mean positive reward

                need to balance between cost and accuracy
        
        cost-sensitive measures (no need of cost matrix):
            precision (p)
                precision = tp / tp + fp

                low precision: 
                    bluffing (making too many positive claim)

            recall (r):
                among all the positive instance showed to model, how many corrrect it was able to get

                total positive example showed to model: tp + fn

                recall = tp / tp + fn

            f-measure (harmonic mean between recall and precision):
                2 * r * p / (r + p)
        
    what should be the minimum acceptable value of metrics:
        acc, p, r, f should be greater than 50%

        if we take less than 50%, it mean the inverse of model is better than the model

learning curve:
    accuracy vs sample size

    wait until the curve flatten out to not add more data

how to carry out trials:

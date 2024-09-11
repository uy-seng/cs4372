recap:
    bias variance tradeoff
        for a singular model, you cannot do better than a certain point

    solution to the problem:
        ensemble model (aggregate models) (stacking of models)
        ensemble = group

        instead of single of model, we train a group of model and we aggregate the results

    analogy:
        suppose we have 1M dollar
        volatile = variance
        we have the following choice:
            - put all in a single basket
            - take the average of n stock -> lower variance
    
in term of ml:
    how do you train n models on the same dataset (same mode, same output?)
    
    solution (bagging):
        we create (n) variants of the given dataset using technique called bootstrapping (statistical technique)
        each varaint  -> use that to train 1 model

        drawback:
            the model are correlated

    random forest:
        difference to bagging:
            each time a split is perfomed, the search for split variable is limited to a random subset of m of the p variables. (trees are more lighter and uncorrelated)

            keyword: mtry

the problem with bagging and random forest:
    the tree does not know what the other tree is doing
    models are trained in isolation

boosting:
    model are trained sequentially
    model are kept simple by design
    limit the growth of tree
        max_depth = 1 -> stumps

    adaboost:
        data is re-weighted in every round based on the error
        this is a form of knowledge transfer between iterations
    xgboost (for tabular data)
        the most popular model
1. Project Introduction
in this project, we select the lung cancer survey data from kaggle to perform a classification on whether or not someone has lung cancer based on the metrics of GENDER,AGE,SMOKING,YELLOW_FINGERS,ANXIETY,PEER_PRESSURE,CHRONIC DISEASE,FATIGUE ,ALLERGY ,WHEEZING,ALCOHOL CONSUMING,COUGHING,SHORTNESS OF BREATH,SWALLOWING DIFFICULTY,CHEST PAIN.

2. data pre-processing
we perform the following operations to sanitize and check the data:
- check for null values
- check for duplicated values and remove it
- replace all categorical values to numerical values
- sanitize the column names for the dataset

we will take all the column as feature as the correlation matrix shows that each column has a relevant correlation with the target output. (refer to table 1)
<-- insert correlation with lung cancer here -->
<-- insert heatmap here -->


3. Tree Model Building
we will be using 4 different models to compare the classification result and accuracy of those model. further more for each model, we will dissect the accuracy difference when using different parameters and different method of splitting dataset. the model that we will use is as follows:
- plain decision tree classifier
- random forest classifier
- adaboost classifier
- xgboost classifier

3.1 Plain decision tree classifier
we are testing the combination of the parameters, max_depth and criterions. the table below shows the accuracy and the set of combinations hyperparameter it use. the result shows that the best parameter for this model is:
- max_depth:
- criterion

the combination of different criterion does not seems to affect the accuracy of the model in this experiment. so we will ommit criterion and use the default gini index for the subsequent experiments. the result of tree trained using the best parameters is shown as below:
<-- insert tree image here-->

we notice that shallow depth also produces us the accuracy of <-- insert percent --> %. therefore  it tells us that this data is linearly separable as the model was able to identify the apparent feature that groups the data and give out accurate target results.

the classification report of the model is as follow:
<-- insert report here -->

we notice that the accuracy of the model is good and blah blah blah.

in the roc curve,
yap here

in the precision-recall curve,
yap here

the output of the tree is shown as follows
we also notice that the performance of the model using the holdout method for dataset is <--insert ideas here> to the performance of using cross-validation method for the dataset. it give us the accuracy of:
- model using holdout: <--insert number-->
- model using cross-validation: <--insert number-->

3.2. random forest classficiation tree
the plain decision tree has a drawback of <-- insert opinion here-->. let's try using the same parameter of experiment as plain decision tree for this random forest classfication.
in random forest we eliminate out the hyperparameter criterion from our combinations and instead try using max_leaf_node in order to find the optimal performance of the model.

the table below shows the accuracy and the different combinations of hyperparameter that we used:
<--insert table here-->

we notice that the best accuracy is <--insert here-->
in the roc curve,
yap here

in the precision-recall curve,
yap here

the output of the tree is shown as follows
<--insert tree-->

similar to the previous experiment, <--compare holdout and cross-validation-->
in this model we notice that the accuracy <--insert ideas here--> compared to the plain decision tree.

3.3. adaboost classification
<--insert drawback of tree-->
<-- introduce boosting-->
in this experiement we are using the combinations of the following hyperparameter,  <--insert hyperparameter here-->

the result of accuracy from the combinations of hyperparameter is as follow:
<--insert table here-->
<--yap about table-->
 
 <--compare the model to the tree-->
 <--explain about holdout and cross validation for this-->

 3.4. xgboost classification
 in xgboost, we notice the performance as following with the same combinations of hyperparameter,
 <--insert table here-->

 <--compare the mdoel to adaboost and tree-->
 
 in xgboost model, we notice the cross-validation method perform worse compared to the hold-out method. we suspect it is due to the complexity of the model itself.

 <-- show how this model perform well compared to all the other previous models -->

 we notice that the best accuracy is <--insert here-->
in the roc curve,
yap here

in the precision-recall curve,
yap here

4. Summary
the bar chart below shows the best accuracy obtained through different parameters combination of each mode. we notice that <--which model perform best--> and <--which model perform worse-->

<--explain the behavior, and show that it aligned with our expectations-->
<--analyze the reason why X models works best-->
# Assignment 1 Report
## 1. Obtaining Dataset
In this assignment, I used the [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) datasets from [UCI Dataset](https://archive.ics.uci.edu/dataset).

The table below contains all the attribute for the dataset:
| Variable Name         | Role    | Type       | Description             | Missing Values |
|-----------------------|---------|------------|-------------------------|----------------|
| fixed_acidity         | Feature | Continuous |                         | no             |
| volatile_acidity      | Feature | Continuous |                         | no             |
| citric_acid           | Feature | Continuous |                         | no             |
| residual_sugar        | Feature | Continuous |                         | no             |
| chlorides             | Feature | Continuous |                         | no             |
| free_sulfur_dioxide   | Feature | Continuous |                         | no             |
| total_sulfur_dioxide  | Feature | Continuous |                         | no             |
| density               | Feature | Continuous |                         | no             |
| pH                    | Feature | Continuous |                         | no             |
| sulphates             | Feature | Continuous |                         | no             |
| alcohol               | Feature | Continuous |                         | no             |
| quality               | Target  | Integer    | score between 0 and 10  | no             |


## 2. Preprocessing Dataset
The data in the dataset have **missing value**. However, there exist some outlier for each features. In this assignment, I decided to keep the outlier since it does not affects our model.

Below contains the correlation between our target, **quality** and other features:
| Feature               | Correlation |
|-----------------------|-------------|
| fixed acidity         | 0.124052    |
| volatile acidity      | -0.390558   |
| citric acid           | 0.226373    |
| residual sugar        | 0.013732    |
| chlorides             | -0.128907   |
| free sulfur dioxide   | -0.050656   |
| total sulfur dioxide  | -0.185100   |
| density               | -0.174919   |
| pH                    | -0.057731   |
| sulphates             | 0.251397    |
| alcohol               | 0.476166    |

We noticed that features such as **residual sugar**, **free sulfur dioxide** and **pH** do not play a huge role for our model so we can exclude the features from our model.

## 3. Regression Model
### 3.1. SGDRegressor
We use **Standard Scaler** to normalize our dataset in order to have better performance. We divide the datasets into **80/20** split for training and testing.

After testing, we have an **R2** score of **0.37**.

I initially thought that the low score might be due to the outlier data in the dataset. I tried removing the outlier from the dataset. However, the **R2** score seems to be worse than the one without removed outliers.

In my conclusion for the low **R2** score, I believe that **Linear Regression** model is not suitable for this problem because the data that we have is not close to linear, causing our regression results to deviate from the ground truth.

### 3.2. Ordinary Least Square(OLS)
I used the **OLS** model from **statsmodel** package. The OLS model gave us a similar **R2** score compared to the **SGDRegressor**. This confirms that our conclusion above is correct.
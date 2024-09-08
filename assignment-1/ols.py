import statsmodels.api as sm

df  = pd.read_csv("https://raw.githubusercontent.com/uy-seng/cs4372/main/assignment-1/dataset/winequality-red.csv", names=column_names, delimiter=";")

x = df.drop('quality', axis=1)
y = df['quality']

features = ["volatile acidity",
                "chlorides","total sulfur dioxide", "pH", "sulphates", "alcohol"]

x = x[features]

x = sm.add_constant(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


ols_model = sm.OLS(y_train, x_train)

res = ols_model.fit()
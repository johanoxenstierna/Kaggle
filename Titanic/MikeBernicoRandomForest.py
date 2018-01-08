
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

from tabulate import tabulate

X = pd.read_csv("train.csv")
y = X.pop("Survived")

# print(tabulate(X, ))

# aggregate table to view statistics
print(tabulate(X.describe(), X))

# fill NULLS
X["Age"].fillna(X.Age.mean(), inplace=True)



# BUILD FAST SIMPLE MODEL TO GET FIRST BENCHMARK*********************************
# get numeric variables
numeric_variables = list(X.dtypes[X.dtypes != "object"].index)
print(tabulate((X[numeric_variables].head()), X))

# build model
model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
model.fit(X[numeric_variables], y)
# model score is c-stat.
# model.oob_score

y_oob = model.oob_prediction_
print("c-stat: ", roc_auc_score(y, y_oob))
# print(y_oob)  #probability of survival (this is what is then converted into classes)

# *******************************************************************************

# # function that describes categorical variables
# def describe_categorical(X):
#










print("EOF")
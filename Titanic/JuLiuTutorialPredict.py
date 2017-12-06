import pandas as pd
import JuLiuUtils as utils
from sklearn import linear_model, preprocessing, tree, model_selection


train = pd.read_csv("train.csv")
utils.clean_data(train)


# Just looking at Sex
# train["Hyp"] = 0
# train.loc[train.Sex == "female", "Hyp"] = 1
#
# train["Result"] = 0
# train.loc[train.Survived == train["Hyp"], "Result"] = 1
#
# print(train["Result"].value_counts(normalize=True))


target = train["Survived"].values
features = train[["Pclass", "Age", "Fare", "Embarked", "Sex", "SibSp", "Parch"]].values


classifier = linear_model.LogisticRegression()
classifier.fit(features, target)
print("Linear: " + str(classifier.score(features, target)))

poly = preprocessing.PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(features)
classifier.fit(poly_features, target)
print("Poly_linear: " + str(classifier.score(poly_features, target)))

decision_tree = tree.DecisionTreeClassifier(
    random_state=1,
    max_depth = 7,
    min_samples_split=2
)
decision_tree.fit(features, target)
scores = model_selection.cross_val_score(decision_tree, features, target, scoring='accuracy', cv=50)
print("Decision_tree: " + str(scores.mean()))

a = 4
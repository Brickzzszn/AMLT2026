# Core libraries
import numpy as np
import pandas as pd

#visualisation
from matplotlib import pyplot as plt

#Datasets
from sklearn.datasets import load_breast_cancer, make_classification


#Data splitting and validation

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RepeatedStratifiedKFold
)

# Pre-processing
from sklearn.preprocessing import LabelEncoder

# Evaluation
from sklearn.metrics import accuracy_score, f1_score

#Base learners
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#Ensemble methods
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    BaggingClassifier,
    VotingClassifier
)


"--------ACTIVITY 1(AdaBoost boosting method)----------"

#load dataset
breast_cancer = load_breast_cancer()

#create feature DataFrame
X = pd.DataFrame(
    breast_cancer.data,
    columns=breast_cancer.feature_names
)

#create categorical target
y= pd.Categorical.from_codes(
    breast_cancer.target,
    breast_cancer.target_names
)

#convert target labels to integers
encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("Dataset shape", X.shape)
print("Target classes", encoder.classes_)

#create train/test sets:
X_train, X_test, y_train, y_test = train_test_split(
X,
    y,
    test_size=0.25,
    random_state=1,
    stratify=y
)

print("training observations:", X_train.shape[0])
print("testing observations:", X_test.shape[0])


#creating AdaBoost model
max_depth = 1
boosting_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        random_state=1
),
    n_estimators=200,
    learning_rate=1
)
boosting_model.fit(X_train, y_train)
boost_predictions = boosting_model.predict(X_test)

boost_accuracy = accuracy_score(
    y_test,
    boost_predictions
)

boost_f1 = f1_score(
    y_test,
    boost_predictions
)

print(f"AdaBoost Accuracy:{boost_accuracy:.2f}")
print(f"AdaBoost F1 score:{boost_f1:.2f}")


"--------ACTIVITY 2(Random Forest bagging method)----------"

#create the random forest
bagging_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=1,
    random_state=1,
    n_jobs=-1
)
bagging_model.fit(X_train, y_train)

bagging_predictions = bagging_model.predict(X_test)

bagging_accuracy = accuracy_score(
    y_test,
    bagging_predictions
)

bagging_f1 = f1_score (
    y_test,
    bagging_predictions
)
print(f"Random Forest Accuracy:{bagging_accuracy:.2f}")
print(f"Random Forest F1 score:{bagging_f1:.2f}")

"--------ACTIVITY 3 K-nearest neighbours KNN(Bagging))----------"
#classification dataset
X_knn, y_knn = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=5
)

print("Dataset shape:", X_knn.shape)


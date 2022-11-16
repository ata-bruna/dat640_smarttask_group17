'''
    This script is used to evaluate different methods 
    to perform category prediction
'''

import time
from sklearn.metrics import *
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from data_cleaning import create_dataset


def baseline(text_clf, df_train, df_test):
    print("_" * 80)
    print("Training: ")
    print(text_clf)
    t0 = time.time()
    clf = Pipeline(
        [("vect", TfidfVectorizer()), ("clf", text_clf)]
    )

    clf.fit(df_train["question"], df_train["category_new"])
    train_time = time.time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time.time()
    pred = clf.predict(df_test["question"].to_list())
    test_time = time.time() - t0
    print("test time:  %0.3fs" % test_time)

    score = accuracy_score(df_test["category_new"].to_list(), pred)
    print("accuracy:   %0.3f" % score)

    print()
    print("                       Classification report")
    report = classification_report(df_test["category_new"].to_list(), pred)
    print(report)

    return {
            "score": score,
            "report": report, 
            "train_time": train_time, 
            "test_time": test_time, 
            }

def run_baseline(df_train, df_test):
    clfs = [
            ("SVM", SVC(kernel='linear')), 
            ("Perceptron", Perceptron(max_iter=500)),
            ("NB", ComplementNB()),
            ("DT", DecisionTreeClassifier()),
            ("RF", RandomForestClassifier(n_estimators=250))
            ]   

    for classifier in clfs:
        baseline(classifier[1], df_train, df_test)  

if __name__ == "__main__":
    # loading dataset
    df_train, df_test = create_dataset()

    run_baseline(df_train, df_test)

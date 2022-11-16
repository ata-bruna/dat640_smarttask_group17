"""
This script is used to training SVM model for category prediction
on the DBpedia dataset.
    Returns:
        Predictions and accuracy.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
import pandas as pd
import data_cleaning as dc


def svm(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """Trains the model on SVM classifier, and predicts on the testing dataset.
    Args:
        df_train (pd.DataFrame): Dataset used for training the model.
        df_test (pd.DataFrame): Dataset used for testing the model.
    Returns:
        predicted_svm: The predicted values of the trained svm.
        Accuracy: Accuracy of the trained model.
    """

    text_clf_svm = Pipeline(
        [("TF-idf", TfidfVectorizer()), ("clf-svm", SVC(kernel="linear"))]
    )

    text_clf_svm = text_clf_svm.fit(df_train["question"], df_train["category_new"])
    predicted_svm = text_clf_svm.predict(df_test["question"])
    accuracy = metrics.accuracy_score(df_test["category_new"], predicted_svm)
    return predicted_svm, accuracy


if __name__ == "__main__":
    df_train, df_test = dc.create_dataset()
    predicted_svm, accuracy = svm(df_train, df_test)
    print("Accuracy of SVM: ", accuracy.round(2))
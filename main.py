"""
    This script is used to run all the models necessary 
    to achieve the results presented on the report
"""

import data_cleaning as dc
import SVM as SVM
import evaluate as eval
from baseline_category_prediction import run_baseline
import json
from elasticsearch import Elasticsearch
from bm25_model import es_BM25
from advanced_model import create_query_terms, ltr_feature_vectors, ltr_predict
from sklearn.linear_model import SGDRegressor

INDEX_NAME = 'dbpedia'

if __name__ == "__main__":
    df_train, df_test = dc.create_dataset()

    # Evaluating several models for category prediction
    # run_baseline(df_train, df_test)

    # Applying SVM for Category prediction
    print('        Category prediction')
    predicted_svm, accuracy = SVM.svm(df_train, df_test)
    print("SVM")
    print("-" * 80)
    print("Accuracy of SVM: ", accuracy.round(2))
    print()
    print("Results sample:")
    print('actual: ', df_test.iloc[:10,-1].to_list())
    print('predicted: ', (predicted_svm[:10]))
    print('\n \n')

    # Creating new testing and training Dataframes for type predictions
    df_test["predicted_category"] = predicted_svm
    df_train_resource = df_train.loc[df_train["category"] == "resource"]
    df_test_resource = df_test.loc[df_test["predicted_category"] == "resource"]
    
    # Creating json file to be used by elasticsearch
    print('-'*80)
    print('Creating test data to use with elasticsearch')
    dc.save_results_category(df_test, 'test_queries_svm_output')
    print('-'*80)
    print('Test queries saved under "results/test_queries_svm_output.json"')
    print('\n \n')

    # Loading data for Type prediction
    train = dc.load_data('datasets/DBpedia/smarttask_dbpedia_train.json')
    test = dc.load_data('results/test_queries_svm_output.json')
    training_queries = dc.load_queries(train)
    test_queries = dc.load_queries(test)

    print('\n \n')
    print('Initiating Elasticsearch...')
    print('-'*80)
    es = Elasticsearch()
    print(es.info())
    print('\n \n')

    print('Starting training BM25...')
    title = 'bm25_es'
    test_res = es_BM25(es, test_queries, index=INDEX_NAME)
    dc.save_test_results(test_res, test, title=title)
    print(f'Test results saved in "results/{title}_system_output.json"')
    eval.eval_res(f"results/{title}_system_output.json")

    print('Loading training queries for LTR...')
    training_queries = create_query_terms(training_queries, es)
    test_queries = create_query_terms(test_queries, es)
    print('\n \n')
    
    print('Starting training LTR...')
    X, y = ltr_feature_vectors(es, training_queries, k=100, index=INDEX_NAME)
    model = SGDRegressor(max_iter=1000, tol=1e-3, 
                        penalty = "elasticnet",
                        loss="huber",random_state = 42, early_stopping=True)
    model.fit(X, y)

    print('Predicting category types...')
    title = 'advanced_es'
    test_advanced = ltr_predict(es, test_queries, model, k=100, index=INDEX_NAME)
    dc.save_test_results(test_advanced, test, title=title)
    print(f'Test results saved in "results/{title}_system_output.json"')

    # Evaluation 
    eval.eval_res(f"results/{title}_system_output.json")

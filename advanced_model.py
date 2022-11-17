import json
from typing import Callable, Dict, List, Set, Tuple
import numpy as np
from elasticsearch import Elasticsearch
import re
import math
from sklearn.linear_model import SGDRegressor
import re
from collections import Counter
from evaluate import load_type_hierarchy, get_type_path
from data_cleaning import load_data, load_queries, save_test_results
from evaluate import load_ground_truth, load_type_hierarchy, load_system_output, evaluate


# elastic search index
INDEX_NAME = 'dbpedia_toy'

def analyze_query(
    es: Elasticsearch, query: str, field: str, index: str = INDEX_NAME
) -> List[str]:
    """Analyzes a query with respect to the relevant index.

    Args:
        es: Elasticsearch object instance.
        query: String of query terms.
        field: The field with respect to which the query is analyzed.
        index: Name of the index with respect to which the query is analyzed.

    Returns:
        A list of query terms that exist in the specified field among the
        documents in the index.
    """
    tokens = es.indices.analyze(index=index, body={"text": query})["tokens"]
    query_terms = []
    for t in sorted(tokens, key=lambda x: x["position"]):
        # Use a boolean query to find at least one document that contains the
        # term.
        hits = (
            es.search(
                index=index,
                query={"match": {field: t["token"]}},
                _source=False,
                size=1,
            )
            .get("hits", {})
            .get("hits", {})
        )
        doc_id = hits[0]["_id"] if len(hits) > 0 else None
        if doc_id is None:
            continue
        query_terms.append(t["token"])
    return query_terms


def create_query_terms(query_dict: Dict, es: Elasticsearch):
    """Analyze all queries and add terms to dictionary
    Args: 
        query_dict: a dictionary containing {
                                        'query_id': {
                                            'question': 'some question',
                                            'category': 'some category',
                                            'type': 'some type'
                                        }
                                    }
        es: Elasticsearch object instance.
    Returns: 
        A dictionary with query_ids as keys, question, category, 
        type and analized query_terms as values.
    """
    for query_id, query_features in query_dict.items():
        query_dict[query_id]['query_terms'] = analyze_query(es, 
                                            query_features['question'], 
                                            'abstract')
    
    return query_dict


def extract_features(query_terms: List, 
                    doc_id: str, 
                    es: Elasticsearch, 
                    index=INDEX_NAME):
    """
    Extracts query features, document features and 
    query-document features of a query and document pair.
    
        Arguments:
            query_terms: List of analyzed query terms.
            doc_id: Document identifier of indexed document.
            es: Elasticsearch object instance.
            index: Name of relevant index on the running Elasticsearch 
            service. 
            
        Returns:
            List of extracted feature values in a fixed order.
    """
    
    total_docs = es.count(index=INDEX_NAME)['count'] 
    query = dict(Counter(query_terms))    

    doc_term_freqs = {} 
    tv = es.termvectors(index=index, id=doc_id, fields='abstract', term_statistics=False)    
    for term, term_stat in tv['term_vectors']['abstract']['terms'].items():
        doc_term_freqs[term] = term_stat['term_freq']
    
    idf = []
    term_doc_freq_dict = {}
    for term in query_terms:
        if not term in term_doc_freq_dict:
            n = 0
            hits = es.search(
                index=index, 
                query = {
                    "bool": {
                        "must": {
                            "match": {
                                "abstract": term
                                }
                            }, 
                            "must_not": {
                                "match": {
                                    "instance": "owl:Thing"
                                    }
                                }
                            }
                        }, 
                _source=False, size=1).get('hits',{}).get('hits',{})
            
            doc_id = (hits[0]['_id'] if (len(hits) > 0) else None)

            if doc_id is not None:
                tv = es.termvectors(index=index, id=doc_id, fields='abstract', term_statistics=True)['term_vectors']['abstract']['terms']
                if term in tv:
                    n = tv[term]['doc_freq']                    
            term_doc_freq_dict[term] = n
            
        n = term_doc_freq_dict[term]
        if n: 
            idf.append(math.log(total_docs/n))

    terms_doc_unique = [v for k,v in doc_term_freqs.items() if k in query] 
    
    return [
        len(query_terms),
        sum(idf),
        max([0] + idf),
        (sum(idf) / max(len(idf),1)),
        sum(doc_term_freqs.values()),
        len(terms_doc_unique),
        sum(terms_doc_unique),
        max([0] + terms_doc_unique),
        (sum(terms_doc_unique) / max(len(query.keys()),1))
    ]


def ltr_feature_vectors(es: Elasticsearch, 
                        training_queries: Dict, 
                        k=100, index=INDEX_NAME):
    """
    LTR model, generates the feature vectors X and the 
    relevance labels y.
    
    Relevancy scores are defined as:
        0 = Not relevant
        1 = Partially relevant
        2 = Relevant
    
    Arguments:
        es: Elasticsearch object instance.
        k: How many documents to be retrieved from elasticsearch.
        training_queries: training queries.
        index: Name of the index with respect to which the query 
        is analyzed.  
    
    Returns:
        Returns a list of feature vectors, a list of their respective 
        labels (relevance) and a list of instances (type instances) 
        noted from every retrieved doc.
    """
    progress, N = 0, len(training_queries)
    X=[]
    y=[] 
    instances = []
    type_hierarchy, _ = load_type_hierarchy("evaluation\dbpedia\dbpedia_types.tsv")

    
    for query_id, query_features in training_queries.items():
        type_relevancy = {}        
        for typ in query_features['type']:
            if not typ in type_hierarchy:
                continue
            hierarchy = get_type_path(typ, type_hierarchy)[::-1]
            for v in hierarchy:
                type_relevancy[v] = 1 # Relevant, its in the same hierarchy but in a diff pos.            
                
        for typ in query_features['type']:
            type_relevancy[typ] = 2 # This is the type we want. Give it the highest weight.
            
        if len(type_relevancy) == 0:
            continue

        query = query_features['query_terms']
        if len(query) == 0:
            continue
        hits = es.search(index=index, _source=True, size=k, 
           query = {"bool": {
            "must": {
                "match": {
                    "abstract": ' '.join(query)
                    }
                }, 
            "must_not": {
                "match": {
                    "instance": "owl:Thing"
                    }
                }
            }
        })['hits']['hits']

        for hit in hits:
            X.append(extract_features(query, hit['_id'], es, index))
            relevancy = 0 # Default = not relevant
            instance_type = hit['_source']['instance']
            if instance_type in type_relevancy:
                relevancy = type_relevancy[instance_type]
            elif instance_type in type_hierarchy:                
                relevancy = get_type_path(instance_type, type_hierarchy)[::-1]
                relevancy = max([(1 if (t in type_relevancy) else 0) for t in relevancy] + [0])
            y.append(relevancy)
            instances.append(instance_type)

        progress += 1
        if progress % 500 == 0:
            print(
                "Processing query {}/{} ID {}".format(
                    progress, len(training_queries), query_id
                )
            )
        if progress == N:
            print(
                "{}/{} queries processed ID {}".format(
                    progress, len(test_queries), query_id
                )
            )
            break

    return X, y #, instances


def ltr_predict(es: Elasticsearch, 
                test_queries: Dict, 
                model, k=100, index=INDEX_NAME):
    """
    Predicts the category types for test queries. 
    Re-ranks the first k documents from elasticsearch.
    
    Args:
        es: Elasticsearch object instance.
        model: A sklearn regressor.
        k: How many documents to be retrieved from elasticsearch.
        index: Name of the index with respect to which the query 
        is analyzed.  
    
    Returns:
        A dictionary containing the query_ids and the top 10 predicted 
        types.
    """
    progress, N = 0, len(test_queries)
    results = {}
    for query_id, query_features in test_queries.items():
        query = query_features['query_terms']
        hits = es.search(index=index, _source=True, size=k, 
            query = {"bool": {
                "must": {
                    "match": {
                        "abstract": ' '.join(query)
                        }
                    }, 
                "must_not": {
                    "match": {
                        "instance": "owl:Thing"}
                        }
                    }
                })['hits']['hits']
        feat_vecs, instances = [], []
        for hit in hits:
            feat_vecs.append(extract_features(query, hit['_id'], es, index))
            instances.append(hit['_source']['instance'])
        if len(instances) == 0: # No hits!
            results[query_id] = 'N/A'
            continue
        instances_rerank = [instances[idx] for idx in np.argsort(model.predict(feat_vecs))[::-1]]
        results[query_id] = [x[0] for x in Counter(instances_rerank).most_common(10)]

        progress += 1
        if progress % 500 == 0:
            print(
                "Processing query {}/{} ID {}".format(
                    progress, len(test_queries), query_id
                )
            )
        if progress == N:
            print(
                "{}/{} queries processed ID {}".format(
                    progress, len(test_queries), query_id
                )
            )
            break

    return results


def save_test_results(test_res, test, title):
    """Saves test results in a json file with the appropriate 
    format for evaluation.
    
    Args:
        test_res: list of results containing query_id and predicted 
        types.
        test: the test set containing the predicted categories.
    """

    all_test_queries = {}
    for x in test:
        all_test_queries.update({x['id']:{
                                    'id': x['id'],
                                    'category': x['category'],
                                    'type': x['type']}
                                })
    
    for key, _ in all_test_queries.items():
        try:
            all_test_queries[key]['type'] = test_res[key]['type']
        except:
            continue

    f = open(f"results/{title}_system_output.json", "w")
    json.dump(list(all_test_queries.values()), f)
    f.close()


if __name__ == "__main__":
    train = load_data('datasets/DBpedia/smarttask_dbpedia_train.json')
    test = load_data('results/test_queries_svm_output.json')
    training_queries = load_queries(train)
    test_queries = load_queries(test)

    es = Elasticsearch()

    training_queries = create_query_terms(training_queries, es)
    test_queries = create_query_terms(test_queries, es)

    print('Starting training LTR...')
    X, y = ltr_feature_vectors(es, training_queries, k=30, index=INDEX_NAME)
    model = SGDRegressor(max_iter=1000, tol=1e-3, 
                        penalty = "elasticnet",
                        loss="huber",random_state = 42, early_stopping=True)
    model.fit(X, y)

    print('Predicting category types...')
    title = 'advanced_es'
    test_advanced = ltr_predict(es, test_queries, model, k=30, index=INDEX_NAME)
    save_test_results(test_advanced, test, title=title)
    print(f'Test results saved in "results/{title}_system_output.json"')

    # Evaluation 
    type_hierarchy, max_depth = load_type_hierarchy('evaluation/dbpedia/dbpedia_types.tsv')
    ground_truth = load_ground_truth('datasets/DBpedia/smarttask_dbpedia_test.json', type_hierarchy)
    system_output = load_system_output('results/advanced_es_system_output.json')
    evaluate(system_output, ground_truth, type_hierarchy, max_depth)


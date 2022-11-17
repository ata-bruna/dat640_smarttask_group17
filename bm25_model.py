#%%
import numpy as np
import os
from collections import Counter
from collections import defaultdict
from elasticsearch import Elasticsearch
from data_cleaning import load_data, load_queries, save_test_results
from evaluate import load_ground_truth, load_type_hierarchy, load_system_output, evaluate


INDEX_NAME = 'dbpedia'


def baseline_retrieval(es, query:str, field = 'abstract', index = INDEX_NAME):
    """
    Baseline retrieval using the inbuilt BM25 index from elasticsearch

    Args:
        es: Elasticsearch object
        index: string
        query: string, space separated terms
        k: integer
    
    Returns:
        List of k first entity IDs(string)
    """
    hits = es.search(index=index, size=200,
                query = {"bool": {"must": {"match": 
                                          {"abstract": query}}, 
                                 "must_not": {"match": 
                                            {"instance": "owl:Thing"}
                                            }}})['hits']['hits']
    hit_ids = [obj['_id'] for obj in hits]
    hit_types = [es.get(index=index, id=doc)["_source"].get("instance") for doc in hit_ids]
    result = [h[0] for h in Counter(hit_types).most_common(10)]
    
    return result


def es_BM25(es, data, index=INDEX_NAME):
    """Evaluates BM25 for test queries
    
    Args: 
        es: Elasticsearch
        data: test queries dictionary containing query id, question and category.
    
    Returns
        A Dictionary containing the query id, category and the predicted types.
    """
    
    results = {} 
    for query_id, query in data.items():
        if len(query['question'])>0:
            response = baseline_retrieval(es, query['question'],  
                                        field = 'abstract', index=index)
            results.update({query_id:{
                                "id": query_id,
                                "category": query["category"],
                                "type": response
                                }
                            })
        else:
            continue
    return results


# %%
if __name__ == "__main__":
    es = Elasticsearch()
    test = load_data('results/test_queries_svm_output.json')
    test_queries = load_queries(test)
    test_res = es_BM25(es, test_queries, index=INDEX_NAME)
    save_test_results(test_res, test, title='bm25_es')

    type_hierarchy, max_depth = load_type_hierarchy('evaluation/dbpedia/dbpedia_types.tsv')
    ground_truth = load_ground_truth('datasets/DBpedia/smarttask_dbpedia_test.json', type_hierarchy)
    system_output = load_system_output('results/bm25_es1_system_output.json')
    evaluate(system_output, ground_truth, type_hierarchy, max_depth)


#%%
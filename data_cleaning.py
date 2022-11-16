import numpy as np
import pandas as pd
import json
import os
import string
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def load_data(path: str) -> dict:
    """Loads the data from the path.
    Args:
        path (str): relative path of the data files.
    Returns:
        dict: DBpedia ontology.
    """
    f = open(path, "r")
    data = json.load(f)
    f.close()
    return data


def load_data_as_dataframe(path: str) -> pd.DataFrame:
    """Loads the data and converts it into pandas dataframe.
    Args:
        path (str): relative path of the data files.
    Returns:
        pd.DataFrame: A pandas df with keys as columns and values as entries.
    """
    df = pd.DataFrame(load_data(path))
    return df


def separate_cat(df: pd.DataFrame) -> pd.DataFrame:
    """ Reassigns categories to dataframe.
    Args:
        df (pd.DataFrame): Dataset containing category column.
    Returns:
        pd.DataFrame: with additional column category_new
    """
    df["category_new"] = np.where(df["category"] == "literal", df["type"].str[0], df["category"])
    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """ Rearrange category into a new column and drop nan vales.
    Args:
        df (pd.DataFrame): Dataset.
    Returns:
        pd.DataFrame: with additional column category_new
    """
    df = separate_cat(df)
    df = df.dropna()
    return df


def get_dataframe():
    cur_path = os.path.dirname(os.path.abspath("__file__"))
    new_path = cur_path.replace("source", "")
    train_path = new_path + "\datasets\DBpedia\smarttask_dbpedia_train.json"
    test_path = new_path + "\datasets\DBpedia\smarttask_dbpedia_test.json"
    df_train = load_data_as_dataframe(train_path)
    df_test = load_data_as_dataframe(test_path)
    return df_train, df_test


stop_words = stopwords.words('english')
question_tags = ['who', 'what', 'when', 'where', 'which', 'whom', 'whose', 'why']
stop_words = [word for word in stop_words if word not in question_tags]

def preprocess_txt(text: str):
    """ Preprocess text
    Args:
        str
    Returns:
        preprocessed str
    """
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text).lower()
    text = re.sub(' +', ' ', text)
    word_list = [word for word in text.split() if word not in stop_words]
    text = " ".join(word_list)
    return text


puncts = string.punctuation
def preprocess_questions(df: pd.DataFrame):
    """ Preprocess query text of pandas Dataframe
    Args:
        df (pd.DataFrame): Dataset containing a column named question.
    Returns:
        questions (List): a list with preprocessed texts
    """
    questions = []

    # sets text to lower and remove punctuation
    for i in df.question.to_list():
        query = preprocess_txt(i)
        questions.append(query)
    
    return questions  


def create_dataset():
    """ Creates dataset from data
    Returns:
        pd.DataFrame: returns train and test datasets with preprocessed text.
    """
    df_train, df_test = get_dataframe()
    df_train = preprocess_dataframe(df_train)
    df_train['question'] = preprocess_questions(df_train)   
    
    df_test = preprocess_dataframe(df_test)
    df_test['question'] = preprocess_questions(df_test)  
    return df_train, df_test



def save_results_category(df_test: pd.DataFrame, filename:str):
    """Save results of category prediction as json 
       for test queries
    
    Args:
       df_test: pd.Dataframe containing "id" and 
       "predicted_category" columns 
       filename: str 
    """

    res = []
    for i in range(len(df_test)):
        pred_category = df_test.predicted_category.iloc[i]
        id_ = df_test.id.iloc[i]
        question = df_test.question.iloc[i]
        
        if pred_category == 'date' or pred_category =='number' or pred_category == 'string':
            res.append({
                        'id':id_, 
                        'question': question,
                        'category': 'literal', 
                        'type': [pred_category]
                    })

        elif pred_category == 'boolean':
            res.append({
                        'id':id_, 
                        'question': question,
                        'category': pred_category, 
                        'type': [pred_category]
                    })

        else:          
            res.append({
                        'id':id_, 
                        'question': question,
                        'category': pred_category, 
                        'type': ''
                    })

    f = open(f"results/{filename}.json", "w")
    json.dump(res, f)
    f.close()


def load_queries(docs):
    """Loads train and test queries for type prediction.
    
    Args: 
        A List containg the queries
    Returns:
        A dictionary containing only queries of resource category.
    """
    resource_queries = {}
    count = 0
    for x in docs:
        if x['category'] != 'resource':
            count += 1
            continue
        
        if x['question'] is not None:
            q = preprocess_txt(x['question'])
        
            doc = {
                'question': q,
                'category': x['category'],
                'type': x['type']
            } 
            resource_queries.update({x['id']:doc})
        
    return resource_queries


def save_test_results(test_res, test, title='advanced_es'):
    """Saves results from Bm25 and advanced model in the appropriate 
    format to evaluation of results.
        Args:
        test_res: output from baseline model or advanced method
        test: test queries
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
    df_train, df_test = create_dataset()
    print(df_train.head())
    

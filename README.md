# Final project DAT640 - Semantic Answer Type prediction task (Group17)

This repository contains the files for the project entitled "SeMantic AnsweR Type prediction task"

## Table of contents


* [General Info](#general-information)
* [Requirements](#requiements)
* [Data](#data)
* [How to navigate the repository](#how-to-navigate-the-repository)


## General Information

Given a query formulated in natural language, the aim is to predict the expected answer type from a set of candidate entitites from collected target ontology. 
This project uses target ontology from DBpedia 2016 dump.


## Requirements

This project requires Python >= 3.7, Elasticsearch == 7.17.6. You must have a running local Elasticsearch instance on your machine.

```
pip install --upgrade numpy pandas scikit-learn nltk elasticsearch==7.17.6
```

## Data 

The data must be downloaded separately due to its overall size.

- [short_abstracts_en.ttl](http://downloads.dbpedia.org/2016-10/core/short_abstracts_en.ttl.bz2)</br>
- [intance_types_en.ttl](http://downloads.dbpedia.org/2016-10/core/instance_types_en.ttl.bz2)</br>
- [smart_dataset_questions](https://github.com/smart-task/smart-dataset/tree/master/datasets/DBpedia)</br>
- [dbpedia_types.tsv](https://github.com/smart-task/smart-dataset/tree/master/evaluation/dbpedia)</br>

### Project structure

The project is organized as follows:

``` sh
📦dat640_smarttask_group17
 ┣ 📂datasets
 ┃ ┣ 📂DBpedia
 ┃ ┃ ┣ 📜smarttask_dbpedia_test.json  *
 ┃ ┗ ┗ 📜smarttask_dbpedia_train.json *
 ┣ 📂evaluation
 ┃ ┣ 📂dbpedia
 ┃ ┗ ┗📜dbpedia_types.tsv *
 ┣ 📂mappings
 ┃ ┣ 📜instance_types_en.ttl  *
 ┃ ┗ 📜short_abstracts_en.ttl *
 ┣ 📂results
 ┃ ┣ 📜advanced_es_system_output.json
 ┃ ┣ 📜bm25_es_system_output.json
 ┃ ┣ 📜test_queries_svm_output.json
 ┃ ┗ 📜test_type_predictions.csv
 ┣ 📜.gitignore
 ┣ 📜advanced_model.py
 ┣ 📜baseline_category_prediction.py
 ┣ 📜bm25_model.py
 ┣ 📜data_cleaning.py 
 ┣ 📜evaluate.py
 ┣ 📜indexer.ipynb
 ┣ 📜main.py
 ┣ 📜README.md
 ┗ 📜SVM.py

```

## How to navigate the repository

Before running the models it is necessary to create the Elasticsearch index. Run the `indexer.ipynb` notebook to do so. 
Run the file `main.py` to perform category and type prediction. Alternatively, the files `advanced_model.py` and `b25_model.py` can be run separately for type prediction. The description of the remaining files is provided below.

-  `data_cleaning.py` contains helper functions used to perform text preprocessing or to convert the data into a specific format used by the models.
-  `SVM.py` contains the pipeline used to perform category prediction.
-  `baseline_category_prediction.py` contains all algorithms tested to perform category prediction
-  `evaluate.py` contains the evaluation metrics implementation. This function was implemented by [Krisztian Balog](https://github.com/smart-task/smart-dataset/blob/master/evaluation/dbpedia/evaluate.py). The original script was kept, and the `eval_res` function was added at the end to perform the evaluation of results.


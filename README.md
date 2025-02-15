# External_Adaptive_Retrieval

### Description

Large Language Models (LLMs) are prone to hallucinations, and Retrieval-Augmented Generation (RAG) helps mitigate this, but at a high computational cost while risking misinformation. Adaptive retrieval aims to retrieve only when necessary, but existing approaches rely on LLM-based uncertainty estimation, which remain inefficient and impractical.
In this study, we introduce lightweight LLM-independent adaptive retrieval methods based on external information. We investigated 27 features, organized into 7 groups, and their hybrid combinations. We evaluated these methods on 6 QA datasets, assessing the QA performance and efficiency. The results show that our approach matches the performance of complex LLM-based methods while achieving significant efficiency gains, demonstrating the potential of external information for adaptive retrieval.

<img src="[https://user-images.githubusercontent.com/44041257/229779630-0b4c8588-da19-43d0-900f-00f48cf7ec37.png](https://github.com/marialysyuk/External_Adaptive_Retrieval/blob/main/flops_pic.png)" width="500"  height="300" >

### Data 

Data with all proposed features can be downloaded from [here](https://drive.google.com/file/d/1IYgrofvcw4pN681Em7NsLYf5T5bTd376/view?usp=sharing).

### External features

In the folder external features are collected notebooks with the code for their parsing.

* Graph features. First, run BELA.ipynb to find IDs of the entities from Wikidata KG. Then, with Graph_popularity.ipynb collect number of triples for where this entity is either the subject
  or an object with SPARQL queries. Collect 6 features:
  'graph_subj_mean',
    'graph_subj_min',
    'graph_subj_max',
    'graph_obj_mean',
    'graph_obj_min',
    'graph_obj_max'.
* Frequency features. First, run NER module in BELA.ipynb and find entity label from the question. Then, run Frequency.ipynb and find frequencies of entities in a large corpus
 of text (download it form [here](https://drive.google.com/file/d/1of-oUB-OKPBB9bUlnVyssxuKwKkw1fOJ/view?usp=sharing)). Collect
  4 features: 'freq_exact_mean',
    'freq_exact_min',
    'freq_exact_max',
    'freq_inexact_min'.
* Popularity features. First, run NER module in BELA.ipynb and find entity label from the question. Then, run Wikimedia_api.ipynb to collect amount of pageviews of the page with
  the corresponding label in Wikipedia. Collect 3 features: 'popularity_mean',
    'popularity_min',
    'popularity_max',
    'freq_exact_mean'.
* Question type. Collect probabilities of the nine possible types of question. First, train a simple bert-based classifier with Question_type_train.ipynb. Pretrained model can be downloaded from [here](https://drive.google.com/file/d/1MtVkqBu0_lcpWPmmwF7Ccrcyk9Q1nKp2/view?usp=sharing).Then, predict probabilities of
  nine question types with Predict_type_of_the_question_probas.ipynb. Collect 9 features:
     'prob_ordinal',
    'prob_intersection',
    'prob_superlative',
    'prob_yesno',
    'prob_comparative',
    'prob_multihop',
    'prob_difference',
    'prob_count', 'prob_generic'.
* Context relevance. Train a simple cross-encoder model to identify the relevance score for each of the retrieved context with Context_quality.ipynb. Collect 4 features: 'context_relevance_min',
    'context_relevance_max',
    'context_relevance_mean',
    'context_length'.
* Knowledgability. Prompt your model to evaluate its internal knowledgability of the entity. Example for the Llama3.1-8b Instruct can be found in Llama_knowledgability.ipynb.

### Find optimal hyperparameters

Example 
```python

python analyze_all_features_tuned_inaccuracy_long.py --data_path "data_hf/external_rag_hotpotqa_extra_v2.hf"\
                          --no_context_col "new_retriever_adaptive_rag_no_retrieve"\
                          --with_context_col "new_retriever_adaptive_rag_one_retrieve"\
                          --gt_col "reference"\
                          --seed 24
```

### Evaluate your model on the test using optimal hyperparameters

Example 
```python

python analyze_all_features_after_grid_search_voting.py --data_path "data_hf/external_rag_natural_questions_extra_v2.hf"\
                          --no_context_col "new_retriever_adaptive_rag_no_retrieve"\
                          --with_context_col "new_retriever_adaptive_rag_one_retrieve"\
                          --gt_col "reference"\
                          --data_short "natural_questions"
```

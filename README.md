# External_Adaptive_Retrieval

### Description

Large Language Models (LLMs) are prone to hallucinations, and Retrieval-Augmented Generation (RAG) helps mitigate this, but at a high computational cost while risking misinformation. Adaptive retrieval aims to retrieve only when necessary, but existing approaches rely on LLM-based uncertainty estimation, which remain inefficient and impractical.
In this study, we introduce lightweight LLM-independent adaptive retrieval methods based on external information. We investigated 27 features, organized into 7 groups, and their hybrid combinations. We evaluated these methods on 6 QA datasets, assessing the QA performance and efficiency. The results show that our approach matches the performance of complex LLM-based methods while achieving significant efficiency gains, demonstrating the potential of external information for adaptive retrieval.  

### External features

In the folder external features are collected notebooks with the code for their parsing.

* Graph features. First, run BELA.ipynb to find IDs of the entities from Wikidata KG. Then, with Graph_popularity.ipynb collect number of triples for where this entity is either the subject
  or an object with SPARQL queries. Collect six features as a consequence:
  'graph_subj_mean',
    'graph_subj_min',
    'graph_subj_max',
    'graph_obj_mean',
    'graph_obj_min',
    'graph_obj_max'.
* Frequency features. First, run NER module in BELA.ipynb and find entity label from the question. Then, run Frequency.ipynb and find frequencies of entities in a large corpus of text. Collect
  four features as a consequence: 'freq_exact_mean',
    'freq_exact_min',
    'freq_exact_max',
    'freq_inexact_min'.

from joblib import Parallel, delayed
import argparse
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from scipy.stats import spearmanr
from functools import partial
import datasets
from datasets import Dataset
from tqdm import tqdm
from utils import add_metric, need_context_new, ThresholdOptimizerClassifier, add_all_context, add_no_context, add_ideal, calc_stats, unc_based_KM
from eval_utils import has_answer, EM_compute, F1_compute
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from itertools import product
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


np.set_printoptions(suppress=True, precision=3)

import warnings
warnings.filterwarnings("ignore")

from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

# Define classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(
        class_weight={0: 1, 1: 1}, max_iter=15000
    ),
    "KNN": KNeighborsClassifier(n_neighbors=15),
    "MLP": MLPClassifier(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(max_depth=3),
    "CatBoosting": CatBoostClassifier(
    iterations=10, 
    learning_rate=0.1,
    random_seed = 13), 
    "SkBoosting": GradientBoostingClassifier(random_state = 13),
    "RandomForest":  RandomForestClassifier(random_state = 13)
}

metric_names = ['popularity_mean',
    'popularity_min',
    'popularity_max',
    'freq_exact_mean',
    'freq_exact_min',
    'freq_exact_max',
    'freq_inexact_min',
    'graph_subj_mean',
    'graph_subj_min',
    'graph_subj_max',
    'graph_obj_mean',
    'graph_obj_min',
    'graph_obj_max',
    "MeanTokenEntropy",
    "MaxTokenEntropy",
    "SAR",
    "EigValLaplacian_NLI_score_entail",
    "LexicalSimilarity_rougeL",
    'llama_know',        
    'prob_ordinal',
    'prob_intersection',
    'prob_superlative',
    'prob_yesno',
    'prob_comparative',
    'prob_multihop',
    'prob_difference',
    'prob_count',
    'is_complex',
    'context_relevance_min',
    'context_relevance_max',
    'context_relevance_mean',
    'context_length',
    'count_2019',
    'count_2021']


metrics_names_uncertainty = [
    "MeanTokenEntropy",
    "MaxTokenEntropy",
    "SAR",
    "EigValLaplacian_NLI_score_entail",
    "LexicalSimilarity_rougeL",
]


metrics_names_graph = [
    'graph_subj_mean',
    'graph_subj_min',
    'graph_subj_max',
    'graph_obj_mean',
    'graph_obj_min',
    'graph_obj_max']


metrics_knowledgability = [
    'llama_know',
]
                  
metrics_ques_complexity = [
    # 'prob_generic',
    'prob_ordinal',
    'prob_intersection',
    'prob_superlative',
    'prob_yesno',
    'prob_comparative',
    'prob_multihop',
    'prob_difference',
    'prob_count']

metrics_justcomplex = [
  'is_complex']

metrics_green = [
    'count_2019',
    'count_2021']

metrics_context = [
    'context_relevance_min',
    'context_relevance_max',
    'context_relevance_mean',
    'context_length']

metrics_popularity = [
    'popularity_mean',
    'popularity_min',
    'popularity_max'
]

metrics_frequency = [
    'freq_exact_mean',
    'freq_exact_min',
    'freq_exact_max',
    'freq_inexact_min'
]

metrics_external_graph = [
    'graph_subj_mean',
    'graph_subj_min',
    'graph_subj_max',
    'graph_obj_mean',
    'graph_obj_min',
    'graph_obj_max',
    'llama_know',
    'prob_ordinal',
    'prob_intersection',
    'prob_superlative',
    'prob_yesno',
    'prob_comparative',
    'prob_multihop',
    'prob_difference',
    'prob_count',
    'is_complex',
    'context_relevance_min',
    'context_relevance_max',
    'context_relevance_mean',
    'context_length']

metrics_external_all = [
    'popularity_mean',
    'popularity_min',
    'popularity_max',
    'freq_exact_mean',
    'freq_exact_min',
    'freq_exact_max',
    'freq_inexact_min',
    'graph_subj_mean',
    'graph_subj_min',
    'graph_subj_max',
    'graph_obj_mean',
    'graph_obj_min',
    'graph_obj_max',
    'llama_know',
    'prob_ordinal',
    'prob_intersection',
    'prob_superlative',
    'prob_yesno',
    'prob_comparative',
    'prob_multihop',
    'prob_difference',
    'prob_count',
    'is_complex',
    'context_relevance_min',
    'context_relevance_max',
    'context_relevance_mean',
    'context_length']

metrics_all = [
    'popularity_mean',
    'popularity_min',
    'popularity_max',
    'freq_exact_mean',
    'freq_exact_min',
    'freq_exact_max',
    'freq_inexact_min',
    'graph_subj_mean',
    'graph_subj_min',
    'graph_subj_max',
    'graph_obj_mean',
    'graph_obj_min',
    'graph_obj_max',
    "MeanTokenEntropy",
    "MaxTokenEntropy",
    "SAR",
    "EigValLaplacian_NLI_score_entail",
    "LexicalSimilarity_rougeL",
    'llama_know',
    'prob_ordinal',
    'prob_intersection',
    'prob_superlative',
    'prob_yesno',
    'prob_comparative',
    'prob_multihop',
    'prob_difference',
    'prob_count',
    'is_complex',
    'context_relevance_min',
    'context_relevance_max',
    'context_relevance_mean',
    'context_length']
                   
metrics_external_graph_unc = [
    "MeanTokenEntropy",
    "MaxTokenEntropy",
    "SAR",
    "EigValLaplacian_NLI_score_entail",
    "LexicalSimilarity_rougeL",
    'graph_subj_mean',
    'graph_subj_min',
    'graph_subj_max',
    'graph_obj_mean',
    'graph_obj_min',
    'graph_obj_max',
    'llama_know',
    'prob_ordinal',
    'prob_intersection',
    'prob_superlative',
    'prob_yesno',
    'prob_comparative',
    'prob_multihop',
    'prob_difference',
    'prob_count',
    'is_complex',
    'context_relevance_min',
    'context_relevance_max',
    'context_relevance_mean',
    'context_length']


metrics_external_graph_unc_wocontext = [
    "MeanTokenEntropy",
    "MaxTokenEntropy",
    "SAR",
    "EigValLaplacian_NLI_score_entail",
    "LexicalSimilarity_rougeL",
    'graph_subj_mean',
    'graph_subj_min',
    'graph_subj_max',
    'graph_obj_mean',
    'graph_obj_min',
    'graph_obj_max',
    'llama_know',
    'prob_ordinal',
    'prob_intersection',
    'prob_superlative',
    'prob_yesno',
    'prob_comparative',
    'prob_multihop',
    'prob_difference',
    'prob_count',
    'is_complex']

metrics_external_wollama = [
    "MeanTokenEntropy",
    "MaxTokenEntropy",
    "SAR",
    "EigValLaplacian_NLI_score_entail",
    "LexicalSimilarity_rougeL",
    'graph_subj_mean',
    'graph_subj_min',
    'graph_subj_max',
    'graph_obj_mean',
    'graph_obj_min',
    'graph_obj_max',
    'prob_ordinal',
    'prob_intersection',
    'prob_superlative',
    'prob_yesno',
    'prob_comparative',
    'prob_multihop',
    'prob_difference',
    'prob_count',
    'is_complex',
    'context_relevance_min',
    'context_relevance_max',
    'context_relevance_mean',
    'context_length']


def save_results_to_file(results, file_path, headers):
    with open(file_path, "w") as f:
        # Write header
        f.write(tabulate(results, headers=headers, tablefmt="grid"))
        f.write("\n")


def train_and_predict(
    X_train, y_train, X_test, y_test, col, ds_test, classifier, classifier_name, with_context_name, no_context_name
):
    X_train = np.nan_to_num(X_train, nan=0)
    X_test = np.nan_to_num(X_test, nan=0)
    corr = 0 if col == "hybrid" else spearmanr(X_test, y_test)[0]

    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    if classifier_name == "KNN":

        params = {'n_neighbors':  [5, 7, 9, 11, 13, 15],
                   'metric': ['euclidean', 'manhattan'],
                   'algorithm': ['auto', 'ball_tree', 'kd_tree'],
         'weights': ['uniform', 'distance']}
        
        param_combinations = list(product(*params.values()))

        general_metrics = []

        for param_values in param_combinations:
            params = dict(zip(params.keys(), param_values))
            model = KNeighborsClassifier(**params)
            model.fit(X_train, y_train)
            
            # Validate the model
            y_pred = model.predict(X_test)

            final_accuracy, mean_tokens, em, f1 = unc_based_KM(
                ds_test.add_column(f"pred_{col}", list(y_pred)),
                with_context_name, no_context_name, f"pred_{col}")
            general_metrics.append([col, classifier_name, final_accuracy, params])
            

        
    if classifier_name == "Logistic Regression":
        params = {'C': [0.01, 0.1, 1],
                  'solver': ['lbfgs','liblinear'],
                  'class_weight': ['balanced',  {0: 1, 1: 1}, None],
                  'max_iter': [10000, 15000, 20000]}
    
        param_combinations = list(product(*params.values()))

        general_metrics = []
    
        
        for param_values in param_combinations:
            params = dict(zip(params.keys(), param_values))
            params['class_weight'] = 'balanced'
            model = LogisticRegression(**params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            
            final_accuracy, mean_tokens, em, f1 = unc_based_KM(
                ds_test.add_column(f"pred_{col}", list(y_pred)),
                with_context_name, no_context_name, f"pred_{col}")
            general_metrics.append([col, classifier_name, final_accuracy, params])
            
        
    if classifier_name == "MLP":
        params = {'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                  'activation': ['relu', 'tanh'],
                  'solver': ['adam', 'sgd'],
                  'alpha': [0.00001, 0.0001, 0.001, 0.01],
                  'learning_rate': ['constant', 'adaptive'],
                  'max_iter': [200, 500]}
        
        param_combinations = list(product(*params.values()))
       
        general_metrics = []

        for param_values in param_combinations:
            params = dict(zip(params.keys(), param_values))
            params['random_state'] = 13
            params['early_stopping'] = True
            model =  MLPClassifier(**params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            
            # Validate the model
            y_pred = model.predict(X_test)

            final_accuracy, mean_tokens, em, f1 = unc_based_KM(
                ds_test.add_column(f"pred_{col}", list(y_pred)),
                with_context_name, no_context_name, f"pred_{col}")
            general_metrics.append([col, classifier_name, final_accuracy, params])
  
    if classifier_name == "Decision Tree":
        params = {'max_depth': [3, 5, 7, 10, None],
                  'max_features': [0.2, 0.4, 'sqrt', 'log2', None],
                  'criterion': ['gini', 'entropy'],
                  'splitter': ['best', 'random']}
        param_combinations = list(product(*params.values()))
        
        general_metrics = []

        for param_values in param_combinations:
            params = dict(zip(params.keys(), param_values))
            params['random_state'] = 13
            params['class_weight'] = 'balanced'
            model = DecisionTreeClassifier(**params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
                
            y_pred = model.predict(X_test)

            final_accuracy, mean_tokens, em, f1 = unc_based_KM(
                ds_test.add_column(f"pred_{col}", list(y_pred)),
                with_context_name, no_context_name, f"pred_{col}")
            general_metrics.append([col, classifier_name, final_accuracy, params])
                            
                              
    if classifier_name ==  "RandomForest":
        params = {
            'n_estimators': [25, 35, 50],
            'max_depth': [3, 5, 7, 9, 11],
            'max_features': [0.2, 0.4, 'sqrt', 'log2', None],
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy'],
            'class_weight': ['balanced', {0: 1, 1: 1}, None]
        }

        param_combinations = list(product(*params.values()))
       
        general_metrics = []
        
        for param_values in param_combinations:
            params = dict(zip(params.keys(), param_values))
            params['random_state'] = 13
            params['class_weight'] = 'balanced'
            model =  RandomForestClassifier(**params)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)

            final_accuracy, mean_tokens, em, f1 = unc_based_KM(
                ds_test.add_column(f"pred_{col}", list(y_pred)),
                with_context_name, no_context_name, f"pred_{col}")
            general_metrics.append([col, classifier_name, final_accuracy, params])
       
        
    if classifier_name == "SkBoosting":
        params = {
            'n_estimators': [25, 35, 50],
            'learning_rate': [0.001, 0.01, 0.05],
            'max_depth': [3, 4, 5, 7, 9],
            'max_features': [0.2, 0.4, 'sqrt', 'log2', None]}
        
        param_combinations = list(product(*params.values()))
        
        general_metrics = []

        for param_values in param_combinations:
            params = dict(zip(params.keys(), param_values))
            params['random_state'] = 13
            model =  GradientBoostingClassifier(**params)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")   
                model =  GradientBoostingClassifier(**params)
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            final_accuracy, mean_tokens, em, f1 = unc_based_KM(
                ds_test.add_column(f"pred_{col}", list(y_pred)),
                with_context_name, no_context_name, f"pred_{col}")
            general_metrics.append([col, classifier_name, final_accuracy, params])
        
       
        
    if classifier_name == "CatBoosting":
        params = {
            'iterations': [10, 50, 100, 200],
            'learning_rate': [0.001, 0.01, 0.05],
            'depth': [3, 4, 5, 7, 9],
            'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS']}
        
        param_combinations = list(product(*params.values()))
        general_metrics = []
        
        for param_values in param_combinations:
            params = dict(zip(params.keys(), param_values))
            params['random_seed'] = 13
            params['silent'] = True
            model =  CatBoostClassifier(**params)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train, early_stopping_rounds=10)

            y_pred = model.predict(X_test)

            final_accuracy, mean_tokens, em, f1 = unc_based_KM(
                ds_test.add_column(f"pred_{col}", list(y_pred)),
                with_context_name, no_context_name, f"pred_{col}")
            general_metrics.append([col, classifier_name, final_accuracy, params])
            

        
    return general_metrics


def main(args):
    ds = datasets.load_from_disk(args.data_path)
    with_context_name, no_context_name, gt_name = (
        args.with_context_col,
        args.no_context_col,
        args.gt_col,
    )

    desired_metrics = [(has_answer, 'InAccuracy'), (EM_compute, 'EM'), (F1_compute, 'F1')]
    for metric_func, metric_name in desired_metrics:
        key_args = {
            'gt_col': gt_name,
            'metric': metric_func,
            'metric_name': metric_name
        }
        ds = ds.map(partial(add_metric, target_col=with_context_name, **key_args))
        ds = ds.map(partial(add_metric, target_col=no_context_name, **key_args))


    ds = ds.map(
        partial(
            need_context_new,
            with_context_col=with_context_name,
            without_context_col=no_context_name,
            need_context_col="gt_need_retrieval",
        )
    )
    ds = ds.map(partial(calc_stats, col_name='TokenEntropy'))
    ds = ds.map(partial(calc_stats, col_name='MaximumTokenProbability'))

    y_train, y_test = np.array(ds["train"]["gt_need_retrieval"]), np.array(
        ds["test"]["gt_need_retrieval"]
    )
    
    train_data = pd.DataFrame(
        {
            name: ds["train"][name]
            for name in metric_names
            if (
                (name in ds["train"].column_names) and (name in ds["test"].column_names)
            )
        }
    )

    
    
    scaler = StandardScaler().fit(train_data.values)
    scaled_features_train = scaler.transform(train_data.values)
    train_data = pd.DataFrame(scaled_features_train, index=train_data.index, columns=train_data.columns)


    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    train_data, ds["train"]["gt_need_retrieval"], test_size=0.2, random_state=args.seed, stratify = ds["train"]["gt_need_retrieval"])
    test_ind = X_train_new.index
    hf_test = Dataset.from_pandas(pd.DataFrame(ds['train']).loc[X_test_new.index])
    general_results = []
    
    def run_all_classifiers_for_feature(cols, cols_desc):
        X_train, X_test =  X_train_new[cols].values, X_test_new[cols].values
        return Parallel(n_jobs=-1)(
            delayed(train_and_predict)(
                X_train, np.array(y_train_new), X_test, np.array(y_test_new), cols_desc, hf_test, clf, clf_name, with_context_name, no_context_name
            )
            for clf_name, clf in classifiers.items()
        )

    groups_cols = [
        metrics_names_uncertainty,
                   metrics_names_graph,
                   metrics_popularity,
                   metrics_frequency,
                   metrics_knowledgability,
                   metrics_ques_complexity,
                   metrics_justcomplex,
                   metrics_context,
                   metrics_external_all,
                   metrics_all,
                   metrics_external_graph,
                   metrics_external_graph_unc,
    ]



    groups_cols_lbls = [
        "uncertainty", "graph", "popularity", "frequency",
        "knowledgability", "ques_complexity", "justcomplex",
        "context", "external_all", "all", "external_graph", "external_graph_unc"
    ]



                        
    #feature groups
    
    with tqdm(
        total=len(groups_cols) * len(classifiers), desc="Processing features"
    ) as pbar:
        for i in range(len(groups_cols)):
            print(i)
            cur_cols = groups_cols[i]
            cur_cols_lbls = groups_cols_lbls[i]
            print(cur_cols_lbls)
            classifier_results = run_all_classifiers_for_feature(cur_cols, cur_cols_lbls)
            for q in range(len(classifier_results)):
                for m in range(len(classifier_results[q])):
                    general_results.append(classifier_results[q][m])
            pbar.update(len(classifiers))

    general_headers = ["Feature", "Classifier", "Best_score", "Best_params"]
    general_results = sorted(general_results, key=lambda x: x[2], reverse = True)

    gen_res_df = pd.DataFrame(general_results, columns = general_headers) 
   
      # All Context and No Context rows
    detailed_metrics, general_metrics = add_all_context(hf_test, without_context_col=no_context_name, with_context_col=with_context_name)

    print("ALL contex", general_metrics)

    detailed_metrics, general_metrics = add_no_context(hf_test, without_context_col=no_context_name, with_context_col=with_context_name)
    print("NO contex", general_metrics)

    # Ideal
    detailed_metrics, general_metrics = add_ideal(hf_test, without_context_col=no_context_name, with_context_col=with_context_name)

    print("IDEAL", general_metrics)

    
    # Save detailed and general results to respective files
    data_name = args.data_path.split("/")[-1]
    gen_res_df.to_csv(f"logs_grid_search_inacc/{data_name}_groups_new_llama_seed{args.seed}.csv", index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run uncertainty estimations for transformer models."
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the Hugging Face dataset"
    )
    parser.add_argument(
        "--no_context_col", type=str, required=True, help="Column name for questions"
    )
    parser.add_argument(
        "--with_context_col", type=str, help="Optional column name for context"
    )
    parser.add_argument(
        "--gt_col", type=str, help="Optional column name for output (answers)"
    )

    parser.add_argument(
        "--seed", type=int, required=True, help="Seed for holdout"
    )
    args = parser.parse_args()

    main(args)

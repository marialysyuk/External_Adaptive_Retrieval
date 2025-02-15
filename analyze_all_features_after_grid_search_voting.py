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
from tqdm import tqdm
from utils import add_metric, need_context_new, ThresholdOptimizerClassifier, add_all_context, add_no_context, add_ideal, calc_stats, unc_based_KM
from eval_utils import has_answer, EM_compute, F1_compute
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
import ast
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
    "KNN": KNeighborsClassifier(algorithm= 'auto', leaf_size = 5, metric= 'manhattan', n_neighbors = 15, weights = 'uniform'),
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
    'context_length']


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
    'llama_know'
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

metrics_external_graph = ['graph_subj_mean',
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

metrics_external_frequency = ['freq_exact_mean',
    'freq_exact_min',
    'freq_exact_max',
    'freq_inexact_min',
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

metrics_external_popularity = ['popularity_mean',
    'popularity_min',
    'popularity_max',
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
                  

metrics_external_all = ['popularity_mean',
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

metrics_all = ['popularity_mean',
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
                   
metrics_external_graph_unc = ["MeanTokenEntropy",
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

metrics_external_frequency_unc = [
    "MeanTokenEntropy",
    "MaxTokenEntropy",
    "SAR",
    "EigValLaplacian_NLI_score_entail",
    "LexicalSimilarity_rougeL",
    'freq_exact_mean',
    'freq_exact_min',
    'freq_exact_max',
    'freq_inexact_min',
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

metrics_external_popularity_unc = [
    "MeanTokenEntropy",
    "MaxTokenEntropy",
    "SAR",
    "EigValLaplacian_NLI_score_entail",
    "LexicalSimilarity_rougeL",
    'popularity_mean',
    'popularity_min',
    'popularity_max',
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


def save_results_to_file(results, file_path, headers):
    with open(file_path, "w") as f:
        # Write header
        f.write(tabulate(results, headers=headers, tablefmt="grid"))
        f.write("\n")



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

    test_data = pd.DataFrame(
        {
            name: ds["test"][name]
            for name in metric_names
            if (
                (name in ds["train"].column_names) and (name in ds["test"].column_names)
            )
        }
    )

    scaler = StandardScaler().fit(train_data.values)
    scaled_features_train = scaler.transform(train_data.values)
    train_data = pd.DataFrame(scaled_features_train, index=train_data.index, columns=train_data.columns)
    scaled_features_test = scaler.transform(test_data.values)
    test_data = pd.DataFrame(scaled_features_test, index=test_data.index, columns=test_data.columns)

    detailed_results = []
    general_results = []


    def train_and_predict(
        X_train, y_train, X_test, y_test, col, ds_test, classifier, classifier_name, with_context_name, no_context_name
    ):
        X_train = np.nan_to_num(X_train, nan=0)
        X_test = np.nan_to_num(X_test, nan=0)
        corr = 0 if col == "hybrid" else spearmanr(X_test, y_test)[0]
        print(classifier_name, classifier)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_pred_prob = (
            classifier.predict_proba(X_test)[:, 1]
            if hasattr(classifier, "predict_proba")
            else y_pred
        )

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = (
            roc_auc_score(y_test, y_pred_prob)
            if hasattr(classifier, "predict_proba")
            else np.nan
        )
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrix = conf_matrix / np.sum(conf_matrix)
        final_accuracy, mean_tokens, em, f1 = unc_based_KM(
            ds_test.add_column(f"pred_{col}", list(y_pred)),
            with_context_name,
            no_context_name,
            f"pred_{col}",
        )
    
        retrieval_calls = np.mean(y_pred)
    
        detailed_metrics = [
            col,
            classifier_name,
            np.around(roc_auc, 3),
            np.around(accuracy, 3),
            np.around(corr, 3),
            conf_matrix.tolist(),
            np.around(final_accuracy, 3),
            np.around(em, 3),
            np.around(f1, 3),
            np.around(mean_tokens, 1),
            np.around(retrieval_calls, 2)
        ]
        general_metrics = [
            col,
            classifier_name,
            np.around(final_accuracy, 3),
            np.around(retrieval_calls, 2),
            np.around(roc_auc, 3),
            np.around(em, 3),
            np.around(f1, 3),
            np.around(mean_tokens, 1),
           
        ]
    
        return general_metrics


    groups_cols = [metrics_names_uncertainty,  
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
                  metrics_external_graph_unc]


    groups_cols_lbls = ["uncertainty", "graph", "popularity", "frequency",
                        "knowledgability", "ques_complexity", "justcomplex", "context",
                       "external_all", "all", "external_graph", "external_graph_unc"]

    





    seed9 = pd.read_csv("logs_grid_search_inacc/external_rag_"+args.data_short+"_extra_v2.hf_groups_seed9.csv")
    seed2 = pd.read_csv("logs_grid_search_inacc/external_rag_"+args.data_short+"_extra_v2.hf_groups_seed2.csv")
    seed24 = pd.read_csv("logs_grid_search_inacc/external_rag_"+args.data_short+"_extra_v2.hf_groups_seed24.csv")

    
    seed24 = seed24.rename(columns ={"Best_score": "Best_score_24"})
    seed2 = seed2.rename(columns ={"Best_score": "Best_score_2"})
    seed9 = seed9.rename(columns ={"Best_score": "Best_score_9"})
    
    seed24['rank_24'] = seed24.index+1
    seed2['rank_2'] = seed2.index+1
    seed9['rank_9'] = seed9.index+1
    
    joined_df = seed24.merge(seed9, on = ["Feature", "Classifier", "Best_params"])
    joined_df = joined_df.merge(seed2, on = ["Feature", "Classifier", "Best_params"])
    # joined_df = seed24
    joined_df['Best_score_mean'] = joined_df[['Best_score_24', 'Best_score_2', "Best_score_9"]].mean(axis=1)
    joined_df['Mean_rank'] = joined_df[['rank_24', 'rank_2', 'rank_9']].mean(axis=1)

    
    with tqdm(
        total=len(groups_cols), desc="Processing features"
    ) as pbar:
        for i in range(len(groups_cols)):
            
            cur_cols = groups_cols[i]
            cur_cols_lbls = groups_cols_lbls[i]
            X_train, X_test = train_data[cur_cols].values, test_data[cur_cols].values
            
            def return_clf_function(cur_algo, cur_params):
    
                if cur_algo == "KNN":
                    clf_name = KNeighborsClassifier(**cur_params)
                if cur_algo == "Logistic Regression":
                    clf_name = LogisticRegression(**cur_params)
                if cur_algo == "MLP":
                    cur_params['random_state']= 13
                    cur_params['early_stopping'] = True
                    clf_name = MLPClassifier(**cur_params)
                if cur_algo == "Decision Tree":
                    cur_params['random_state']= 13
                    clf_name = DecisionTreeClassifier(**cur_params)
                if cur_algo ==  "RandomForest":
                    cur_params['random_state']= 13
                    clf_name = RandomForestClassifier(**cur_params)
                if cur_algo == "SkBoosting":
                    cur_params['random_state']= 13
                    clf_name = GradientBoostingClassifier(**cur_params)
                if cur_algo == "CatBoosting":
                    cur_params['random_seed']= 13
                    cur_params['silent']= True
                    clf_name = CatBoostClassifier(**cur_params)
                return clf_name


            ind_logreg = joined_df.query("Feature == @cur_cols_lbls").sort_values(by=['Best_score_mean', 'Mean_rank'], 
                                                    ascending=[False, True]).query("Classifier == 'Logistic Regression'").index[0]
            ind_RF = joined_df.query("Feature == @cur_cols_lbls").sort_values(by=['Best_score_mean', 'Mean_rank'], 
                                                                ascending=[False, True]).query("Classifier == 'RandomForest'").index[0]
            ind_DT = joined_df.query("Feature == @cur_cols_lbls").sort_values(by=['Best_score_mean', 'Mean_rank'], 
                                                                ascending=[False, True]).query("Classifier == 'Decision Tree'").index[0]
            ind_MLP = joined_df.query("Feature == @cur_cols_lbls").sort_values(by=['Best_score_mean', 'Mean_rank'], 
                                                                ascending=[False, True]).query("Classifier == 'MLP'").index[0]
            ind_CatBoost = joined_df.query("Feature == @cur_cols_lbls").sort_values(by=['Best_score_mean', 'Mean_rank'], 
                                                                ascending=[False, True]).query("Classifier == 'CatBoosting'").index[0]
            ind_SkBoost = joined_df.query("Feature == @cur_cols_lbls").sort_values(by=['Best_score_mean', 'Mean_rank'], 
                                                                ascending=[False, True]).query("Classifier == 'SkBoosting'").index[0]
            ind_KNN = joined_df.query("Feature == @cur_cols_lbls").sort_values(by=['Best_score_mean', 'Mean_rank'], 
                                                                ascending=[False, True]).query("Classifier == 'KNN'").index[0]
            inds_algos = [ind_logreg, ind_RF, ind_DT, ind_MLP, ind_CatBoost, ind_SkBoost, ind_KNN]
            non_trees_algos =  [ind_logreg, ind_MLP, ind_KNN]
            joined_df_cur = joined_df.loc[inds_algos].reset_index(drop=True)
            joined_df_non_tree = joined_df.loc[non_trees_algos].reset_index(drop=True)

            clf1, clf2, clf3, clf4, clf5 = list(joined_df_cur.query("Feature == @cur_cols_lbls").sort_values(by=['Best_score_mean', 'Mean_rank'], 
                                                                         ascending=[False, True])[:5].Classifier.values)
            params1, params2, params3, params4, params5 = list(joined_df_cur.query("Feature == @cur_cols_lbls").sort_values(
                by=['Best_score_mean','Mean_rank'], ascending=[False, True])[:5].Best_params.values)

            params1, params2, params3, params4, params5 = ast.literal_eval(params1), ast.literal_eval(params2), ast.literal_eval(params3), ast.literal_eval(params4), ast.literal_eval(params5)

        
            clf_name1 = return_clf_function(clf1, params1)
            clf_name2 = return_clf_function(clf2, params2)

            
            from sklearn.ensemble import StackingClassifier, VotingClassifier
            estimators = [('cl1', clf_name1), ('cl2', clf_name2)]
            clf_stack = VotingClassifier(
                estimators=estimators, voting= 'soft')
            cur_algo = clf1+'_'+clf2

            classifier_results = train_and_predict(
         X_train, y_train, X_test, y_test, cur_cols_lbls, ds["test"], clf_stack, cur_algo, with_context_name, no_context_name)
            general_results.append(classifier_results)
            pbar.update(len(groups_cols))

    
    # All Context and No Context rows
    detailed_metrics, general_metrics = add_all_context(ds['test'], without_context_col=no_context_name, with_context_col=with_context_name)
    general_results.append(general_metrics)
    detailed_results.append(detailed_metrics)

    detailed_metrics, general_metrics = add_no_context(ds['test'], without_context_col=no_context_name, with_context_col=with_context_name)
    general_results.append(general_metrics)
    detailed_results.append(detailed_metrics)
    

    # Ideal
    detailed_metrics, general_metrics = add_ideal(ds['test'], without_context_col=no_context_name, with_context_col=with_context_name)
    general_results.append(general_metrics)
    detailed_results.append(detailed_metrics)


    general_headers = ["Feature", "Classifier", "In-Accuracy", "Retrieval Calls", "ROC-AUC", "EM", "F1", "Mean Tokens",]
    print(general_results)
    general_results = sorted(general_results, key=lambda x: x[2], reverse = True)

    # Save detailed and general results to respective files
    data_name = args.data_path.split("/")[-1]
    save_results_to_file(
        general_results, f"logs_after_grid_search/{data_name}_valid_1holdout_stacking_2algo_remove_corr.log", general_headers
    )


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
        "--data_short", type=str, help="Short name of the dataset"
    )
    args = parser.parse_args()

    main(args)

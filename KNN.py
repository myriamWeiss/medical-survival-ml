from dataclasses import dataclass
from ucimlrepo import fetch_ucirepo 
import sweetviz as sv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
import numpy as np
import pandas as pd
from itertools import combinations
from LAMDA_SSL.Algorithm.Classification.TSVM import TSVM
from LAMDA_SSL.Dataset.LabeledDataset import LabeledDataset
from LAMDA_SSL.Dataset.UnlabeledDataset import UnlabeledDataset
from LAMDA_SSL.Evaluation.Classifier.Recall import Recall
from LAMDA_SSL.Evaluation.Classifier.Accuracy import Accuracy
from LAMDA_SSL.Evaluation.Classifier.Precision import Precision
from LAMDA_SSL.Evaluation.Classifier.AUC import AUC
from LAMDA_SSL.Evaluation.Classifier.Confusion_Matrix import Confusion_Matrix
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import train_test_split
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
import os
from main import Data, Observations, get_data, split_data



#--------------------------KNN--------------------------------#
knn_possibility = {
    "metrics": ["euclidean", "manhattan"],
    "weights": ["uniform", "distance"],
    "neighbors": [5,10,15]#[2, 4,7]#[4, 5, 6]
}

#-----------------KNN learning vs regular ---------------------
def test_self_learning_vs_regular(data, metric, weight, neighbor) -> pd.DataFrame:
    list_result = [
        (metric, weight, neighbor, *get_result_of_regular_knn_combination(data, metric, weight, neighbor)),
        (metric, weight, neighbor, *get_result_of_self_train_knn_combination(data, metric, weight, neighbor))
    ]
    return knn_results_to_df(list_result)

def get_result_of_regular_knn_combination(data: Data, metric, weight, neighbor):
    model = KNeighborsClassifier(metric=metric, weights=weight, n_neighbors=neighbor)
    model.fit(np.concatenate([data.labeled.X, data.unlabeled.X]), np.concatenate([data.labeled.y, data.unlabeled.y]))
    y_test_pred = model.predict(data.test.X)
    test_acc = accuracy_score(data.test.y, y_test_pred)
    unlabeled_acc = 1  # Y are given for unlabeled
    report = classification_report(data.test.y, y_test_pred, output_dict=True)
    return test_acc, unlabeled_acc, report

#-----------------KNN Self Learning ---------------------

def test_self_learning_knn(data: Data ) -> pd.DataFrame:
    list_result = []
    for metric, weight, neighbor in product(knn_possibility["metrics"], knn_possibility["weights"], knn_possibility["neighbors"]):
        test_acc, unlabeled_acc, report = get_result_of_self_train_knn_combination(data, metric, weight, neighbor)
        list_result.append((metric, weight, neighbor, test_acc, unlabeled_acc, report))
    return knn_results_to_df(list_result)


def get_result_of_self_train_knn_combination(data: Data, metric, weight, neighbor):
    # train
    model = train_self_learning_knn(data, metric, weight, neighbor)
    # test
    y_test_pred = model.predict(data.test.X)
    test_acc = accuracy_score(data.test.y, y_test_pred)
    # unlabled from train
    y_unlabeled_pred = model.transduction_[len(data.labeled.y):]
    unlabeled_acc = accuracy_score(data.unlabeled.y, y_unlabeled_pred)

    report = classification_report(data.test.y, y_test_pred, output_dict=True)
    return test_acc, unlabeled_acc , report

def train_self_learning_knn(data: Data, metric, weight, neighbor): #done
    base_knn = KNeighborsClassifier(metric=metric, weights= weight, n_neighbors= neighbor)
    self_training_model = SelfTrainingClassifier(base_estimator=base_knn)
    y_combined = np.concatenate([data.labeled.y, [-1] * len(data.unlabeled.X)])  #-1 for the unlabeled y
    X_combined = np.concatenate([data.labeled.X, data.unlabeled.X])
    self_training_model.fit(X_combined, y_combined)
    return self_training_model

def knn_results_to_df(results_knn) -> pd.DataFrame:
    return pd.DataFrame([
        {'metric': metric, 'weight': weight, 'n_neighbors': neighbor, 'test_accuracy': test_acc, 'unlabeled_accuracy': unlabeled_acc, **flatten_report(report)}
        for metric, weight, neighbor, test_acc, unlabeled_acc, report in results_knn
    ])

#-------------------------Analyze& Graph KNN SL -----------------
def save_mean_accuracy_to_excel():
    result_path = "./KNN result/knn_results.xlsx"
    result_list = mean_accuracy_for_seed(result_path)
    df = pd.DataFrame(result_list, columns= ["Seed", "mean_test_accuracy",  "mean_unlabeled_accuracy"])
    df.to_excel("./KNN result/Accuracy_mean.xlsx", index=False)

def mean_accuracy_for_seed(path) -> list:
    means_over_seed = []
    list_seed = [42, 65, 1, 13,976, 125, 32, 861, 10, 24]
    for seed in list_seed:
        name_sheet = f"Seed {seed}"
        df = pd.read_excel(path, sheet_name= name_sheet)
        mean_test_accuracy = df["test_accuracy"].mean()
        mean_unlabeled_accuracy = df["unlabeled_accuracy"].mean()
        means_over_seed.append((seed, mean_test_accuracy, mean_unlabeled_accuracy))
    return means_over_seed


def make_graphs_from_result():
    path = "./KNN result/knn_results.xlsx"
    list_seed = [42, 65, 1, 13,976, 125, 32, 861, 10, 24]
    for seed in list_seed:
        name_sheet = f"Seed {seed}"
        df = pd.read_excel(path, sheet_name= name_sheet)
        plot_self_learning_knn_results(df,seed, "./image")

   

def plot_self_learning_knn_results(df_results, seed, save_path):
    """
    Plot accuracy vs. n_neighbors for each combination of metric and weight.
    :param df_results: DataFrame from test_self_learning_knn
    """
    df_results['combination'] = df_results['metric'] + "_" + df_results['weight']

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_results,
        x="n_neighbors",
        y="test_accuracy",
        hue="combination",
        marker="o"
    )

    plt.title("Self-Learning KNN Accuracy by n_neighbors")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Test Accuracy")
    plt.legend(title="Distance + Weight")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        seed_part = f"_seed{seed}" if seed is not None else ""
        filename = f"knn_accuracy{seed_part}.png"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=300)
        print(f"✅ Saved plot to: {full_path}")
              
    plt.show()


def flatten_report(report_dict):
    flat = {}
    for key, value in report_dict.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat[f'{key}_{subkey}'] = subvalue
        else:
            flat[key] = value
    return flat


#---------------------------------RUN---------------------------
def compare_train_possibilities():
    X, y = get_data()
    with pd.ExcelWriter('./KNN result/knn_results.xlsx') as writer:
        for seed in [42, 65, 1, 13,976, 125, 32, 861, 10, 24]:
            data = split_data(X, y, seed)
            result_df = test_self_learning_knn(data)
            #plot_self_learning_knn_results(result_df)
            result_df.to_excel(writer, sheet_name=f"Seed {seed}", index=False)    
            print(f"finish: {seed}")



def compare_self_train_vs_regular():
    best_metric = "euclidean"
    best_weight = 'distance'
    best_neighbor = 15
    X, y = get_data()
    with pd.ExcelWriter('KNN result/knn_regular_vs_train.xlsx') as writer:
        for seed in [42, 65, 1, 13,976, 125, 32, 861, 10, 24]:
            data = split_data(X, y, seed)
            result_df = test_self_learning_vs_regular(data, best_metric, best_weight, best_neighbor)
            result_df.to_excel(writer, sheet_name=f"Seed {seed}", index=False)


def run():
    compare_train_possibilities()
    make_graphs_from_result()
    save_mean_accuracy_to_excel()
    compare_self_train_vs_regular()


if __name__ == "__main__":
    run()
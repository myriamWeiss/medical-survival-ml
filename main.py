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
import time

@dataclass
class Observations:
    X: pd.DataFrame
    y: pd.Series


@dataclass
class Data:
    labeled: Observations
    unlabeled: Observations
    test: Observations


#-----------------------Excel---------------------------------

def export_results_to_excel(results_ova, results_ovo, filename='tsvm_results.xlsx'):
    # המרה לרשימות של מילונים
    df_ova = pd.DataFrame([
        {'cu': cu, **flatten_report(report)} for cu, report in results_ova
    ])
    
    df_ovo = pd.DataFrame([
        {'cu': cu, **flatten_report(report)} for cu, report in results_ovo
    ])
    
    # כתיבה לקובץ Excel עם שני גליונות
    with pd.ExcelWriter(filename) as writer:
        df_ova.to_excel(writer, sheet_name='One-vs-All', index=False)
        df_ovo.to_excel(writer, sheet_name='One-vs-One', index=False)

def flatten_report(report_dict):
    flat = {}
    for key, value in report_dict.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat[f'{key}_{subkey}'] = subvalue
        else:
            flat[key] = value
    return flat
#-----------------------------Data------------------------------------

def get_data() -> pd.DataFrame:
    df = pd.read_excel('clean.xlsx', index_col = False)
    y = df['target']
    X = df.drop(['target'], axis=1).copy()
    return X, y

# def split_data(X, y : pd.DataFrame, seed) -> Data:
#     X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=seed)
#     X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X_train_full, y_train_full, test_size=0.666, stratify=y_train_full, random_state=seed)
#     return Data(
#         test=Observations(X_test, y_test),
#         labeled=Observations(X_labeled, y_labeled),
#         unlabeled=Observations(X_unlabeled, y_unlabeled)
#     )

def split_data(X, y, seed) -> Data:
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=seed)

    X_labeled, X_unlabeled, y_labeled, y_unlabeled = stratified_custom_split(
        X_train_full, y_train_full, labeled_fraction=0.6, seed=seed)

    return Data(
        test=Observations(X_test, y_test),
        labeled=Observations(X_labeled, y_labeled),
        unlabeled=Observations(X_unlabeled, y_unlabeled)
    )

# פיצול ידני לפי קטגוריה
def stratified_custom_split(X, y, labeled_fraction, seed):
    labeled_idx = []
    unlabeled_idx = []
    
    for label in y.unique():
        label_indices = y[y == label].index.to_list()
        np.random.seed(seed)
        np.random.shuffle(label_indices)
        n_labeled = int(len(label_indices) * labeled_fraction)
        
        labeled_idx += label_indices[:n_labeled]
        unlabeled_idx += label_indices[n_labeled:]
    
    X_labeled = X.loc[labeled_idx]
    y_labeled = y.loc[labeled_idx]
    X_unlabeled = X.loc[unlabeled_idx]
    y_unlabeled = y.loc[unlabeled_idx]
    
    return X_labeled, X_unlabeled, y_labeled, y_unlabeled



def main():
    print('hey')

if __name__ == "__main__":
    main()

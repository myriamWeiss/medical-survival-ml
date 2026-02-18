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
from main import Data, get_data, split_data
from openpyxl import load_workbook
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


def safe_append_df_to_excel(filename, df, sheet_name='Sheet1'):
    file_exists = os.path.isfile(filename)

    if not file_exists:
        # File doesn't exist → create new Excel file with this sheet
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        # File exists → open workbook
        book = load_workbook(filename)

        if sheet_name not in book.sheetnames:
            # Sheet doesn't exist → add new sheet
            with pd.ExcelWriter(filename, engine='openpyxl', mode='a') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            # Sheet exists → append without headers
            startrow = book[sheet_name].max_row
            with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                df.to_excel(writer, sheet_name=sheet_name, startrow=startrow, index=False, header=False)



TSVM_possibility = {
    "kernels": ["rbf"],
    "gamma": [0.001],
    "cu": [ 10, 5, 1, 0.5, 0.1]
}

#--------------compare--------------

def compare_self_train_vs_regular():
    best_kernel  = "rbf"
    best_gamma = 0.01
    best_cu = 0.5
    best_seed = 10
    X, y = get_data()
    with pd.ExcelWriter('svm vs tsvm.xlsx') as writer:
        data = split_data(X, y, best_seed)
        acc, report = get_result_from_regular_svm_one_vs_one_model(data, best_kernel, best_gamma, best_cu)
        result = (best_kernel, best_gamma, best_cu, acc, report)
        df_result = TSVM_results_to_df([result])
        df_result.to_excel(writer, index=False)

   

def get_result_from_regular_svm_one_vs_one_model(data: Data, kernel: str, gamma: float, C: float):
    models_and_pairs_list = []

    X_labeled = data.labeled.X
    y_labeled = data.labeled.y
    X_test = data.test.X
    y_test = data.test.y

    classes = np.unique(y_labeled)
    for cls_i, cls_j in combinations(classes, 2):
        print(f"🔁 Training SVM on ({cls_i}, {cls_j})")
        mask = (y_labeled == cls_i) | (y_labeled == cls_j)
        X_pair = X_labeled[mask]
        y_pair = y_labeled[mask]

        y_binary = np.where(y_pair == cls_i, 0, 1)

        model = SVC(kernel=kernel, C=C, gamma=gamma)
        model.fit(X_pair, y_binary)

        models_and_pairs_list.append((model, (cls_i, cls_j)))

    y_pred = predict_tsvm_one_vs_one(X_test, models_and_pairs_list)

    acc, report = evaluate_predictions(y_test, y_pred)
    return acc, report

#------------------------graph result--------

def make_graphs_from_result():
    path = "TSVM result/TSVM_OvsO_results.xlsx"
    list_seed = [42, 65, 1, 10] # 13,976, 125, 32, 861, 10, 24]
    for seed in list_seed:
        name_sheet = f"Seed {seed}"
        df = pd.read_excel(path, sheet_name= name_sheet)
        plot_self_learning_knn_results(df,seed, "./image/TSVM")

   

def plot_self_learning_knn_results(df_results, seed, save_path):
    """
    Plot accuracy vs. n_neighbors for each combination of metric and weight.
    :param df_results: DataFrame from test_self_learning_knn
    """
    df_results['combination'] = df_results['kernel'].astype(str) + "_" + df_results['gamma'].astype(str)


    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_results,
        x="cu",
        y="accuracy",
        hue="combination",
        marker="o"
    )

    plt.title("One vs One TSVM Accuracy by c")
    plt.xlabel("c")
    plt.ylabel("Test Accuracy")
    plt.legend(title="Kernel + Gamma")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        seed_part = f"_seed{seed}" if seed is not None else ""
        filename = f"TSVM_accuracy{seed_part}.png"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=300)
        print(f"✅ Saved plot to: {full_path}")
              
    plt.show()


#----------------------------------------------

#----------------Train-----------------#
def train_tsvm_one_vs_all(X_labeled, y_labeled, X_unlabeled, kernel, gamma, cu):
    models = []
    print( f"y_labeled distribution: { y_labeled.value_counts()}" )
    classes = np.unique(y_labeled)
    for cls in classes:
        print(f"Runing train for this class : {cls}")
        evaluation = get_evaluation()
        y_binary = np.where(y_labeled == cls, 1, 0)
        model, evaluation = model_tsvm(X_labeled, y_binary, X_unlabeled, evaluation,kernel, gamma, cu) #TSVM
        models.append(model)
    return models

def train_tsvm_one_vs_one(X_labeled, y_labeled, X_unlabeled, kernel, gamma,cu):
    models_and_pairs_list = []

    classes = np.unique(y_labeled)
    for cls_i, cls_j in combinations(classes, 2):
        print(f"🔄 Runing train for this combination of {cls_i},{cls_j}")
        evaluation = get_evaluation()

        mask = (y_labeled == cls_i) | (y_labeled == cls_j)
        X_pair = X_labeled[mask]
        y_pair = y_labeled[mask]

        y_binary = np.where(y_pair == cls_i, 0, 1)
        model, evaluation = model_tsvm(X_pair, y_binary, X_unlabeled, evaluation,kernel, gamma, cu) #TSVM
        pair = (cls_i, cls_j)
        models_and_pairs_list.append((model, pair))
       
        print(f"✅ Done with ({cls_i},{cls_j})")

    return models_and_pairs_list

#--------------------------TSVM--------------------------------#
# def model_tsvm(X_labeled, y_labeled, X_unlabeled, evaluation, cu):
#     model=TSVM(evaluation=evaluation, Cu= cu) #Cu default  = 0.001
#     model.fit(X=X_labeled,y=y_labeled,unlabeled_X=X_unlabeled)
#     return model, evaluation

def model_tsvm(X_labeled, y_labeled, X_unlabeled, evaluation, kernel, gamma, cu):
    if isinstance(X_labeled, pd.DataFrame):
        X_labeled = X_labeled.to_numpy()
    if isinstance(y_labeled, pd.Series):
        y_labeled = y_labeled.to_numpy()
    if isinstance(X_unlabeled, pd.DataFrame):
        X_unlabeled = X_unlabeled.to_numpy()

    model = TSVM(evaluation=evaluation, kernel=kernel, gamma=gamma, Cu=cu)
    model.fit(X=X_labeled, y=y_labeled, unlabeled_X=X_unlabeled)
    return model, evaluation

#----------------Test-----------------#

def predict_tsvm_one_vs_all(X_test, models : list[TSVM]):
    scores = []
    for model in models:
        print(f"Runing Test OvsA")
        prob = model.predict_proba(X=X_test)[:, 1]  # הסתברות למחלקה החיובית
        scores.append(prob.reshape(-1, 1))        # reshape לצירוף בהמשך
    scores = np.hstack(scores)  # shape: (n_samples, n_classes)
    y_pred = np.argmax(scores, axis=1)
    return y_pred

def predict_tsvm_one_vs_one(X_test, models_and_pairs_list):
    predictions = []

    for model, (cls_i, cls_j) in models_and_pairs_list:
        print(f"Runing Test OvsO")
        y_pred = model.predict(X=X_test)
        print(f"➡ Predicted {len(y_pred)} labels for X_test with shape {X_test.shape}")
        if len(y_pred) != len(X_test):
            print("Truncating")
            y_pred = y_pred[-len(X_test):]
        # המרת תוצאה בינארית בחזרה לשמות המחלקות המקוריים
        y_labels = np.where(y_pred == 0, cls_i, cls_j)
        predictions.append(y_labels)

    predictions = np.array(predictions).T  # shape: (n_samples, n_models)

    # Voting – לכל דוגמה ניקח את הערך הכי נפוץ
    final_predictions = []
    for preds in predictions:
        most_common_label = Counter(preds).most_common(1)[0][0]
        final_predictions.append(most_common_label)

    return np.array(final_predictions)

#---------------------------helper-----------------------------------
def get_evaluation():
    return {
        'accuracy':Accuracy(),
        'precision':Precision(average='macro'),
        'Recall':Recall(average='macro'),
        'AUC':AUC(multi_class='ovo'),
        'Confusion_matrix':Confusion_Matrix(normalize='true')
        }

def evaluate_predictions(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    return acc, report


def TSVM_results_to_df(results_TSVM) -> pd.DataFrame:
    return pd.DataFrame([
        {'kernel': kernel, 'gamma': gamma, 'cu': cu, 'accuracy' : acc ,**flatten_report(report)}
        for kernel, gamma, cu , acc,  report in results_TSVM
    ])
#-----------------------Excel---------------------------------

def export_results_to_excel(results_ova, results_ovo, filename='tsvm_results.xlsx'):
    # המרה לרשימות של מילונים
    df_ova = pd.DataFrame([
        { 'cu': cu, **flatten_report(report)} for cu, report in results_ova
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

#------------------------Get result ------------------------------

def get_result_from_tsvm_one_vs_all_model(data : Data, kernel, gamma, cu):
    #train
    models_tsvm_one_vs_all = train_tsvm_one_vs_all(data.labeled.X, data.labeled.y, data.unlabeled.X, kernel, gamma, cu)
    #test 
    y_pred_OvsA = predict_tsvm_one_vs_all(data.test.X, models_tsvm_one_vs_all)
    acc, report_OvsA = evaluate_predictions(data.test.y, y_pred_OvsA)
    return acc, report_OvsA

def get_result_from_tsvm_one_vs_one_model(data : Data, kernel, gamma, cu):
    #train
    models_and_pairs_list = train_tsvm_one_vs_one(data.labeled.X, data.labeled.y, data.unlabeled.X, kernel, gamma, cu)
    #test 
    y_pred_OvsO = predict_tsvm_one_vs_one(data.test.X, models_and_pairs_list)
    acc, report_OvsO = evaluate_predictions(data.test.y,  y_pred_OvsO)
    return acc, report_OvsO

#--------------------Run-----------------------------

def train_and_test_tsvm_OvsO(data : Data):
    list_results_OvsO = []
    result_for_combination = []
    for kernel, gamma, cu in product(TSVM_possibility["kernels"], TSVM_possibility["gamma"], TSVM_possibility["cu"]):
        acc_OvsO , report_OvsO = get_result_from_tsvm_one_vs_one_model(data, kernel, gamma, cu)
        #to excel
        result_for_combination = (kernel, gamma, cu, acc_OvsO, report_OvsO)
        df_result_comb = TSVM_results_to_df([result_for_combination])
        safe_append_df_to_excel('TSVM result/dynamic_result.xlsx', df_result_comb)

        list_results_OvsO.append((kernel, gamma,cu, acc_OvsO, report_OvsO))
    df_result_OvsO = TSVM_results_to_df(list_results_OvsO)
    return df_result_OvsO

def train_and_test_tsvm_OvsA(data : Data):
    list_results_OvsA = []
    for kernel, gamma, cu in product(TSVM_possibility["kernels"], TSVM_possibility["gamma"], TSVM_possibility["cu"]):
        acc_OvsA , report_OvsA = get_result_from_tsvm_one_vs_all_model(data, kernel, gamma, cu)
        list_results_OvsA.append((kernel, gamma ,cu, acc_OvsA,report_OvsA)) 
    df_result_OvsA = TSVM_results_to_df(list_results_OvsA)
    return  df_result_OvsA 

#---------------------------------------------------------------------------------
def run():
    X, y = get_data()
    with pd.ExcelWriter('TSVM result/TSVM_OvsO_results.xlsx') as writer_ovo:
        for seed in [10]: #42, 65, 1 ]: #13,976, 125, 32, 861, 10, 24]]:
            data = split_data(X, y, seed)
            df_result_OvsO = train_and_test_tsvm_OvsO(data)
            df_result_OvsO.to_excel(writer_ovo, sheet_name=f"Seed {seed}", index=False)
            print(f"finish: {seed}")



if __name__ == "__main__":
    run()
    #make_graphs_from_result()
    #compare_self_train_vs_regular()
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sweetviz as sv
from sweetviz import analyze, FeatureConfig
import numpy
from pandasgui import show
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


list_col_name = ["age", "edu",  "num.co", "scoma","charges", "totcst", "totmcst", "avtisst", "hday",  "sps", "aps", "prg2m", "prg6m","dnrday", "adlsc"]

# --------------- get clear Data ------------------
def make_Dict_by_categories():
  
    column_categories = {
    "demographics": [
        "age", "sex", "race", "edu", "income"
    ],
    # "outcomes": [
    #     "death", "hospdead", "surv2m", "surv6m", "sfdm2" #drop all
    # ],
    "disease_info": [
        "disease_code", "num.co", "scoma", "ca", "diabetes", "dementia" #drop "dzgroup", "dzclass",
    ],
    "hospitalization_and_costs": [
        "totcst", "avtisst", "hday" #"charges, "totmcst"
    ],
    "prognosis_scores": [
        "aps", "prg2m", "prg6m" #"sps", 
    ],
    "dnr_status": [
        "dnr", "dnrday"
    ],
    "clinical_physiology": [  
        "meanbp", "wblc", "hrt", "resp", "temp", "crea", "sod", "pafi_binned", "alb_binned" ,"bun_binned","ph_binned", "glucose_binned",  "bili_binned"   # ["pafi", "alb","bun", "urine", "ph", "glucose",  "bili"] ,
        "urine_binned"
    ] ,                   
    "functional_status": [ "adlsc" ] #"adlp", "adls",
    }
    return column_categories


def classify_patient(row, threshold=0.5) -> int : 
    if row["death"] == 0:
        return 1 #Alive
    elif row["hospdead"] == 1: # in H
        if row["surv2m"] <= threshold:
            return 2 # dead in H after 0-2m 
        if row["surv2m"] > threshold and row["surv6m"] <= threshold:
            return 3 #dead in H after 2m-6m 
        elif row["surv6m"] > threshold:
            return 4 #dead in H after 6m+ 
    elif row["hospdead"] == 0: #not in H 
        if row["surv2m"] <= threshold:
            return 2 # dead not in H after 0-2m
        if row["surv2m"] > threshold and row["surv6m"] <= threshold:
            return 3 # dead not in H after 2-6m
        elif row["surv6m"] > threshold:
            return 4 # dead after 6m+ 
    raise RuntimeError("failed to classify patient: " + str(row))

#main
def prepare_target(df : pd.DataFrame) -> pd.DataFrame:
    df["target"] = df.apply(classify_patient, axis=1)
    df = df.drop(["death", "surv2m", "surv6m"], axis=1)
    return df


# --------------- encode categorials column ------------------


def encode_columns(df : pd.DataFrame) ->  pd.DataFrame:
    #demographics #"age", "sex", "race", "edu", "income"
    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    df["race"] = df["race"].astype(str).str.lower()
    
    race_map = {"white": 1, "black": 2, "hispanic": 3, "asian": 4, "other": 5}
    df["race"] = df["race"].map(race_map)

    income_map = {
    "under $11k": 1,
    "$11-$25k": 2,
    "$25-$50k": 3,
    ">$50k": 4
    }
    df["income"] = df["income"].map(income_map)

    #encode_outcomes =["death", "hospdead", "surv2m", "surv6m"] # "sfdm2"
    sfdm2_map = {
        "no(M2 and SIP pres)" : 1, 
        "adl>=4 (>=5 if sur)" : 2,
        "SIP>=30" : 3, 
        "Coma or Intub" : 4, 
        "<2 mo. follow-up" : 5,
    }
    df["sfdm2"] = df["sfdm2"].map(sfdm2_map)

    #disease_info "dzgroup", "dzclass", "num.co", "scoma", "ca", "diabetes", "dementia"
    #"dzgroup", "dzclass"
    dzgroup_map = {
        "Lung Cancer": 11,
        "Colon Cancer": 12,
        "Cirrhosis": 21,
        "COPD": 22,
        "CHF": 23,
        "Coma": 31,
        "MOSF w/Malig": 41,
        "ARF/MOSF w/Sepsis": 42
    }
    df["disease_code"] = df["dzgroup"].map(dzgroup_map)

    ca_map = {
    "no": 1,
    "yes": 2,
    "metastatic": 3
    }
    df["ca"] = df["ca"].map(ca_map)

    #hospitalization_and_costs_column  "charges", "totcst", "totmcst", "avtisst", "hday"

    #prognosis_scoresame_and_dnr_status "sps", "aps", "prg2m", "prg6m", "dnrday"] #"dnr"
    dnr_map = {
        "no dnr": 0,
        "dnr before sadm": 1,
        "dnr after sadm": 2,
        "missing": 3
    }
    df["dnr"] = df["dnr"].map(dnr_map)

    #clinical_physiology
    # df["pafi_binned"] = df["pafi"].apply(bin_pafi)
    df["alb_binned"] = df["alb"].apply(bin_alb)
    # df["bili_binned"] = df["bili"].apply(bin_bili)
    # df["ph_binned"] = df["ph"].apply(bin_ph)
    df["glucose_binned"] = df["glucose"].apply(bin_glucose)
    df["bun_binned"] = df["bun"].apply(bin_bun)
    df["urine_binned"] = df["urine"].apply(bin_urine)

    return df

def fill_na(df : pd.DataFrame) ->  pd.DataFrame:  
    #demographics #"age", "sex", "race", "edu", "income"
    df["age"] = df["age"].fillna(df["age"].mean())
    df["sex"] = df["sex"].fillna(0).astype(int)
    df["race"] = df["race"].fillna(5).astype(int)
    df["edu"] = df["edu"].fillna(df["edu"].mean()) 
    df["income"] = df["income"].fillna(0).astype(int)

    #encode_outcomes =["death", "hospdead", "surv2m", "surv6m"] # "sfdm2"
    df["sfdm2"] = df["sfdm2"].fillna(0).astype(int)

    #disease_info "dzgroup", "dzclass", "num.co", "scoma", "ca", "diabetes", "dementia"
    df["disease_code"] = df["disease_code"].fillna(0).astype(int)

    #hospitalization_and_costs_column  "charges", "totcst", "totmcst", "avtisst", "hday"
    list_of_col = ["charges", "totcst", "totmcst", "avtisst", "hday"] 
    for col_name in list_of_col:
        df[col_name] = df[col_name].fillna(df[col_name].mean())

    df["totmcst"] = df["totmcst"].fillna(df["totcst"])

    #prognosis_scoresame_and_dnr_status "sps", "aps", "prg2m", "prg6m", "dnrday"] #"dnr"
    list_of_col = ["sps", "aps", "prg2m", "prg6m", "dnrday"] #"dnr"
    for col_name in list_of_col:
        df[col_name] = df[col_name].fillna(df[col_name].mean())
    df["dnr"] = df["dnr"].fillna(3).astype(int)

    #clinical_physiology
    list_of_col = [  "ph", "bili", "pafi","meanbp", "wblc", "hrt", "resp", "temp", "crea", "sod" ] 
    other_col = [ "alb","bun", "urine", "glucose",  ]
    other_col = ["pafi_binned", "alb_binned" ,"bun_binned", "urine_binned","ph_binned", "glucose_binned",  "bili_binned"]
    for col_name in list_of_col:
        df[col_name] = df[col_name].fillna(df[col_name].mean())

    return df



def drop_cols(df : pd.DataFrame) ->  pd.DataFrame:  
    #demographics

    # encode_outcomes =["death", "hospdead", "surv2m", "surv6m"] # "sfdm2"
    
    list_of_col =["death",  "surv2m", "surv6m"] # "sfdm2" "hospdead",
    for col_name in list_of_col:
        df = df.dropna(subset=list_of_col)

    list_of_col =["num.co", "scoma", "ca", "diabetes", "dementia"]
    df = df.dropna(subset=list_of_col)

    #disease_info "dzgroup", "dzclass", "num.co", "scoma", "ca", "diabetes", "dementia"
    df = df.drop(columns=["dzgroup", "dzclass"])

    #hospitalization_and_costs_column  "charges", "totcst", "totmcst", "avtisst", "hday"
    #prognosis_scoresame_and_dnr_status "sps", "aps", "prg2m", "prg6m", "dnrday"] #"dnr"

    #clinical_physiology 
    # df = df.drop(columns=["pafi"])
    df = df.drop(columns=["alb"])
    # df = df.drop(columns=["bili"])
    # df = df.drop(columns=["ph"])
    df = df.drop(columns=["glucose"])
    df = df.drop(columns=["bun"])
    df = df.drop(columns=["urine"])

    #functional_status  # ["adlp", "adls", "adlsc"]
    df = df.drop(columns=["adlp", "adls"])

    df = df.drop(columns=[ "urine_binned", "totmcst", "totcst", "sps",  "prg2m", "prg6m"]) #column with hight corelation ,
    return df

#--------------func to encode column ------------
# פונקציות binning פשוטות לערך יחיד:
def bin_pafi(value):
    if pd.isna(value):
        return 0
    elif value > 300:
        return 1
    elif value > 200:
        return 2
    elif value > 100:
        return 3
    else:
        return 4

def bin_alb(value):
    if pd.isna(value):
        return 0
    elif value > 3.5:
        return 1
    elif value > 2.5:
        return 2
    else:
        return 3

def bin_bili(value):
    if pd.isna(value):
        return 0
    elif value <= 1.2:
        return 1
    elif value <= 3:
        return 2
    else:
        return 3

def bin_ph(value):
    if pd.isna(value):
        return 0
    elif 7.35 <= value <= 7.45:
        return 1
    elif 7.25 <= value < 7.35 or 7.45 < value <= 7.55:
        return 2
    else:
        return 3

def bin_glucose(value):
    if pd.isna(value):
        return 0
    elif value < 70:
        return 3
    elif value <= 140:
        return 1
    elif value <= 200:
        return 2
    else:
        return 4
    
def bin_bun(value):
    if pd.isna(value):
        return 0
    elif value <= 20:
        return 1
    elif value <= 40:
        return 2
    else:
        return 3

def bin_urine(value):
    if pd.isna(value):
        return 0
    elif value >= 1500:
        return 1
    elif value >= 500:
        return 2
    else:
        return 3

#-----------------------END-------------------
  



#--------------------Remove outlier -------------------
def remove_outliers(df: pd.DataFrame, numeric_columns: list, iqr_factor: float = 2.0) -> pd.DataFrame:
    original_rows = df.shape[0]

    for col in numeric_columns:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_factor * IQR
        upper = Q3 + iqr_factor * IQR

        # שורות שיוצאות מהטווח (ונשמרים NaN)
        df = df[(df[col].isna()) | ((df[col] >= lower) & (df[col] <= upper))]

    final_rows = df.shape[0]
    removed = original_rows - final_rows

    print(f"Removed {removed} rows due to outliers ({round((removed/original_rows)*100, 2)}%).")
    print(f"Remaining rows: {final_rows}")

    return df


#-------------------Normalize-------------------------------
def make_normalization(df : pd.DataFrame)-> pd.DataFrame:
    minmax_scaler_columns = [
    "avtisst", "hday", "aps", "prg2m", "prg6m", "dnrday", "adlsc",
    "meanbp", "wblc", "hrt", "resp", "temp", "crea", "sod", "totcst", "disease_code"]
    standard_scaler_columns = ['scoma', 'age', 'edu', 'num.co', 'sfdm2' ]

    # standard_scaler_columns = [
    # "scoma", "edu", "prg2m", "prg6m", "sod",  "resp", 
    # "temp", "aps", "crea",  "meanbp", "dnrday", "hday", "num.co"]
    
    # minmax_scaler_columns = ["dementia", "sex",  "ca", "dnr", "income", "race", "sfdm2"]

    df = normalize_columns_standars(df , standard_scaler_columns)
    df = normalize_columns_by_max_min(df, minmax_scaler_columns)
    return df


def normalize_columns_by_max_min(df : pd.DataFrame, columns) -> pd.DataFrame:
    df_norm = df.copy()
    for col in columns:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        if max_val - min_val != 0:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0.0  # או NaN, תלוי במדיניות הרצויה
    return df_norm

from sklearn.preprocessing import StandardScaler

def normalize_columns_standars(df : pd.DataFrame, columns_to_scale):
    """
    Normalize selected columns in the dataframe using StandardScaler.

    :param df: pandas DataFrame
    :param columns_to_scale: list of column names to normalize
    :return: new DataFrame with normalized columns
    """
    df_scaled = df.copy()
    scaler = StandardScaler()
    df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])
    return df_scaled


# --------------- Correlation Table ------------------

def show_correlation_matrix(data: pd.DataFrame, title: str = "Correlation Matrix"):
    if data.shape[1] < 2:
        print("Not enough variables for correlation matrix.")
        return None

    corr = data.corr(numeric_only=True)
    plt.figure(figsize=(max(8, len(corr.columns) * 0.6), 6))
    plt.matshow(corr, fignum=1)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title(title, pad=20)
    plt.tight_layout()
    plt.show()
    return corr

def corelation_for_categories(df: pd.DataFrame):
    dict_categories = make_Dict_by_categories()
    for category, columns_list in dict_categories.items():
        filtered_cols = [col for col in columns_list if col in df.columns]
        df_by_category = df[filtered_cols]
        if df_by_category.shape[1] >= 2:
            print(f"\n🔷 Correlation Matrix for Category: {category}")
            show_correlation_matrix(df_by_category, title=f"Correlation: {category}")
        else:
            print(f"\n⚠️ Skipping '{category}' - not enough numeric columns found.")


# --------------- Sweetviz ------------------
def analyze_data(df: pd.DataFrame):
    config = FeatureConfig(force_num=["target"])  # force 'target' to be numeric
    report = sv.analyze(df, target_feat="target", feat_cfg=config)
    report.show_html("SWEETVIZ_REPORT.html")



def get_top_correlated_features(df: pd.DataFrame, target_column: str, threshold: float = 0.1):
    """
    Calculate Pearson correlation between all numeric features and target.
    Return sorted Series of correlations and list of top features above threshold.
    """
    numeric_df = df.select_dtypes(include='number')
    correlations = numeric_df.corr()[target_column].drop(target_column)
    sorted_corr = correlations.reindex(correlations.abs().sort_values(ascending=False).index)
    top_features = sorted_corr[sorted_corr.abs() >= threshold].index.tolist()
    return sorted_corr, top_features

#main 
def filter_features_by_correlation(df: pd.DataFrame, target_column: str, threshold: float = 0.1):
    """
    Returns a DataFrame with only the features most correlated with the target.
    """
    _, top_features = get_top_correlated_features(df, target_column, threshold)
    selected_cols = top_features + [target_column]
    return df[selected_cols]

#-------------------------PCA------------------------
def get_data_pca(df: pd.DataFrame) -> pd.DataFrame:
    y = df["target"]
    X = df.drop(columns=["target"]).copy()
    df_X = run_pca(X)
    return  pd.concat([df_X, y], axis=1)

def run_pca(df: pd.DataFrame, n_components: int = 0.95) -> pd.DataFrame:
    """
    מבצע PCA על DataFrame ומחזיר DataFrame חדש עם הרכיבים הראשיים.

    :param df: DataFrame עם עמודות מספריות בלבד
    :param n_components: מספר רכיבים לשמור, או אחוז שונות מצטברת (למשל 0.95)
    :return: DataFrame חדש לאחר PCA
    """
    # בודק רק עמודות מספריות
    df_numeric = df.select_dtypes(include=['number']).dropna()

    # תקנון הנתונים (סטנדרטיזציה)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)

    # הפעלת PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)

    # יצירת עמודות בשם PC1, PC2 וכו'
    col_names = [f"PC{i+1}" for i in range(pca_result.shape[1])]
    df_pca = pd.DataFrame(pca_result, columns=col_names, index=df_numeric.index)

    return df_pca

#-------------------------------------------------------------#

def prepare_data(df : pd.DataFrame) -> pd.DataFrame:
    if "index" in df.columns or "INDEX" in df.columns:
        df = df.drop(columns=["index", "INDEX", ""], errors="ignore")
    df = df[~(df["ph"].isna() & df["alb"].isna())] #remove row with empty
    df = endale_categorical_columns_and_missing_value(df)
    df = prepare_target(df)
    #corelation_for_categories(df)
    return df

#main 
def endale_categorical_columns_and_missing_value(df: pd.DataFrame) -> pd.DataFrame:
    df = encode_columns(df)
    df = fill_na(df)
    df = drop_cols(df)
    return df

# --------------- ???????????????????? ------------------

def main():
    path = "All_data.xlsx"
    df = pd.read_excel(path)
    df = prepare_data(df)
    df = get_data_pca(df)
    #df = make_normalization(df)
    df.to_excel("clean.xlsx")
    print("Finish")

def run_analyze():
    path = "All_data.xlsx"
    df = pd.read_excel(path)
    df = prepare_data(df)
    df.to_excel("before PCA.xlsx")
    #analyze_data(df)
    #filter_features_by_correlation(df, "target")


def binary_tree(path):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree

   
    df = pd.read_excel('clean.xlsx', index_col = False)
    y = df['target']
    X = df.drop(['target'], axis=1).copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print(classification_report(y_test, y_pred))
  

    importances = pd.Series(tree.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    print(importances.head(10))





if __name__ == "__main__":
    #main()
    run_analyze()
    #binary_tree("clean.xlsx")
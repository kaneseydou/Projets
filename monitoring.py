import multiprocessing
import time
from csv import writer
import os
import csv
import numpy as np
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import learning_curve

# Importation des modèles
from xgboost import  XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import seaborn as sns
²import matplotlib.pyplot as plt
import warnings 

warnings.filterwarnings("ignore")

""" 
Load data

"""
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv", index_col=0)
pd.set_option('display.max_columns',None)

warnings.filterwarnings("ignore")
def separate_instances_labels(data):
    df = data.copy()
    labels = df[["Churn"]]
    instances = df.drop(["Churn"], axis = 1)
    return instances, labels

def conver_tot_char_to_num(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fonction qui convertie TotalCharges en numerique et remplace les catactères invalides par des NAN
    Parameters
    ----------
    data: pd.DataFrame 
    Returns
    ---------
    df: pd.DataFrame
    """
    df = data.copy()
    df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
    return df

def impute(data):
    """
    Fonction qui remplace les NAN de TotalCharges par le MonthlyCharges correspondant
    
    Parameters
    ----------
    data: pd.DataFrame 
    Returns
    ---------
    data: pd.DataFrame
    """
    idx = data[data["TotalCharges"].isna()].index
    data.loc[idx,"TotalCharges"] = data.loc[idx,"MonthlyCharges"]
    return data


def drop_doub_var(data):
    """
    Fonction qui supprime les variables qui existe en doublons
    """
    return data.drop(["ElectronicCheck", "MonthtoMonth"], axis=True)
    
def select_feat_by_corr(X, y, threshold=0.09):
    data = X.merge(y, right_on="customerID", left_index=True)
    churn_corr = data.corr()[["Churn"]].sort_values("Churn")
    feat_to_drop=list(churn_corr[(churn_corr["Churn"]< threshold)& (churn_corr["Churn"]>-threshold)].index)
    return feat_to_drop


def feature_eng(data: pd.DataFrame) ->pd.DataFrame:
    """Fontion qui cree 4 nouvelles variables
    """
    data.loc[:,"ElectronicCheck"] = data.loc[:,"PaymentMethod"]=="Electronic check"
    data.loc[:,"tenure_less_th_15"] = data.loc[:,"tenure"]<=15
    data.loc[:,"MonthtoMonth"] = data.loc[:, "Contract"]=="Month-to-month"
    data.loc[:,"Nb_service"] = (data[["OnlineSecurity","OnlineBackup", "DeviceProtection",
                                        "TechSupport","StreamingTV", "StreamingMovies"]]=='Yes').sum(axis=1)
    return data

def split_cat_num(data):
    """
    Fonction qui separe les variables categrorielle d'avec les numerique
    
    Parameters
    ----------
    data: pd.DataFrame 
    
    Returns
    ---------
    cat: pd.DataFrame
    num: pd.DataFrame
    """
    cat = data.select_dtypes(['object', int, bool])
    num = data.select_dtypes([float])
    #num.loc[num.index, "tenure"] = data.loc[num.index, "tenure"]
    return cat, num


def split_cat_feat(data: pd.DataFrame) ->pd.DataFrame :
    """Separation des variables qualitatives en deux: binaire et non binaire
    """
    binary_feat = []
    for col in data.columns:
        if data[col].unique().shape[0]==2:
            binary_feat.append(col)
    other_cat_feat = data.columns.drop(binary_feat)
    return data[binary_feat], data[other_cat_feat]


def binaire_encoding_fit(data: pd.DataFrame) -> OrdinalEncoder :
    """
    fonction qui entraine le OrdinalEncoder
    """
    ord_enc = OrdinalEncoder()
    ord_enc.fit(data)
    
    return ord_enc


def binaire_encoding_transfrom(data, encoder):
    """Fonction qui transforme les données en utilisant le ordinalendoder
    """
    vals = encoder.transform(data)
    col = data.columns
    return pd.DataFrame(data=vals, columns=col, index=data.index)

def one_hot_encoding_fit(data: pd.DataFrame):
    """
    Fonction qui fait du OneHotEncoder
    """
    one_hot_enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    one_hot_enc.fit(data)
    return one_hot_enc

def one_hot_encoding_transform(data: pd.DataFrame, encoder):
    """
    Fonction qui fait du OneHotEncoder
    """
    vals = encoder.transform(data)
    cols = []
    for i, col in enumerate(data.columns):
        for unique in encoder.categories_[i]:
            cols.append(f"{col}_{unique}")
    df = pd.DataFrame(data = np.hstack((data.values, vals)), columns = list(data.columns) + cols, index=data.index)
    return df.drop(data.columns, axis=1).astype(int)


def scaling_num_feat_fit(data):
    """Fonction qui entaine le minmaxscaler
    """
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler

def scaling_num_feat_transform(data, encoder):
    """Fonction qui transforme les donnée en utilisant le minmaxscaler
    """
    vals = encoder.transform(data)
    col = data.columns
    return pd.DataFrame(data=vals, columns=col, index=data.index)


def creat_mater_table(bin_data, no_bin_data, num_data):
    df = bin_data.merge(no_bin_data, right_on="customerID", left_index=True)
    df = df.merge(num_data, right_on="customerID", left_index=True)
    return df


def label_encoding(data: pd.DataFrame) -> OrdinalEncoder :
    """
    fonction qui fait du OrdinalEncoder0
    """
    df = data.copy()
    ord_enc = LabelEncoder()
    vals = ord_enc.fit_transform(df["Churn"])
    df.loc[:,"Churn"] = vals
    return df

def pipeline_transfrom_data(X, binaire_encoder, one_hot_encoder, min_max_scaler, colunms_to_drop):
    """ 
    pipeline qui applique toutes les modifications et transformations sur X
    
    """
    X = conver_tot_char_to_num(X)
    X = impute(X)
    X = feature_eng(X)
    X_cat, X_num = split_cat_num(X)
    bin_feat, no_bin_feat = split_cat_feat(X_cat)
    bin_feat_encoded = binaire_encoding_transfrom(bin_feat, binaire_encoder)
    #binaire_encoder.transform(bin_feat)
    no_bin_feat_encoded = one_hot_encoding_transform(no_bin_feat, one_hot_encoder)
    #one_hot_encoder.transform(no_bin_feat)
    num_feat_scaled = scaling_num_feat_transform(X_num, min_max_scaler)
    #min_max_scaler.transform(X_num)
    master_table = creat_mater_table(bin_feat_encoded, no_bin_feat_encoded, num_feat_scaled)
    master_table = drop_doub_var(master_table)
    new_master_table = master_table.drop(colunms_to_drop, axis=1)

    return new_master_table


print('Monitoring starting below...')
print('Battery capacity set to',sys.argv[2])

dict_models = {"xgb":XGBClassifier(eval_metric='auc', use_label_encoder=False) , "log_reg": LogisticRegression(), "rnf": RandomForestClassifier()}

try:
    model = dict_models[sys.argv[3]]
except KeyError:
    model = XGBClassifier()    


if len(sys.argv[1])>2:
    print('Writing data into file',sys.argv[1])
#from codecarbon import EmissionsTracker

import psutil
#cpu_control = psutil.cpu_percent()

def get_battery_charge():
    battery = psutil.sensors_battery()
    charge = battery.percent
    #print('Battery percent',charge,'%')
    return(charge)
'''
def get_cpu_temp():
    temp = psutil.sensors_temperatures()
    res = []
    for k in range(4):
        res.append(temp['coretemp'][k+1][1])
    #print('Temperature of CPU',res)
    return(res)

def get_cpu_percent():
    #print('Percentage of CPU used',psutil.cpu_percent(percpu=True))
    return(res)
'''


def monitore():
    #res = {'temp':[],'percent':[]}
    k = 0
    previouscharge = 100
    #periode = 10#in seconds
    # Mesure du temps d'execution du model courrent (sys.arg[3], par defaut XGB)
    begin = time.time()
    train_score_means = []
    train_score_stds = []
    val_score_means = []
    val_score_stds = []
    ext_times = []

    train_score_mean, train_score_std, val_score_mean, val_score_std, exc_time = pipeline_training_evaluation(model)
    train_score_means.append(train_score_mean)
    train_score_stds.append(train_score_std)
    val_score_means.append(val_score_mean)
    val_score_stds.append(val_score_std)
    ext_times.append(exc_time)
    end = time.time()

    periode = end - begin

    # Initialisation des listes pour la sauvegarde
    result = []
    train_score_means = []
    train_score_stds = []
    val_score_means = []
    val_score_stds = []
    ext_times = []
    energy_consumptions = []
    study_time = 0.5 # en heure
    stop_crit = (60/periode)*60*study_time
    #print(f"periode ******* {periode}")
    #print(f"nombre d'iteration ========= {stop_crit}")
    while True and k<stop_crit:
        #96000 = 12000*8heures = (60/periode)parminute*60min*8h
        #res['percent'].append(get_percent())
        #res['temp'].append(get_temp())
        currentcharge = get_battery_charge()
        #cpu_temp = get_cpu_temp()
        #cpu_percent = get_cpu_percent()
        BT_Wh = float(sys.argv[2])#condition of battery in Wh, given by command inxi -B (ubuntu or mac os)
        power_est = 36*(previouscharge - currentcharge)*BT_Wh/periode
        difference = previouscharge - currentcharge
        energy_consumption = 52*(difference/100) #(en Wh)
        if k>0:
            result.append(power_est)
            print('Current charge',currentcharge,'%')
        print(f"pourcentage experience = {np.round(100*k/stop_crit, 2)}")
        if k>0:
            print(k,'| Power estimation =',round(power_est,1),'W')
        previouscharge = currentcharge
        k += 1
        #time.sleep(periode)
        train_score_mean, train_score_std, val_score_mean, val_score_std, exc_time = pipeline_training_evaluation(model)
        train_score_means.append(train_score_mean)
        train_score_stds.append(train_score_std)
        val_score_means.append(val_score_mean)
        val_score_stds.append(val_score_std)
        ext_times.append(exc_time)
        energy_consumptions.append(energy_consumption)

    if len(sys.argv[1])>2:
        np.savetxt(sys.argv[1] + f"_{sys.argv[3]}.txt", result)

    #print(f"train_score_means ========== {train_score_means}")
    df = pd.DataFrame({"train_score_means": train_score_means,
    "train_score_stds": train_score_stds,
    "val_score_means": val_score_means,
    "val_score_stds": val_score_stds,
    "ext_times": ext_times,
    "energy_consumptions": energy_consumptions
    })
    df.to_csv(f"resultats_{sys.argv[3]}.csv")

def pipeline_fit_encoders(X):

    """ 
    """
    X = conver_tot_char_to_num(X)
    X = impute(X)
    X = feature_eng(X)
    X_cat, X_num = split_cat_num(X)
    bin_feat, no_bin_feat = split_cat_feat(X_cat)
    binaire_encoder = binaire_encoding_fit(bin_feat)
    one_hot_encoder = one_hot_encoding_fit(no_bin_feat)
    min_max_scaler = scaling_num_feat_fit(X_num)
    return binaire_encoder, one_hot_encoder, min_max_scaler

def evaluation1(model, X_train, y_train):
    
    model.fit(X_train, y_train)
    b = time.time()
    N, train_score, val_score = learning_curve(model, X_train, y_train,
                                              cv=3, scoring='f1_micro',
                                               train_sizes=np.linspace(0.1, 1, 10),
                                               n_jobs =-1
                                               )
    e = time.time()
    return train_score.mean(), train_score.std(), val_score.mean(), val_score.std(), e-b
    
def pipeline_training_evaluation(model):

    """
    Fonction qui prepare les données (encodage, imputation, ect.), entraine le model
    et sauvegarde les artifacts pour l'inference.
    parametres
    -----------
    row_data: donnees brute
    model: model a entrainer

    return
    -------
    """

    data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv", index_col=0)
    X, y = separate_instances_labels(data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=None)

    binaire_encoder, one_hot_encoder, min_max_scaler = pipeline_fit_encoders(X_train)
    y_train = label_encoding(y_train)
    y_val = label_encoding(y_val)

    # Colonnes à supprimer après encodage et pretraitement

    colunms_to_drop = ['OnlineBackup_Yes',
    'DeviceProtection_Yes',
    'MultipleLines_No',
    'gender',
    'MultipleLines_No phone service',
    'PhoneService',
    'MultipleLines_Yes',
    'StreamingTV_Yes',
    'StreamingMovies_Yes',
    ]

    master_table_train = pipeline_transfrom_data(X_train, binaire_encoder, one_hot_encoder, min_max_scaler, colunms_to_drop)
    master_table_test = pipeline_transfrom_data(X_val, binaire_encoder, one_hot_encoder, min_max_scaler, colunms_to_drop)
    model.fit(X=master_table_train, y=y_train.values.ravel())
    train_score_mean, train_score_std, val_score_mean, val_score_std, exc_time = evaluation1(model, master_table_train, y_train)
    preds = model.predict_proba(master_table_test)
    return train_score_mean, train_score_std, val_score_mean, val_score_std, exc_time


if __name__ == '__main__':
    p1 = multiprocessing.Process(name='p1', target=monitore)
    p1.start()


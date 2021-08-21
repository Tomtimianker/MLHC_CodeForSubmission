import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, average_precision_score, roc_auc_score, plot_precision_recall_curve, plot_roc_curve
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler, minmax_scale, StandardScaler
from sklearn.utils import resample
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import json
from sklearn.linear_model import LogisticRegression
import sys
# sys.path.append(r"C:/Users/MAYA/Documents/university/semester_6/AdaCost/adacost/adacost")
sys.path[0:0] = [r"C:/Users/MAYA/Documents/university/semester_6/AdaCost/adacost/adacost"]
from adacost import AdaCost
import warnings
from sklearn import tree
import pickle
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import defaultdict
warnings.filterwarnings(action='ignore', category=UserWarning)
import time

# preprocessing

def imputateData(data_set):
    imputer = KNNImputer()
    imputer_fit = imputer.fit(data_set)
    data_set_imp = imputer_fit.transform(data_set) # FOR PANDAS
    return data_set_imp, imputer_fit

# resampling

def smote(x, y, sampling_strategy, k):
    features, labels = x, y
    features, labels = SMOTE(sampling_strategy = sampling_strategy, k_neighbors = k, random_state = 7).fit_resample(features, labels.astype("bool"))
    x, y = features, labels
    return (x, y)

def random_down_sampeling(features, labels, down_rate):
    idx_false = np.where(labels == 0)[0]
    idx_true = np.where(labels == 1)[0]
    new_false = random.sample(list(idx_false), k=int(down_rate*idx_false.shape[0]))
    labels = np.concatenate((labels[idx_true], labels[new_false]))
    np.random.seed(7)
    shfl_idx = np.random.permutation(labels.shape[0])
    labels = labels[shfl_idx]
    features = np.concatenate((features[idx_true], features[new_false]))
    features = features[shfl_idx]
    return (features, labels)

def random_down_sampeling_and_smote(features, labels, down_rate, sampling_strategy, k_smt):
    features, labels = smote(features, labels, sampling_strategy, k_smt)
    features, labels = random_down_sampeling(features, labels, down_rate)
    return (features, labels)

# feature selection
def extract_features(features, feature_names):
    chosen_ac_pi_a = [0, 14, 15, 17, 21, 22, 26, 30, 33, 35, 44, 51, 53,
                      54, 55, 58, 61, 70, 77, 90, 98, 99, 101, 115, 117, 118,
                      119, 121, 122, 124, 125, 126, 127, 128, 137, 138, 141, 144,
                      151, 154, 159, 160, 161, 171, 174, 177, 183, 191]
    chosen_rf_pi_b = [13, 45, 49, 65, 80, 93, 105, 118, 120, 126, 127, 128,
                      129, 131, 135, 136, 137, 518]
    chosen_idx = chosen_rf_pi_b
    features = features[:, chosen_idx]
    feature_names = feature_names[chosen_idx]
    return (features, feature_names)

# classifier
def module_5_model_b_creation(model_type, model_b_mimic_cohort):
    
    resampling_func = random_down_sampeling_and_smote
    
    resampling_dict_a = {"down_rate":0.8, "sampling_strategy":0.4, "k_smt":2}
    resampling_dict_b = {"down_rate":1, "sampling_strategy":0.56, "k_smt":1}
    
    classifier_parameter_a = {"algorithm": 'SAMME', "cost_matrix": [[0,1],[3,0]],
                        "learning_rate": 0.5, "max_depth": 10, "n_estimators": 200,
                        "random_state": 26}
    
    classifier_parameter_b = {'ccp_alpha': 0.008, 'class_weight': None, 'criterion': 'gini', 'lho_loo_factor': 3, 'max_depth': None, 'max_features': 'auto', 'max_samples': None, 'n_estimators': 200, 'random_state': 26}

    classifier_parameter = classifier_parameter_b
    resampling_dict = resampling_dict_b

    data_mimic_b = pd.read_csv(fr'{model_b_mimic_cohort}')
    
    X, y = get_X_and_y(data_mimic_b, "target", "identifier")
    labels, features, label_name, feature_names = pd_to_np(X, y)
    
    #MAKE SURE PD NP RIGHT
    X_imp, imputer_fit = imputateData(features)
    scaler = StandardScaler()
    scaler.fit(X_imp) 
    X = scaler.transform(X_imp)
    X, y = resampling_func(X, labels, **resampling_dict)
    X, feature_names = extract_features(X, feature_names)
    
    # classifier = AdaCost(**classifier_parameter)
    classifier = RandomForestClassifier(**classifier_parameter)
    classifier.fit(X, y)
    
    with open("chosen_feature_names_b","wb") as fea:
        pickle.dump(feature_names, fea)
    
    with open("imputer_fit_b","wb") as imp_p:
        pickle.dump(imputer_fit, imp_p)
    with open("scaler_b","wb") as scaler_b:
        pickle.dump(scaler, scaler_b)
    with open("classifier_b","wb") as clf_b:
        pickle.dump(classifier, clf_b)
    
    
    return (imputer_fit, scaler, classifier)

# utils

def get_X_and_y(data_frame, label_name, idx_name):
    y = data_frame.loc[:,[label_name]]
    X = data_frame.drop(columns = [label_name, idx_name])
    return X, y

def np_to_pd(labels, features, label_name, feature_names):
    x = pd.DataFrame(features, columns = feature_names)
    y = pd.DataFrame(labels, columns = label_name)
    return (x, y)

def pd_to_np(x, y):
    label_name = y.columns
    labels = y.to_numpy().astype("bool")
    feature_names = x.columns
    features = x.to_numpy()
    return (labels, features, label_name, feature_names)



module_5_model_a_creation("b", "./processed_data_mimic_task_b.csv")
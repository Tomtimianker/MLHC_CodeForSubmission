import pandas as pd
import warnings
import pickle
warnings.filterwarnings(action='ignore', category=UserWarning)
from adacost.adacost import AdaCost

def module_3_model(test_set, model_type):
    test_set = pd.read_csv(fr'{test_set}')
    X, identifier = get_X_and_id(test_set, "identifier")
    clf = pickle.load(open("trained/task_"+str(model_type)+"/classifier_"+str(model_type), "rb"))
    proba = clf.predict_proba(X)[:,1]
    risk_data = pd.DataFrame({"risk_score": proba})
    output_df = pd.concat([identifier, risk_data], axis=1)
    output_df.to_csv("model_"+str(model_type)+"_mimic_cohort_risk_score_group_2.csv", index=False)
    
# Utils

def get_X_and_id(data_frame, idx_name):
    identifier = data_frame.loc[:,[idx_name]]
    X = data_frame.drop(columns = [idx_name])
    return X, identifier


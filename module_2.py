from preprocessing_constants import *
import pandas as pd
import sklearn.preprocessing
from sklearn.impute import KNNImputer

# Constant that represents the time range for using lab results
# (we will use the results from the DAYS_BACK before target time)
DAYS_BACK = 4
DEBUG = False

def debuggable(func):
    if DEBUG:
        def decorated(*args, **kwargs):
            print("Entering ",func.__name__)
            ret = func(*args,  **kwargs)
            print(func.__name__, "finished ")
            return ret
        return decorated
    else:
        return func

@debuggable
def module_2_preprocessing(external_validation_set, model_type):
    if model_type == 'a':
        DAYS_BACK = 3
    
    df = get_all_features(external_validation_set[0], external_validation_set[1], external_validation_set[2],
                          model_type).set_index('identifier')
    # Imputate data
    imputer = KNNImputer()
    imputer_fit = imputer.fit(df)
    df_imp = imputer_fit.transform(df)
    # Scale data
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(df_imp)
    df_final = pd.DataFrame(scaler.transform(df_imp), columns=df.columns)
    if model_type == "a":
        df_final = df_final[FEATURES_FOR_MODEL_A]
    else:
        df_final = df_final[FEATURES_FOR_MODEL_B]
    return df_final

@debuggable
def get_all_features(microbio_path, drugs_path, lab_path, model_type):
    lab_df = get_lab_full_features_df(lab_path, model_type)
    microbio_df = get_microbio_df(microbio_path, model_type)
    drugs_df = get_drugs_df(drugs_path)
    df = lab_df.merge(microbio_df, how='left', left_index=True, right_on='identifier')
    df = df.merge(drugs_df, how='left', on="identifier")
    return df.dropna(how='all', axis=1)

@debuggable
def get_lab_full_features_df(data_path, model_type):
    df = get_lab_data(data_path, model_type)
    return create_full_features_table(df)

@debuggable
def get_microbio_df(data_path, model_type):
    df = pd.read_csv(fr'{data_path}')
    if model_type == 'a':
        df = df.replace(MICROBIOLOGY_DICT)
        df = df.rename(columns={'spec_type_desc': 'culturesite'})
        df = pd.get_dummies(df, columns=['culturesite'])
    else:
        df = pd.get_dummies(df, columns=['spec_type_desc', 'org_name'])
    df = df.drop_duplicates()
    df = df.groupby('identifier').sum()
    return df.drop_duplicates()

@debuggable
def get_drugs_df(data_path):
    df = pd.read_csv(fr'{data_path}')
    df = df.set_index('identifier')
    # each drug will be represented by the the first word in the name, in capital letters
    df['drug'] = df['drug'].str.replace("-", "")
    df = pd.DataFrame(df.drug.str.split(' ', 1).tolist(), columns=['drug', 'rest'], index=df.index)
    df = df[df['drug'].str.isalpha()] # Removing drugs that start with a number
    df['drug'] = df['drug'].str.upper()
    df = df.drop(['rest'], axis=1)
    df = df.reset_index()
    df = df.drop_duplicates()
    df = pd.get_dummies(df, columns=['drug'])
    df = df.groupby('identifier').sum()
    return df

@debuggable
def get_lab_data(data_path, model_type):
    data_set = pd.read_csv(fr'{data_path}')
    convert_dates_to_daytime_type(data_set)
    # Remove non human values
    data_set = data_set[data_set.apply(is_human_value, axis=1)]
    # Removes features we definitely won't use
    data_set = data_set.drop(
        ['itemid', 'subject_id', 'hadm_id', 'target_time', 'admittime', 'dob', 'hours_from_admittime_to_charttime',
         'hours_from_admittime_to_targettime'], axis=1)
    if model_type == 'a':
        # Replace column name
        data_set = data_set.rename(columns={'estimated_age': 'age'})
        # Replace ethnicity names
        data_set = data_set.replace(ETHNICITY_DICT)
    data_set = pd.get_dummies(data_set, columns=['ethnicity'])
    data_set['gender'] = data_set['gender'].apply(lambda x: 0 if x == 'M' or x == 'Male' else 1).astype(int)
    return data_set.drop_duplicates()


# Return the df such that for each patient (identifier) and each of its examinations (label) we will keep the results
# from it's latest examinations.
@debuggable
def keep_latest_examination(data_set):
    data_set = data_set[data_set.groupby(['identifier', 'label'])['charttime'].transform(max) == data_set['charttime']]
    return data_set.drop_duplicates()


# Return the df such that for each patient (identifier) and each of its examinations (label) we will keep results
# from the latest DAYS_BACK days before the target time.
@debuggable
def keep_recent_lab_results(data_set):
    data_set_from_k_last_days = data_set.copy()
    data_set_from_k_last_days['days_from_charttime_time_to_targettime'] = data_set_from_k_last_days[
        'hours_from_charttime_time_to_targettime'].div(24)
    # Leaving the only the rows with days_from_charttime_time_to_targettime smaller than DAYS_BACK.
    return data_set_from_k_last_days[data_set_from_k_last_days['days_from_charttime_time_to_targettime'] <= DAYS_BACK]

@debuggable
def transform_lab_results(data_set, func, func_name):
    data_frame = data_set[data_set.groupby(['identifier', 'label'])['valuenum'].transform(func) == data_set['valuenum']]
    # Of all of those with the same value, returning with only one row per lab (the latest lab)
    data_frame = keep_latest_examination(data_frame)
    return data_frame.set_index(['identifier', 'label']).valuenum.unstack().add_prefix(
        str(DAYS_BACK) + '_days_' + func_name + '_')


'''
Creates the table that includes all the features, before removing the ones that weren't selected.
For each patient and each examination we will generate:
1. The maximal, minimal and average value from the latest k days before the target time.
2. The variance of it's values
The table that returns has one row for each patient.
'''
@debuggable
def create_full_features_table(data_set):
    lab_values_attributes_dfs = []
    data_frame = keep_recent_lab_results(data_set)
    functions = [(max, "max"), (min, "min")]
    for tpl in functions:
        lab_values_attributes_dfs.append(transform_lab_results(data_frame, tpl[0], tpl[1]))
    lab_values_attributes_dfs.append(
        data_frame.groupby(['identifier', 'label'], as_index=False)['valuenum'].mean().set_index(
            ['identifier', 'label']).valuenum.unstack().add_prefix(str(DAYS_BACK) + '_days_avg_'))
    lab_values_attributes_dfs.append(lab_results_variance(data_set))
    one_row_df = create_one_row_per_patient_data_set(data_set)
    one_row_df = one_row_df.set_index('identifier')
    return pd.concat([one_row_df] + lab_values_attributes_dfs, axis=1)


# return the variance of each lab test
@debuggable
def lab_results_variance(data_set):
    var_exam_data = data_set.groupby(['identifier', 'label'], as_index=False)['valuenum'].var()
    return var_exam_data.set_index(['identifier', 'label']).valuenum.unstack().add_prefix('var_')


# Remove features (columns) that are related to charttime so that we can unite all of the rows of the same patient.
# Removing the 'label' and 'valuenum' features since this information will come from the labels table
@debuggable
def create_one_row_per_patient_data_set(data_set):
    data_set = data_set.drop(['charttime', 'valuenum', 'label', 'hours_from_charttime_time_to_targettime'], axis=1)
    return data_set.drop_duplicates()


# ------------------------------------------------
# -------------- UTILITY FUNCTIONS ---------------
# ------------------------------------------------

# returns True if the lab value in the row is within human ranges, for the ranges we have.
def is_human_value(row):
    if row['label'] not in HUMAN_RANGES.keys():
        return True
    if (float(row['valuenum']) >= HUMAN_RANGES[row['label']][0]) and (
            float(row['valuenum']) <= HUMAN_RANGES[row['label']][1]):
        return True
    return False


# Convert dates to types datetime instead objects so we can compare between them.
@debuggable
def convert_dates_to_daytime_type(data_set):
    data_set['charttime'] = pd.to_datetime(data_set['charttime'], dayfirst=True)
    data_set['target_time'] = pd.to_datetime(data_set['target_time'], dayfirst=True)
    data_set['admittime'] = pd.to_datetime(data_set['admittime'], dayfirst=True)

if __name__ == '__main__':
    df = module_2_preprocessing(["D:/Uni/Year3/MLHC/module_1_cohort_creation/external_validation_set_lab_weight_ethnicity.csv", "D:/Uni/Year3/MLHC/module_1_cohort_creation/external_validation_set_microbio.csv","D:/Uni/Year3/MLHC/module_1_cohort_creation/external_validation_set_drugs.csv"], 'a')
    print(df.head(10))

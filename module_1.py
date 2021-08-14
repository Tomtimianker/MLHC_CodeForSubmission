import pandas as pd
import psycopg2
from psycopg2 import sql
from os import path

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
def module_1_cohort_creation(file_path, db_conn, model_type):
    microbio_output_path = path.abspath("./external_validation_set_microbio.csv")
    drugs_output_path = path.abspath("./external_validation_set_drugs.csv")
    lab_weight_ethnicity_output_path = path.abspath("./external_validation_set_lab_weight_ethnicity.csv")
    cur = db_conn.cursor()
    # Data loading
    query = """
    SET datestyle = dmy;

    DROP TABLE IF EXISTS mimic_cohort;
    CREATE TABLE mimic_cohort (
      identifier VARCHAR(50),
      subject_id VARCHAR(50),
      hadm_id VARCHAR(50),
      admittime TIMESTAMP,
      icu_time TIMESTAMP,
      target_time TIMESTAMP,
      target VARCHAR(50)
    );
    COPY mimic_cohort
    FROM %s
    DELIMITER ','
    CSV HEADER;
    """
    cur.execute(query, (file_path,))
    fetch_microbio(db_conn, model_type).to_csv(microbio_output_path, index=False)
    fetch_drugs(db_conn).to_csv(drugs_output_path, index=False)
    fetch_lab_weight_ethnicity(db_conn).to_csv(lab_weight_ethnicity_output_path, index=False)
    cur.close()
    return (microbio_output_path, drugs_output_path, lab_weight_ethnicity_output_path)

@debuggable
# Get microbiologyevents data, the days back changes between model a and model b
def fetch_microbio(db_conn, model_type):
    if model_type == 'a':
        time_back = 3 * 24 * 60 * 60
        cols = "spec_type_desc"
    else:
        time_back = 2 * 24 * 60 * 60
        cols = "spec_type_desc, org_name"
    query = """
    DROP TABLE IF EXISTS microbio;
    CREATE TABLE microbio as(
        SELECT DISTINCT
                identifier, {cols}
        FROM
            (select *, subject_id||'-'||hadm_id as identifier from mimiciii.microbiologyevents) as t0
            INNER JOIN (select identifier, target_time from mimic_cohort) _tmp2 using (identifier)
        WHERE
             identifier in (select identifier from mimic_cohort)
        AND
            (extract(epoch from target_time - chartdate)) > %s
    );
    """
    return get_table_df(db_conn, query, 'microbio', (cols, time_back))

@debuggable
# Get prescriptions data
def fetch_drugs(db_conn):
    query = """
    DROP TABLE IF EXISTS drugs;
    CREATE TABLE drugs as(
        SELECT
                DISTINCT identifier, drug
        FROM
            (select *, subject_id||'-'||hadm_id as identifier from mimiciii.prescriptions) as t0
            INNER JOIN (select identifier, target_time from mimic_cohort) _tmp2 using (identifier)
        WHERE
             identifier in (select identifier from mimic_cohort)
        AND
            (extract(epoch from target_time - t0.startdate)) > 0
    );
    """
    return get_table_df(db_conn, query, 'drugs')

@debuggable
def get_table_df(db_conn, query, table_name, microbio_cols_time_back_tuple=(None, None)):
    cur = db_conn.cursor()
    if table_name == 'microbio':
        cur.execute(sql.SQL(query).format(cols=sql.Identifier(microbio_cols_time_back_tuple[0])),
                    (microbio_cols_time_back_tuple[1],))
    else:
        cur.execute(query)
    cur.execute(sql.SQL("SELECT * FROM {};").format(sql.Identifier(table_name)))
    # Retrieve query results
    records = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    return pd.DataFrame(data=records, columns=colnames)

def fetch_lab_weight_ethnicity(db_conn):
    return get_table_df(db_conn, get_lab_weight_ethnicity_query(), 'relevant_events')

def get_lab_weight_ethnicity_query():
    return """/* (4_a) Create Table Format*/ 							  
    DROP TABLE IF EXISTS _relevantFeatures;
    create table _relevantFeatures(
    item_id INT,
    _table TEXT	
    );
    
    /* (4) Inset to the table all the IDs of the relevant features.*/
    insert into _relevantFeatures (item_id , _table) 
    values 	
        /* Complete Blood Count: */
        (50889, 'labevents'), -- C-Reactive Protein
        (51256, 'labevents'), -- Neturophils
        (51279, 'labevents'), -- Red Blood Cells
        (50811, 'labevents'), -- Hemoglobin
        (51221, 'labevents'), -- Hematocrit
        (51250, 'labevents'), -- MCV
        (51248, 'labevents'), -- MCH
        (51249, 'labevents'), -- MCHC
        (51277, 'labevents'), -- RDW
        (51244, 'labevents'), -- Lymphocytes
        (51254, 'labevents'), -- Monocytes
        (51200, 'labevents'), -- Eosinophils
        (51146, 'labevents'), -- Basophils
        (51265, 'labevents'), -- Platelet Count
        
        /* Basic Metabolic Panel: */
        (50971,'labevents'), -- Potassium
        (50983,'labevents'), -- Sodium
        (50912,'labevents'), -- Creatinine
        (50902,'labevents'), -- Chloride
        (51006,'labevents'), -- Urea Nitrogen
        (50882,'labevents'), -- Bicarbonate
        (50868,'labevents'), -- Anion Gap
        (50931,'labevents'), -- Glucose
        (50960,'labevents'), -- Magnesium
        (50893,'labevents'), -- Calcium, Total
        (50970,'labevents'), -- Phosphate
        (50820,'labevents'), -- pH
        
        /* Blood Gases: */
        (50802,'labevents'), -- Base Excess 
        (50804,'labevents'), -- Calculated Total CO2 
        (50821,'labevents'), -- pO2
        (50818,'labevents'), -- pCO2
        
        /* Cauglation Panel: */
        (51275,'labevents'), -- PTT
        (51237,'labevents'), -- INR(PT)
        (51274,'labevents'), -- PT
        
        -- Vital signs:
        (223762, 'chartevents'), -- Temp C, metavision
        (676, 'chartevents'), -- Temp C, careVue
        (220045, 'chartevents'), -- HearRate, MetaVision
        (211, 'chartevents'), -- HearRate, CareVue
        (220277, 'chartevents'), -- SpO2 saturation, metavision
        (646, 'chartevents'), -- SpO2 saturation, CareVue
        (220181, 'chartevents'), -- Non Invasive Blod Pressure (mean), MetaVision
        (456, 'chartevents') -- Non Invasive Blod Pressure (mean), CareVue
    ;			
    
    /* (5_a) Create tables of all relevant rows from labevents for the cohort: */
    DROP TABLE IF EXISTS relevant_labevents_for_cohort;
    CREATE TABLE relevant_labevents_for_cohort as (
    select subject_id||'-'||hadm_id as identifier, subject_id, hadm_id, itemid, charttime, valuenum, label
    from mimiciii.labevents join (select itemid, label from mimiciii.d_labitems) as t1 using (itemid)
    where subject_id||'-'||hadm_id in (select identifier from mimic_cohort) 
    AND itemid in (select item_id from _relevantFeatures where _table='labevents')
    );
    
    /* (5) Create tables of all relevant rows from chartevents for the cohort: */
    DROP TABLE IF EXISTS relevant_chartevents_for_cohort;
    CREATE TABLE relevant_chartevents_for_cohort as (
    select subject_id||'-'||hadm_id as identifier, subject_id, hadm_id, itemid, charttime, valuenum, label
    from mimiciii.chartevents join (select itemid, label from mimiciii.d_items) as t1 using (itemid)
    where subject_id||'-'||hadm_id in (select identifier from mimic_cohort) 
        AND itemid in (select item_id from _relevantFeatures where _table='chartevents')
    );
    
    /* (5_c) Create a unified table of feature from the tables created above:*/
    DROP TABLE IF EXISTS all_relevant_lab_features;
    CREATE TABLE all_relevant_lab_features as (
    select * from 
    relevant_chartevents_for_cohort
    union 
    (select * from relevant_labevents_for_cohort)
    );
    
    /* (5_d) Create a table of patients weight */
    DROP TABLE IF EXISTS patients_weight;
    CREATE TABLE patients_weight as(
    SELECT 
            subject_id||'-'||hadm_id as identifier, AVG(patientweight) as weight
    FROM 
        mimiciii.inputevents_mv
    WHERE 
         subject_id||'-'||hadm_id in (select identifier from mimic_cohort)
    GROUP BY 
        identifier
    );
    
    
    /* (5_f) Create a table of relevant events (features) received near when the target (culture) was received */
    DROP TABLE IF EXISTS relevant_events;
    CREATE TABLE relevant_events as(
    SELECT 
            *,
            date_part('year', admittime) - date_part('year', dob) as estimated_age,
            round(CAST((extract(epoch from target_time - all_relevant_lab_features.charttime) / 3600.0) as numeric),2) as hours_from_charttime_time_to_targettime,
            round(CAST((extract(epoch from charttime - admittime) / 3600.0 ) as numeric),2) as hours_from_admittime_to_charttime,
            round(CAST((extract(epoch from target_time - admittime) / 3600.0) as numeric),2) as hours_from_admittime_to_targettime
    FROM 
        all_relevant_lab_features			
        INNER JOIN (select identifier, target, target_time, admittime from mimic_cohort) _tmp2 using (identifier)
        INNER JOIN (select subject_id,gender, dob from mimiciii.patients where subject_id in (
                                        select CAST (subject_id as INTEGER) 
                                        from mimic_cohort)) as t3 	
                    using (subject_id)
        LEFT JOIN patients_weight using (identifier)
        LEFT JOIN (select subject_id||'-'||hadm_id as identifier, ethnicity from mimiciii.admissions) as _tmp1 using (identifier)
    WHERE 
         identifier in (select identifier from mimic_cohort)
    AND
        (extract(epoch from target_time - all_relevant_lab_features.charttime)) > 0
    );
    """


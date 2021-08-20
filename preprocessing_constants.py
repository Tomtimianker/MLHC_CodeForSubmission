'''
Conatents used in module_2
'''

# Used to convert the ethnicity names to be identical between MIMIC and eICU DBs
ETHNICITY_DICT = {
    "AMERICAN INDIAN/ALASKA NATIVE": "Other/Unknown",
    "ASIAN": "Other/Unknown",
    "ASIAN - ASIAN INDIAN": "Other/Unknown",
    "ASIAN - CAMBODIAN": "Other/Unknown",
    "ASIAN - CHINESE": "Other/Unknown",
    "ASIAN - VIETNAMESE": "Other/Unknown",
    "ASIAN - OTHER": "Other/Unknown",
    "BLACK/AFRICAN": "African American",
    "BLACK/AFRICAN AMERICAN": "African American",
    "BLACK/CAPE VERDEAN": "African American",
    "BLACK/HAITIAN": "African American",
    "HISPANIC OR LATINO": "Hispanic",
    "HISPANIC/LATINO - COLOMBIAN": "Hispanic",
    "HISPANIC/LATINO - MEXICAN": "Hispanic",
    "HISPANIC/LATINO - DOMINICAN": "Hispanic",
    "HISPANIC/LATINO - GUATEMALAN": "Hispanic",
    "HISPANIC/LATINO - PUERTO RICAN": "Hispanic",
    "MIDDLE EASTERN": "Other/Unknown",
    "MULTI RACE ETHNICITY": "Other/Unknown",
    "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER": "Other/Unknown",
    "OTHER": "Other/Unknown",
    "PATIENT DECLINED TO ANSWER": "Other/Unknown",
    "PORTUGUESE": "Other/Unknown",
    "UNABLE TO OBTAIN": "Other/Unknown",
    "UNKNOWN/NOT SPECIFIED": "Other/Unknown",
    "WHITE": "Caucasian",
    "WHITE - BRAZILIAN": "Caucasian",
    "WHITE - OTHER EUROPEAN": "Caucasian",
    "WHITE - RUSSIAN": "Caucasian"
}

# Used to convert the microbiology events names to be identical between MIMIC and eICU DBs
MICROBIOLOGY_DICT = {
    "SPUTUM": "Sputum, Expectorated",
    "ANORECTAL/VAGINAL CULTURE": "Rectal Swab",
    "ABSCESS": "Other",
    "ASPIRATE": "Other",
    "BILE": "Other",
    "BIOPSY": "Other",
    "BONE MARROW": "Other",
    "BONE MARROW - CYTOGENETICS": "Other",
    "BRONCHIAL BRUSH - PROTECTED": "Other",
    "BRONCHIAL WASHINGS": "Other",
    "DIALYSIS FLUID": "Other",
    "Direct Antigen Test for Herpes Simplex Virus Types 1 & 2": "Other",
    "DIRECT ANTIGEN TEST FOR VARICELLA-ZOSTER VIRUS": "Other",
    "EYE": "Other",
    "FLUID,OTHER": "Other",
    "FOOT CULTURE": "Other",
    "FOREIGN BODY": "Other",
    "IMMUNOLOGY": "Other",
    "Immunology (CMV)": "Other",
    "Influenza A/B by DFA": "Other",
    "JOINT FLUID": "Other",
    "Mini-BAL": "Other",
    "MRSA SCREEN": "Other",
    "Rapid Respiratory Viral Screen & Culture": "Other",
    "Staph aureus Screen": "Other",
    "SWAB": "Other",
    "VIRAL CULTURE: R/O CYTOMEGALOVIRUS": "Other",
    "URINE": "Urine, Catheter Specimen",
    "CATHETER TIP-IV": "Urine, Catheter Specimen",
    "THROAT CULTURE": "Sputum, Tracheal Specimen",
    "THROAT FOR STREP": "Sputum, Tracheal Specimen",
    "TRACHEAL ASPIRATE": "Sputum, Tracheal Specimen",
    "CSF;SPINAL FLUID": "CSF",
    "Blood (CMV AB)": "Blood, Venipuncture",
    "Blood (EBV)": "Blood, Venipuncture",
    "Blood (Malaria)": "Blood, Venipuncture",
    "Blood (Toxo)": "Blood, Venipuncture",
    "BLOOD CULTURE": "Blood, Venipuncture",
    "BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)": "Blood, Venipuncture",
    "SEROLOGY/BLOOD": "Blood, Venipuncture",
    "NEOPLASTIC BLOOD": "Blood, Venipuncture",
    "FLUID RECEIVED IN BLOOD CULTURE BOTTLES": "Blood, Venipuncture",
    "Stem Cell - Blood Culture": "Blood, Venipuncture",
    "BRONCHOALVEOLAR LAVAGE": "Bronchial Lavage",
    "FLUID WOUND": "Wound, Decubitus",
    "STOOL": "Stool",
    "STOOL (RECEIVED IN TRANSPORT SYSTEM)": "Stool",
    "URINE,KIDNEY": "Urine, Voided Specimen",
    "PLEURAL FLUID": "Pleural Fluid",
    "PERITONEAL FLUID": "Peritoneal Fluid",
    "TISSUE": "Skin",
}

# Used to remove non human value from the lab results
HUMAN_RANGES = {
    "age": (0, 120),
    "BMI": (10, 80),
    "WEI": (0, 250),
    "A1C": (0, 240),
    "A1CB": (0, 24),
    "ALB": (0, 100),
    "ALT": (0, 20000),
    "AST": (0, 20000),
    "IBIL": (0, 20),
    "DBIL": (0, 20),
    "BNP": (0, 10000),
    "BRE": (1, 100),
    "BUN": (2, 200),
    "Calcium, Total": (0, 20),
    "CKMB": (0, 10000),
    "CO2tV": (0, 100),
    "CPK": (0, 10000),
    "CRP": (0, 1000),
    "Creatinine": (0, 20),
    "DBP": (20, 240),
    "DDM": (0, 50),
    "FER": (0, 20000),
    "FIB": (0, 1500),
    "FIO2": (20, 100),
    "Glucose": (0, 2000),
    "HCO3": (0, 100),
    "Hemoglobin": (2, 25),
    "INR": (0.5, 5),
    "LCT": (0.2, 15),
    "LDH": (0, 50000),
    "LYM(#)": (0, 20),
    "Lymphocytes": (0.2, 100),
    "NEU(#)": (0.1, 60),
    "Neutrophils": (0.2, 100),
    "NRBC": (0, 100),
    "OSB": (50, 500),
    "OSU": (50, 2000),
    "PAO2": (0, 1000),
    "PAO2V": (0, 1000),
    "pCO2": (0, 150),
    "PEEP": (0, 20),
    "pH": (6.6, 7.8),
    "Platelet Count": (0, 1000),
    "PLTM": (0, 1000),
    "Potassium": (1, 10),
    "PTSec": (5, 50),
    "PTT": (5, 200),
    "PUL": (10, 300),
    "Red Blood Cells": (1, 8),
    "RDW": (5, 40),
    "RRM": (1, 30),
    "RRP": (1, 60),
    "SBP": (40, 250),
    "Sodium": (110, 200),
    "SpO2": (5, 100),
    "SUTUR": (5, 100),
    "SUTURO": (5, 100),
    "TBIL": (0, 20),
    "Temperature C": (20, 43),
    "TRG": (10, 2000),
    "TRP": (1, 40000),
    "TVM": (100, 1000),
    "TVP": (100, 3500),
    "VB1": (10, 100),
    "VB12": (100, 2500),
    "WBC": (0.2, 100)
}

ALL_FEATURES_MODEL_A = ["gender", "age", "ethnicity_African American", "ethnicity_Caucasian",
                        "ethnicity_Hispanic",
                        "ethnicity_Other/Unknown", "3_days_max_Anion Gap", "3_days_max_Base Excess",
                        "3_days_max_Basophils", "3_days_max_Bicarbonate", "3_days_max_Calcium, Total",
                        "3_days_max_Chloride", "3_days_max_Creatinine", "3_days_max_Eosinophils", "3_days_max_Glucose",
                        "3_days_max_Hematocrit", "3_days_max_Hemoglobin", "3_days_max_Lymphocytes", "3_days_max_MCH",
                        "3_days_max_MCHC", "3_days_max_MCV", "3_days_max_Magnesium", "3_days_max_Monocytes",
                        "3_days_max_PT", "3_days_max_PTT", "3_days_max_Phosphate", "3_days_max_Platelet Count",
                        "3_days_max_Potassium", "3_days_max_RDW", "3_days_max_Red Blood Cells", "3_days_max_Sodium",
                        "3_days_max_pCO2", "3_days_max_pH", "3_days_max_pO2", "3_days_min_Anion Gap",
                        "3_days_min_Base Excess", "3_days_min_Basophils", "3_days_min_Bicarbonate",
                        "3_days_min_Calcium, Total", "3_days_min_Chloride", "3_days_min_Creatinine",
                        "3_days_min_Eosinophils", "3_days_min_Glucose", "3_days_min_Hematocrit",
                        "3_days_min_Hemoglobin", "3_days_min_Lymphocytes", "3_days_min_MCH", "3_days_min_MCHC",
                        "3_days_min_MCV", "3_days_min_Magnesium", "3_days_min_Monocytes", "3_days_min_PT",
                        "3_days_min_PTT", "3_days_min_Phosphate", "3_days_min_Platelet Count", "3_days_min_Potassium",
                        "3_days_min_RDW", "3_days_min_Red Blood Cells", "3_days_min_Sodium", "3_days_min_pCO2",
                        "3_days_min_pH", "3_days_min_pO2", "3_days_avg_Anion Gap", "3_days_avg_Base Excess",
                        "3_days_avg_Basophils", "3_days_avg_Bicarbonate", "3_days_avg_Calcium, Total",
                        "3_days_avg_Chloride", "3_days_avg_Creatinine", "3_days_avg_Eosinophils", "3_days_avg_Glucose",
                        "3_days_avg_Hematocrit", "3_days_avg_Hemoglobin", "3_days_avg_Lymphocytes", "3_days_avg_MCH",
                        "3_days_avg_MCHC", "3_days_avg_MCV", "3_days_avg_Magnesium", "3_days_avg_Monocytes",
                        "3_days_avg_PT", "3_days_avg_PTT", "3_days_avg_Phosphate", "3_days_avg_Platelet Count",
                        "3_days_avg_Potassium", "3_days_avg_RDW", "3_days_avg_Red Blood Cells", "3_days_avg_Sodium",
                        "3_days_avg_pCO2", "3_days_avg_pH", "3_days_avg_pO2", "var_Anion Gap", "var_Base Excess",
                        "var_Basophils", "var_Bicarbonate", "var_Calcium, Total", "var_Chloride", "var_Creatinine",
                        "var_Eosinophils", "var_Glucose", "var_Hematocrit", "var_Hemoglobin", "var_Lymphocytes",
                        "var_MCH", "var_MCHC", "var_MCV", "var_Magnesium", "var_Monocytes", "var_PT", "var_PTT",
                        "var_Phosphate", "var_Platelet Count", "var_Potassium", "var_RDW", "var_Red Blood Cells",
                        "var_Sodium", "var_pCO2", "var_pO2", "culturesite_Blood, Venipuncture",
                        "culturesite_Bronchial Lavage", "culturesite_CSF", "culturesite_Other",
                        "culturesite_Peritoneal Fluid", "culturesite_Pleural Fluid", "culturesite_Rectal Swab",
                        "culturesite_Skin", "culturesite_Sputum, Expectorated", "culturesite_Sputum, Tracheal Specimen",
                        "culturesite_Stool", "culturesite_Urine, Catheter Specimen",
                        "culturesite_Urine, Voided Specimen", "culturesite_Wound, Decubitus", "drug_ACETAMINOPHEN",
                        "drug_ALBUMIN", "drug_ALBUTEROL", "drug_ALPRAZOLAM", "drug_AMIODARONE", "drug_AMLODIPINE",
                        "drug_ASPIRIN", "drug_AZITHROMYCIN", "drug_BACITRACIN", "drug_BISACODYL", "drug_CALCIUM",
                        "drug_CEFAZOLIN", "drug_CEFTRIAXONE", "drug_CHLORHEXIDINE", "drug_DEXAMETHASONE",
                        "drug_DEXTROSE", "drug_DILTIAZEM", "drug_DIPHENHYDRAMINE", "drug_DOCUSATE", "drug_ENOXAPARIN",
                        "drug_EPINEPHRINE", "drug_ETOMIDATE", "drug_FAMOTIDINE", "drug_FENTANYL", "drug_FUROSEMIDE",
                        "drug_GLUCAGON", "drug_HALOPERIDOL", "drug_HEPARIN", "drug_HYDRALAZINE", "drug_HYDROCORTISONE",
                        "drug_HYDROMORPHONE", "drug_INSULIN", "drug_KCL", "drug_LABETALOL", "drug_LACTULOSE",
                        "drug_LANTUS", "drug_LEVOFLOXACIN", "drug_LEVOTHYROXINE", "drug_LIDOCAINE", "drug_LISINOPRIL",
                        "drug_LORAZEPAM", "drug_MAGNESIUM", "drug_METHYLPREDNISOLONE", "drug_METOPROLOL",
                        "drug_METRONIDAZOLE", "drug_MIDAZOLAM", "drug_MORPHINE", "drug_MULTIVITAMIN",
                        "drug_MULTIVITAMINS", "drug_NOREPINEPHRINE", "drug_ONDANSETRON", "drug_OXYCODONE",
                        "drug_PANTOPRAZOLE", "drug_PHENYLEPHRINE", "drug_PHYTONADIONE", "drug_POTASSIUM",
                        "drug_PROPOFOL", "drug_QUETIAPINE", "drug_ROCURONIUM", "drug_SIMVASTATIN", "drug_SODIUM",
                        "drug_VANCOMYCIN"]

ALL_FEATURES_MODEL_B = ["gender", "estimated_age", "ethnicity_BLACK/AFRICAN AMERICAN",
                        "ethnicity_HISPANIC OR LATINO", "ethnicity_HISPANIC/LATINO - DOMINICAN", "ethnicity_OTHER",
                        "ethnicity_PATIENT DECLINED TO ANSWER", "ethnicity_UNABLE TO OBTAIN",
                        "ethnicity_UNKNOWN/NOT SPECIFIED", "ethnicity_WHITE", "4_days_max_Anion Gap",
                        "4_days_max_Base Excess", "4_days_max_Basophils", "4_days_max_Bicarbonate",
                        "4_days_max_Calcium, Total", "4_days_max_Calculated Total CO2", "4_days_max_Chloride",
                        "4_days_max_Creatinine", "4_days_max_Eosinophils", "4_days_max_Glucose",
                        "4_days_max_Heart Rate", "4_days_max_Hematocrit", "4_days_max_Hemoglobin", "4_days_max_INR(PT)",
                        "4_days_max_Lymphocytes", "4_days_max_MCH", "4_days_max_MCHC", "4_days_max_MCV",
                        "4_days_max_Magnesium", "4_days_max_Monocytes", "4_days_max_NBP Mean", "4_days_max_Neutrophils",
                        "4_days_max_PT", "4_days_max_PTT", "4_days_max_Phosphate", "4_days_max_Platelet Count",
                        "4_days_max_Potassium", "4_days_max_RDW", "4_days_max_Red Blood Cells", "4_days_max_Sodium",
                        "4_days_max_SpO2", "4_days_max_Urea Nitrogen", "4_days_max_pCO2", "4_days_max_pH",
                        "4_days_max_pO2", "4_days_min_Anion Gap", "4_days_min_Base Excess", "4_days_min_Basophils",
                        "4_days_min_Bicarbonate", "4_days_min_Calcium, Total", "4_days_min_Calculated Total CO2",
                        "4_days_min_Chloride", "4_days_min_Creatinine", "4_days_min_Eosinophils", "4_days_min_Glucose",
                        "4_days_min_Heart Rate", "4_days_min_Hematocrit", "4_days_min_Hemoglobin", "4_days_min_INR(PT)",
                        "4_days_min_Lymphocytes", "4_days_min_MCH", "4_days_min_MCHC", "4_days_min_MCV",
                        "4_days_min_Magnesium", "4_days_min_Monocytes", "4_days_min_NBP Mean", "4_days_min_Neutrophils",
                        "4_days_min_PT", "4_days_min_PTT", "4_days_min_Phosphate", "4_days_min_Platelet Count",
                        "4_days_min_Potassium", "4_days_min_RDW", "4_days_min_Red Blood Cells", "4_days_min_Sodium",
                        "4_days_min_SpO2", "4_days_min_Urea Nitrogen", "4_days_min_pCO2", "4_days_min_pH",
                        "4_days_min_pO2", "4_days_avg_Anion Gap", "4_days_avg_Base Excess", "4_days_avg_Basophils",
                        "4_days_avg_Bicarbonate", "4_days_avg_Calcium, Total", "4_days_avg_Calculated Total CO2",
                        "4_days_avg_Chloride", "4_days_avg_Creatinine", "4_days_avg_Eosinophils", "4_days_avg_Glucose",
                        "4_days_avg_Heart Rate", "4_days_avg_Hematocrit", "4_days_avg_Hemoglobin", "4_days_avg_INR(PT)",
                        "4_days_avg_Lymphocytes", "4_days_avg_MCH", "4_days_avg_MCHC", "4_days_avg_MCV",
                        "4_days_avg_Magnesium", "4_days_avg_Monocytes", "4_days_avg_NBP Mean", "4_days_avg_Neutrophils",
                        "4_days_avg_PT", "4_days_avg_PTT", "4_days_avg_Phosphate", "4_days_avg_Platelet Count",
                        "4_days_avg_Potassium", "4_days_avg_RDW", "4_days_avg_Red Blood Cells", "4_days_avg_Sodium",
                        "4_days_avg_SpO2", "4_days_avg_Urea Nitrogen", "4_days_avg_pCO2", "4_days_avg_pH",
                        "4_days_avg_pO2", "var_Anion Gap", "var_Base Excess", "var_Basophils", "var_Bicarbonate",
                        "var_Calcium, Total", "var_Calculated Total CO2", "var_Chloride", "var_Creatinine",
                        "var_Eosinophils", "var_Glucose", "var_Heart Rate", "var_Hematocrit", "var_Hemoglobin",
                        "var_INR(PT)", "var_Lymphocytes", "var_MCH", "var_MCHC", "var_MCV", "var_Magnesium",
                        "var_Monocytes", "var_NBP Mean", "var_Neutrophils", "var_PT", "var_PTT", "var_Phosphate",
                        "var_Platelet Count", "var_Potassium", "var_RDW", "var_Red Blood Cells", "var_Sodium",
                        "var_SpO2", "var_Urea Nitrogen", "var_pCO2", "var_pO2", "spec_type_desc_ABSCESS",
                        "spec_type_desc_BLOOD CULTURE", "spec_type_desc_BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)",
                        "spec_type_desc_BRONCHOALVEOLAR LAVAGE", "spec_type_desc_Blood (CMV AB)",
                        "spec_type_desc_Blood (EBV)", "spec_type_desc_Blood (Toxo)", "spec_type_desc_CATHETER TIP-IV",
                        "spec_type_desc_CSF;SPINAL FLUID", "spec_type_desc_FLUID RECEIVED IN BLOOD CULTURE BOTTLES",
                        "spec_type_desc_FLUID,OTHER", "spec_type_desc_IMMUNOLOGY", "spec_type_desc_Immunology (CMV)",
                        "spec_type_desc_Influenza A/B by DFA", "spec_type_desc_MRSA SCREEN",
                        "spec_type_desc_PERITONEAL FLUID", "spec_type_desc_PLEURAL FLUID",
                        "spec_type_desc_Rapid Respiratory Viral Screen & Culture", "spec_type_desc_SEROLOGY/BLOOD",
                        "spec_type_desc_SPUTUM", "spec_type_desc_STOOL", "spec_type_desc_SWAB",
                        "spec_type_desc_THROAT FOR STREP", "spec_type_desc_URINE", "org_name_2ND ISOLATE",
                        "org_name_ACINETOBACTER BAUMANNII", "org_name_BETA STREPTOCOCCI, NOT GROUP A",
                        "org_name_BETA STREPTOCOCCUS GROUP B", "org_name_CANDIDA ALBICANS",
                        "org_name_CLOSTRIDIUM DIFFICILE", "org_name_ENTEROBACTER AEROGENES",
                        "org_name_ENTEROCOCCUS FAECIUM", "org_name_ENTEROCOCCUS SP.", "org_name_ESCHERICHIA COLI",
                        "org_name_GRAM NEGATIVE ROD #1", "org_name_GRAM NEGATIVE ROD #2",
                        "org_name_GRAM NEGATIVE ROD #3", "org_name_GRAM NEGATIVE ROD(S)",
                        "org_name_GRAM POSITIVE BACTERIA", "org_name_HAEMOPHILUS INFLUENZAE, BETA-LACTAMASE NEGATIVE",
                        "org_name_MOLD", "org_name_MORAXELLA CATARRHALIS, PRESUMPTIVE IDENTIFICATION",
                        "org_name_NEISSERIA MENINGITIDIS", "org_name_POSITIVE FOR METHICILLIN RESISTANT STAPH AUREUS",
                        "org_name_PRESUMPTIVE CLOSTRIDIUM SEPTICUM", "org_name_PRESUMPTIVE PEPTOSTREPTOCOCCUS SPECIES",
                        "org_name_PSEUDOMONAS AERUGINOSA", "org_name_SERRATIA MARCESCENS",
                        "org_name_STAPH AUREUS COAG +", "org_name_STAPHYLOCOCCUS, COAGULASE NEGATIVE", "org_name_YEAST",
                        "org_name_YEAST, PRESUMPTIVELY NOT C. ALBICANS", "drug_ACDA", "drug_ACETAMINOPHEN",
                        "drug_ACETAMINOPHENCAFFBUTALBITAL", "drug_ACETAZOLAMIDE", "drug_ACETYLCYSTEINE", "drug_ACTONEL",
                        "drug_ACYCLOVIR", "drug_ADENOSINE", "drug_ALBUMIN", "drug_ALBUTEROL",
                        "drug_ALBUTEROLIPRATROPIUM", "drug_ALLOPURINOL", "drug_ALPRAZOLAM", "drug_ALTEPLASE",
                        "drug_ALUMINUM", "drug_ALUMINUMMAGNESIUM", "drug_AMIODARONE", "drug_AMLODIPINE",
                        "drug_AMOXICILLINCLAVULANIC", "drug_AMP", "drug_AMPICILLIN", "drug_AMPICILLINSULBACTAM",
                        "drug_ANTITHYMOCYTE", "drug_ARGATROBAN", "drug_ARTIFICIAL", "drug_ASPIRIN", "drug_ATENOLOL",
                        "drug_ATORVASTATIN", "drug_ATOVAQUONE", "drug_ATROPINE", "drug_AZITHROMYCIN", "drug_AZTREONAM",
                        "drug_BACITRACIN", "drug_BAG", "drug_BENZONATATE", "drug_BISACODYL", "drug_BRIMONIDINE",
                        "drug_BRINZOLAMIDE", "drug_BUMETANIDE", "drug_BUPIVACAINE", "drug_CALCITRIOL", "drug_CALCIUM",
                        "drug_CAPTOPRIL", "drug_CARBOPLATIN", "drug_CARVEDILOL", "drug_CASPOFUNGIN", "drug_CEFAZOLIN",
                        "drug_CEFEPIME", "drug_CEFTAZIDIME", "drug_CEFTRIAXONE", "drug_CEPACOL", "drug_CEPHALEXIN",
                        "drug_CETYLPYRIDINIUM", "drug_CHLORHEXIDINE", "drug_CHLOROTHIAZIDE", "drug_CIPROFLOXACIN",
                        "drug_CISATRACURIUM", "drug_CITALOPRAM", "drug_CITRATE", "drug_CITRIC", "drug_CLINDAMYCIN",
                        "drug_CLOFARABINE", "drug_CLONAZEPAM", "drug_CLONIDINE", "drug_CLOPIDOGREL", "drug_CLORAZEPATE",
                        "drug_CLOTRIMAZOLE", "drug_CODEINE", "drug_COLCHICINE", "drug_COSYNTROPIN", "drug_CREON",
                        "drug_CYANOCOBALAMIN", "drug_CYCLOBENZAPRINE", "drug_CYCLOPHOSPHAMIDE", "drug_CYTARABINE",
                        "drug_DAPTOMYCIN", "drug_DESITIN", "drug_DESMOPRESSIN", "drug_DEXAMETHASONE",
                        "drug_DEXMEDETOMIDINE", "drug_DEXTRAN", "drug_DEXTROSE", "drug_DIAZEPAM", "drug_DICYCLOMINE",
                        "drug_DIGOXIN", "drug_DILTIAZEM", "drug_DIPHENHYDRAMINE", "drug_DIPHENOXYLATEATROPINE",
                        "drug_DOBUTAMINE", "drug_DOCUSATE", "drug_DOLASETRON", "drug_DOPAMINE", "drug_DORZOLAMIDE",
                        "drug_DOXAZOSIN", "drug_DOXORUBICIN", "drug_DULOXETINE", "drug_ENALAPRIL", "drug_ENALAPRILAT",
                        "drug_ENOXAPARIN", "drug_EPIDURAL", "drug_EPINEPHRINE", "drug_EPOETIN", "drug_EPTIFIBATIDE",
                        "drug_ERYTHROMYCIN", "drug_ESCITALOPRAM", "drug_ESMOLOL", "drug_ETOMIDATE", "drug_ETOPOSIDE",
                        "drug_EZETIMIBE", "drug_FAMOTIDINE", "drug_FAT", "drug_FENOFIBRATE", "drug_FENTANYL",
                        "drug_FERROUS", "drug_FEXOFENADINE", "drug_FILGRASTIM", "drug_FISH", "drug_FLUCONAZOLE",
                        "drug_FLUDROCORTISONE", "drug_FLUTICASONE", "drug_FLUTICASONESALMETEROL", "drug_FOLIC",
                        "drug_FUROSEMIDE", "drug_GABAPENTIN", "drug_GANCICLOVIR", "drug_GENTAMICIN", "drug_GLIPIZIDE",
                        "drug_GLUCAGON", "drug_GLYBURIDE", "drug_GLYCOPYRROLATE", "drug_GOLYTELY", "drug_GUAIFENESIN",
                        "drug_GUAIFENESINDEXTROMETHORPHAN", "drug_HAEMOPHILUS", "drug_HALOPERIDOL", "drug_HEPARIN",
                        "drug_HESPAN", "drug_HUMULINR", "drug_HYDRALAZINE", "drug_HYDROCERIN",
                        "drug_HYDROCHLOROTHIAZIDE", "drug_HYDROCODONEACETAMINOPHEN", "drug_HYDROCORTISONE",
                        "drug_HYDROMORPHONE", "drug_HYDROMORPHONEHP", "drug_HYDROXYZINE", "drug_IBUPROFEN",
                        "drug_IMIPENEMCILASTATIN", "drug_INDINAVIR", "drug_INFLUENZA", "drug_INSULIN",
                        "drug_IPRATROPIUM", "drug_ISOOSMOTIC", "drug_ISOSORBIDE", "drug_ISOTONIC", "drug_KETOROLAC",
                        "drug_LABETALOL", "drug_LACRILUBE", "drug_LACTATED", "drug_LACTULOSE", "drug_LAMIVUDINE",
                        "drug_LANSOPRAZOLE", "drug_LATANOPROST", "drug_LEPIRUDIN", "drug_LEVALBUTEROL",
                        "drug_LEVETIRACETAM", "drug_LEVOFLOXACIN", "drug_LEVOTHYROXINE", "drug_LIDOCAINE",
                        "drug_LINEZOLID", "drug_LIOTHYRONINE", "drug_LISINOPRIL", "drug_LORAZEPAM", "drug_LR",
                        "drug_LYRICA", "drug_MAGNESIUM", "drug_MANNITOL", "drug_MECLIZINE", "drug_MEGESTROL",
                        "drug_MENINGOCOCCAL", "drug_MEPERIDINE", "drug_MERCAPTOPURINE", "drug_MEROPENEM",
                        "drug_MESALAMINE", "drug_METFORMIN", "drug_METHADONE", "drug_METHIMAZOLE", "drug_METHYLDOPA",
                        "drug_METHYLENE", "drug_METHYLPREDNISOLONE", "drug_METOCLOPRAMIDE", "drug_METOLAZONE",
                        "drug_METOPROLOL", "drug_METRONIDAZOLE", "drug_MEXILETINE", "drug_MICONAZOLE", "drug_MIDAZOLAM",
                        "drug_MIDODRINE", "drug_MILK", "drug_MILRINONE", "drug_MIRTAZAPINE", "drug_MISOPROSTOL",
                        "drug_MONTELUKAST", "drug_MORPHINE", "drug_MUCINEX", "drug_MULTI", "drug_MULTIVITAMIN",
                        "drug_MULTIVITAMINS", "drug_MUPIROCIN", "drug_MYCOPHENOLATE", "drug_NALOXONE", "drug_NEOMYCIN",
                        "drug_NEOSTIGMINE", "drug_NEPHROCAPS", "drug_NESIRITIDE", "drug_NEUTRAPHOS", "drug_NEXIUM",
                        "drug_NICARDIPINE", "drug_NICOTINE", "drug_NIMODIPINE", "drug_NITROFURANTOIN",
                        "drug_NITROGLYCERIN", "drug_NITROPRUSSIDE", "drug_NOREPINEPHRINE", "drug_NORMOCARB", "drug_NS",
                        "drug_NYSTATIN", "drug_OCTREOTIDE", "drug_OLANZAPINE", "drug_OMEPRAZOLE", "drug_ONDANSETRON",
                        "drug_OXACILLIN", "drug_OXAZEPAM", "drug_OXYCODONE", "drug_OXYCODONEACETAMINOPHEN",
                        "drug_OXYMETAZOLINE", "drug_PAMIDRONATE", "drug_PANTOPRAZOLE", "drug_PAROXETINE",
                        "drug_PENTOBARBITAL", "drug_PENTOXIFYLLINE", "drug_PHENAZOPYRIDINE", "drug_PHENOBARBITAL",
                        "drug_PHENTOLAMINE", "drug_PHENYLEPHRINE", "drug_PHENYTOIN", "drug_PHYTONADIONE",
                        "drug_PIPERACILLINTAZOBACTAM", "drug_PNEUMOCOCCAL", "drug_POLYETHYLENE", "drug_POTASSIUM",
                        "drug_PRAMOXINEMINERAL", "drug_PRAVASTATIN", "drug_PRAZOSIN", "drug_PREDNIS",
                        "drug_PREDNISOLONE", "drug_PREDNISONE", "drug_PRISMASATE", "drug_PROCHLORPERAZINE",
                        "drug_PROMETHAZINE", "drug_PROPOFOL", "drug_PROPOXYPHENE", "drug_PROPRANOLOL", "drug_PROTAMINE",
                        "drug_QUETIAPINE", "drug_QUININE", "drug_RACEPINEPHRINE", "drug_RALOXIFENE", "drug_RANITIDINE",
                        "drug_RIFAXIMIN", "drug_RISPERIDONE", "drug_RITONAVIR", "drug_SALMETEROL", "drug_SARNA",
                        "drug_SCOPOLAMINE", "drug_SENNA", "drug_SERTRALINE", "drug_SEVELAMER", "drug_SILDENAFIL",
                        "drug_SIMETHICONE", "drug_SIMVASTATIN", "drug_SIROLIMUS", "drug_SODIUM", "drug_SOLN",
                        "drug_SPIRONOLACTONE", "drug_STAVUDINE", "drug_STERILE", "drug_SUCCINYLCHOLINE",
                        "drug_SUCRALFATE", "drug_SW", "drug_SYRINGE", "drug_TACROLIMUS", "drug_TAMSULOSIN",
                        "drug_TETANUSDIPHTHERIA", "drug_THIAMINE", "drug_TIMOLOL", "drug_TIOTROPIUM", "drug_TIROFIBAN",
                        "drug_TIZANIDINE", "drug_TOBRAMYCIN", "drug_TOLTERODINE", "drug_TORSEMIDE", "drug_TRAMADOL",
                        "drug_TRAZODONE", "drug_TUBERCULIN", "drug_TUCKS", "drug_UNASYN", "drug_URSODIOL",
                        "drug_VALGANCICL", "drug_VALGANCICLOV", "drug_VALGANCICLOVIR", "drug_VALPROATE",
                        "drug_VALPROIC", "drug_VANCOMYCIN", "drug_VASOPRESSIN", "drug_VECURONIUM", "drug_VENLAFAXINE",
                        "drug_VIAL", "drug_VINCRISTINE", "drug_VITAMIN", "drug_VORICONAZOLE", "drug_WARFARIN",
                        "drug_XOPENEX", "drug_ZOLPIDEM"]

SELECTED_FEATURES_MODEL_A = ['gender', '3_days_max_Glucose', '3_days_max_Hematocrit',
                             '3_days_max_Lymphocytes', '3_days_max_Magnesium',
                             '3_days_max_Monocytes', '3_days_max_Platelet Count',
                             '3_days_max_Sodium', '3_days_max_pO2', '3_days_min_Base Excess',
                             '3_days_min_Hemoglobin', '3_days_min_PT', '3_days_min_Phosphate',
                             '3_days_min_Platelet Count', '3_days_min_Potassium',
                             '3_days_min_Sodium', '3_days_min_pO2', '3_days_avg_Glucose',
                             '3_days_avg_Magnesium', 'var_Anion Gap', 'var_Glucose',
                             'var_Hematocrit', 'var_Lymphocytes', 'var_pCO2',
                             'culturesite_Blood, Venipuncture', 'culturesite_Bronchial Lavage',
                             'culturesite_CSF', 'culturesite_Peritoneal Fluid',
                             'culturesite_Pleural Fluid', 'culturesite_Skin',
                             'culturesite_Sputum, Expectorated',
                             'culturesite_Sputum, Tracheal Specimen', 'culturesite_Stool',
                             'culturesite_Urine, Catheter Specimen', 'drug_ASPIRIN',
                             'drug_AZITHROMYCIN', 'drug_CALCIUM', 'drug_CHLORHEXIDINE',
                             'drug_EPINEPHRINE', 'drug_FENTANYL', 'drug_HYDRALAZINE',
                             'drug_HYDROCORTISONE', 'drug_HYDROMORPHONE', 'drug_LORAZEPAM',
                             'drug_METOPROLOL', 'drug_MORPHINE', 'drug_PANTOPRAZOLE', 'drug_SODIUM']

SELECTED_FEATURES_MODEL_B = ['4_days_max_Bicarbonate', '4_days_min_Anion Gap',
                             '4_days_min_Calcium, Total', '4_days_min_NBP Mean',
                             '4_days_avg_Anion Gap', '4_days_avg_INR(PT)',
                             '4_days_avg_Platelet Count', 'var_Bicarbonate',
                             'var_Calculated Total CO2', 'var_Hematocrit', 'var_Hemoglobin',
                             'var_INR(PT)', 'var_Lymphocytes', 'var_MCHC', 'var_NBP Mean',
                             'var_Neutrophils', 'var_PT', 'drug_VIAL']
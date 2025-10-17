import pandas as pd
import os

def load_mimic_tables(base_path="C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/testing data"):
    patients = pd.read_csv(os.path.join(base_path, "PATIENTS.csv"))
    admissions = pd.read_csv(os.path.join(base_path, "ADMISSIONS.csv"))
    diagnoses = pd.read_csv(os.path.join(base_path, "DIAGNOSES_ICD.csv"))
    diag_dict = pd.read_csv(os.path.join(base_path, "D_ICD_DIAGNOSES.csv"))
    labs = pd.read_csv(os.path.join(base_path, "LABEVENTS.csv"))
    lab_dict = pd.read_csv(os.path.join(base_path, "D_LABITEMS.csv"))
    return patients, admissions, diagnoses, diag_dict, labs, lab_dict

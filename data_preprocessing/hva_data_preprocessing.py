# %% Import package to use
from numpy.lib.function_base import disp
import pandas as pd
import datatable
import os
import glob
import numpy as np
from IPython.display import display
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns

# %% Processing the dataset

df_init = datatable.fread("../Data/processed_whole_set.csv", na_strings=['','NA'], encoding='utf-8-sig').to_pandas()
df_init = df_init[df_init['sex'] == 0].reset_index(drop=True)
display(df_init.head(1))

#%% Exlucde Oultier
print("Inclusion n = ", len(df_init['HPCID'].unique()))
#### Exclusion criteria
df_init['Exclusion'] = np.where(df_init[['Stroke', 'Angina', 'MI', 'Cancer']].isin([1]).any(axis=1), 1, 0)

#### Exclude outlier
for i in df_init[['SM3631', 'SM0104', 'SM3720']].columns:
    if i == 'SM3720':
        df_init = df_init[(df_init[i] <= df_init[i].quantile(0.999)) & (df_init[i] >= df_init[i].quantile(0.0001))]
    else:
        df_init = df_init[(df_init[i] >= df_init[i].quantile(0.005)) & (df_init[i] <= df_init[i].quantile(0.995))]

print("Number of population in set = {}".format(len(set(df_init['HPCID']))))

# %% Select dataset
df_selected = df_init[(df_init['Exclusion'] != 1)]
print("Number of healthy population = {}".format(len(set(df_selected['HPCID']))))

# %% Change columnn Name

columns_to_use = [
    'SM_DATE', 'HPCID', 'sex', 'AGE', 'SM0104', 'SM0101', 
    'SM0102', 'SM316001', 'MVPA', 'SM3631', 'Smoke', 'SM3720', 
    'SM0106', 'SM0111', 'SM0112', 'SM0126', 'SM0151', 'SM0152', 
    'SM0153', 'SM0154', 'SM0155', 'SM3140', 'SM3150', 'SM3170', 
    'CRP', 'CHOLESTEROL', 'TG', 'max_heart_rate', 'BMI_cal', 'BL3118', 
    'ASMI', 'VO2max', 'death', 'delta_time', 'Diabetes', 'Hypertension', 
    'HTN_med', 'Hyperlipidemia', 'Hepatatis', 'ALC', 'BL3142', 'BL314201', 
    'MBP', 'SM0600SBP', 'SM0600DBP', 'MED_HYPERTENSION', 'MED_HYPERLIPIDEMIA', 
    "RER_over_gs"
]

columns_to_rename = {
    'SM0104':'percentage_fat', 'SM0101':'Height', 
    'SM0102':'Weight', 'SM316001': 'BMI', 
    'SM3631':'rest_HR', 'SM3720':'CRF', 
    'SM0106':'비만도', 'SM0111':'Muscle_mass', 
    'SM0112':'복부지방율', 'SM0126':'부종검사', 
    'SM0151':'Muscle_mass(RA)', 'SM0152':'Muscle_mass(LA)', 
    'SM0153':'Muscle_mass(BODY)', 'SM0154':'Muscle_mass(RL)', 
    'SM0155':'Muscle_mass(LL)', 'SM3140':'체지방량', 
    'SM3150':'체수분량', 'SM3170':'제지방량', 'BL3142':"HDL_C", 
    'BL314201':'LDL_C', "SM0600SBP":"SBP", "SM0600DBP":"DBP", 
    "BL3118":'Glucose, Fasting'
}


df_selected_eq = df_selected[columns_to_use].rename(columns = columns_to_rename)
df_selected.rename(columns= columns_to_rename, inplace=True)

# %%
display(df_selected_eq.head())
# %%
for i in ['rest_HR', 'CRF', 'percentage_fat','ASMI']:
    print(i, "Qualtile")
    print("0% : ", df_selected_eq[i].quantile(0))
    print("1% : ", df_selected_eq[i].quantile(0.01))
    print("2% : ", df_selected_eq[i].quantile(0.02))

    print("98 : ", df_selected_eq[i].quantile(0.98))
    print("99% : ", df_selected_eq[i].quantile(0.99))
    print("100% : ", df_selected_eq[i].quantile(1))
# %%

df_selected.to_csv("./HVA_preprop_data.csv", encoding='utf-8-sig', index=False)
df_selected_eq.to_csv("./HVA_preprop_eq_data.csv", encoding='utf-8-sig', index=False)
# %%

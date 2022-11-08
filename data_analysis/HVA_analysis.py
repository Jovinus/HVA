# %%
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn import linear_model
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
pd.set_option("display.max_columns", None)
# %%
df_orig = pd.read_csv("./HVA_PS_matched.csv")
print(len(df_orig[df_orig['RER_over_gs'] == 1]))
df_orig = df_orig.sort_values(['HPCID', 'SM_DATE'], ascending=True).reset_index(drop=True)
df_orig = df_orig[(df_orig['AGE'] >= 40) & (df_orig['RER_over_gs'] == 1)].reset_index(drop=True).groupby(['HPCID']).head(1)
print(len(df_orig))
display(df_orig.head())

# %%
df_orig['sex'].value_counts()
df_orig['Pulse_Pressure'] = df_orig['SBP'] - df_orig['DBP']

# %%
bins = [39, 49, 59, 83]
df_orig['Age'] = pd.cut(df_orig['AGE'], bins=bins).astype(str)
df_orig['CRF_age_cat'] = df_orig.groupby('Age')['CRF'].apply(lambda x: pd.qcut(x=x, q=3, labels=['Low', 'Moderate', 'High']))

# mapper = {"(49, 60]":55, '(60, 70]':65, '(70, 80]':75, '(80, 90]':85}
# df_orig['Age'] = df_orig['Age'].astype(str).map(mapper).astype(float)

df_orig[['CRF_cat']] = pd.qcut(df_orig['CRF'], q=3, labels=['Low', 'Moderate', 'High'], retbins=True)[0]
# %%
from tableone import TableOne

columns = ['AGE', 'BMI', 'percentage_fat', 'Smoke', 'Hypertension', 'Diabetes', 'MED_HYPERTENSION', 
           'rest_HR', 'SBP', 'DBP', "CHOLESTEROL", 'HDL_C', 'LDL_C', 'TG', 'Glucose, Fasting', 'CRP', 'CRF', 'baPWV', 'mean_IMT', 'AJ_130_Score']


categorical = ['Smoke', 'Diabetes', 'Hypertension', 'MED_HYPERTENSION']

dem_table = TableOne(data=df_orig, columns=columns, categorical=categorical, groupby='HVA', pval=True)

display(dem_table)
# dem_table.to_excel("./table_1.xlsx")
# %% Figure 1

fig, ax = plt.subplots(1, figsize=(10,10))
prevalence = df_orig.groupby('Age')['HVA'].value_counts(normalize=True).reset_index(name='count')
prevalence['count'] = prevalence['count'] * 100
prevalence['Age'] = prevalence['Age'].astype(str).map({'(39, 49]':'40-49', '(49, 59]':'50-59', '(59, 83]':'60-83'})
print(prevalence)
sns.barplot(x='Age', y='count', data=prevalence[prevalence['HVA'] == 1], order=['40-49', '50-59', '60-83'])
plt.xlabel('Age', fontsize=18)
plt.ylabel('Prevalence of HVA, %', fontsize=20)
plt.ylim(0, 27)
plt.show()

# %% Figure 2

fig, ax = plt.subplots(1, figsize=(10,10))
prevalence = df_orig.groupby(['Age', 'CRF_age_cat'])['HVA'].value_counts(normalize=True).reset_index(name='count')
prevalence['count'] = prevalence['count'] * 100
prevalence['Age'] = prevalence['Age'].astype(str).map({'(39, 49]':'40-49', '(49, 59]':'50-59', '(59, 83]':'60-83'})
prevalence.rename(columns={'CRF_age_cat':"CRF"}, inplace=True)
print(prevalence)
sns.barplot(x='Age', y='count', data=prevalence[prevalence['HVA'] == 1], order=['40-49', '50-59', '60-83'], hue='CRF')
plt.xlabel('Age', fontsize=18)
plt.ylabel('Prevalence of HVA, %', fontsize=20)
plt.ylim(0, 35)
plt.legend(fontsize=20)
plt.show()

# %% Table 2
feature_mask = ['AGE', 'BMI', 'Diabetes', 'SBP', 'Smoke', 
                'Glucose, Fasting', 'CHOLESTEROL', 'CRP', 'CRF']

df_data = df_orig
df_data = df_data.fillna(df_data.median())

df_data_exog = sm.add_constant(df_data[feature_mask], prepend=False)
outcome = df_data['HVA']

mod = sm.Logit(outcome.astype(float), df_data_exog.astype(float))
res = mod.fit()
print(res.summary())
print("\nHazard Ratio:\n", np.exp(res.params), "\n", np.exp(res.conf_int()))
# %% Table 3 - CAC
from patsy import dmatrices

feature_mask = ['AGE', 'BMI', 'SBP', 'Smoke', 'CHOLESTEROL', 
                'Glucose, Fasting', 'CRP', 'Diabetes', 'HVA', 'CRF_cat']

df_data = df_orig[df_orig['AJ_130_Score'] >= 0]
df_data.rename(columns={'Glucose, Fasting':"Glucose"}, inplace=True)
df_data = df_data.fillna(df_data.median())

df_data['CAC_over'] = np.where(df_data['AJ_130_Score'] > 100, 1, 0)

model_expr = "CAC_over ~ AGE + BMI + SBP  + Smoke + CHOLESTEROL + Glucose + CRP + Diabetes + HVA + CRF_cat"
    
y, X = dmatrices(model_expr, df_data, return_type='dataframe')

mod = sm.Logit(y.astype(float), X.astype(float))
res = mod.fit()
print(res.summary())
print("\nHazard Ratio:\n", np.exp(res.params), "\n", np.exp(res.conf_int()))

# %%

from patsy import dmatrices

feature_mask = ['AGE', 'BMI', 'SBP', 'Smoke', 'CHOLESTEROL', 
                'Glucose, Fasting', 'CRP', 'Diabetes', 'HVA', 'CRF_cat']

df_data = df_orig[df_orig['AJ_130_Score'] >= 0]
df_data.rename(columns={'Glucose, Fasting':"Glucose"}, inplace=True)
df_data = df_data.fillna(df_data.median())

df_data['CAC_over'] = np.where(df_data['AJ_130_Score'] > 10, 1, 0)

model_expr = "CAC_over ~ AGE + BMI + SBP  + Smoke + CHOLESTEROL + Glucose + CRP + Diabetes + HVA + CRF_cat"
    
y, X = dmatrices(model_expr, df_data, return_type='dataframe')

mod = sm.Logit(y.astype(float), X.astype(float))
res = mod.fit()
print(res.summary())
print("\nHazard Ratio:\n", np.exp(res.params), "\n", np.exp(res.conf_int()))



# %% Table 3 - CIMT

feature_mask = ['AGE', 'BMI', 'SBP', 'Smoke', 'CHOLESTEROL', 
                'Glucose, Fasting', 'CRP', 'Diabetes', 'HVA', 'CRF_cat']

df_data = df_orig[df_orig['mean_IMT'].notnull()]
df_data.rename(columns={'Glucose, Fasting':"Glucose"}, inplace=True)
df_data = df_data.fillna(df_data.median())

df_data['CIMT'] = np.where(df_data['mean_IMT'] > df_data['mean_IMT'].quantile(0.75), 1, 0)

model_expr = "CIMT ~ AGE + BMI + SBP  + Smoke + CHOLESTEROL + Glucose + CRP + Diabetes + HVA + CRF_cat"
    
y, X = dmatrices(model_expr, df_data, return_type='dataframe')

mod = sm.Logit(y.astype(float), X.astype(float))
res = mod.fit()
print(res.summary())
print("\nHazard Ratio:\n", np.exp(res.params), "\n", np.exp(res.conf_int()))

# %% Table 3 - CIMT over 0.9

feature_mask = ['AGE', 'BMI', 'SBP', 'Smoke', 'CHOLESTEROL', 
                'Glucose, Fasting', 'CRP', 'Diabetes', 'HVA', 'CRF_cat']

df_data = df_orig[df_orig['mean_IMT'].notnull()]
df_data.rename(columns={'Glucose, Fasting':"Glucose"}, inplace=True)
df_data = df_data.fillna(df_data.median())

df_data['CIMT'] = np.where(df_data['mean_IMT'] > 0.9, 1, 0)

model_expr = "CIMT ~ AGE + BMI + SBP  + Smoke + CHOLESTEROL + Glucose + CRP + Diabetes + HVA + CRF_cat"
    
y, X = dmatrices(model_expr, df_data, return_type='dataframe')

mod = sm.Logit(y.astype(float), X.astype(float))
res = mod.fit()
print(res.summary())
print("\nHazard Ratio:\n", np.exp(res.params), "\n", np.exp(res.conf_int()))

# %% Table 3 - CIMT over 1.2

feature_mask = ['AGE', 'BMI', 'SBP', 'Smoke', 'CHOLESTEROL', 
                'Glucose, Fasting', 'CRP', 'Diabetes', 'HVA', 'CRF_cat']

df_data = df_orig[df_orig['mean_IMT'].notnull()]
df_data.rename(columns={'Glucose, Fasting':"Glucose"}, inplace=True)
df_data = df_data.fillna(df_data.median())

df_data['CIMT'] = np.where(df_data['mean_IMT'] > 1.2, 1, 0)

model_expr = "CIMT ~ AGE + BMI + SBP  + Smoke + CHOLESTEROL + Glucose + CRP + Diabetes + HVA + CRF_cat"
    
y, X = dmatrices(model_expr, df_data, return_type='dataframe')

mod = sm.Logit(y.astype(float), X.astype(float))
res = mod.fit()
print(res.summary())
print("\nHazard Ratio:\n", np.exp(res.params), "\n", np.exp(res.conf_int()))

# %% Table 4 - CRF unadjusted

feature_mask = ['CRF_cat']

df_data = df_orig
display(df_data.groupby('CRF_cat')['HVA'].value_counts())
display(df_data.groupby('CRF_cat')['HVA'].value_counts(normalize=True))

df_data = df_data.fillna(df_data.median())

model_expr = "HVA ~ CRF_cat"
    
y, X = dmatrices(model_expr, df_data, return_type='dataframe')

mod = sm.Logit(y.astype(float), X.astype(float))
res = mod.fit()
print(res.summary())
print("\nHazard Ratio:\n", np.exp(res.params), "\n", np.exp(res.conf_int()))

# %% Table 4 - CRF

feature_mask = ['AGE', 'BMI', 'SBP', 'Smoke', 'CHOLESTEROL', 
                'Glucose, Fasting', 'CRP', 'Diabetes', 'CRF_cat']

df_data = df_orig
df_data.rename(columns={'Glucose, Fasting':"Glucose"}, inplace=True)
df_data = df_data.fillna(df_data.median())

model_expr = "HVA ~ AGE + BMI + SBP  + Smoke + CHOLESTEROL + Glucose + CRP + Diabetes + CRF_cat"
    
y, X = dmatrices(model_expr, df_data, return_type='dataframe')

mod = sm.Logit(y.astype(float), X.astype(float))
res = mod.fit()
print(res.summary())
print("\nHazard Ratio:\n", np.exp(res.params), "\n", np.exp(res.conf_int()))

odds_ratio = np.exp(res.params)
odds_ratio_conf = np.exp(res.conf_int())

# %% Table 5 
from tableone import TableOne

columns = ['CRF']

dem_table = TableOne(data=df_orig[df_orig['Age'] == '(39, 49]'], columns=columns, groupby='CRF_age_cat', pval=True)

display(dem_table)

dem_table = TableOne(data=df_orig[df_orig['Age'] == '(49, 59]'], columns=columns, groupby='CRF_age_cat', pval=True)

display(dem_table)

dem_table = TableOne(data=df_orig[df_orig['Age'] == '(59, 83]'], columns=columns, groupby='CRF_age_cat', pval=True)

display(dem_table)

dem_table = TableOne(data=df_orig, columns=columns, groupby='CRF_age_cat', pval=True)

display(dem_table)
# %%

from tableone import TableOne

columns = ['SBP', 'baPWV', 'mean_IMT', 'AJ_130_Score']

dem_table = TableOne(data=df_orig[df_orig['Age'] == '(39, 49]'], columns=columns, groupby='CRF_age_cat', pval=True)
print(df_orig[df_orig['Age'] == '(39, 49]'].groupby(['CRF_age_cat'])['mean_IMT'].mean())
print(df_orig[df_orig['Age'] == '(39, 49]'].groupby(['CRF_age_cat'])['mean_IMT'].std())
display(dem_table)

dem_table = TableOne(data=df_orig[df_orig['Age'] == '(49, 59]'], columns=columns, groupby='CRF_age_cat', pval=True)

display(dem_table)

dem_table = TableOne(data=df_orig[df_orig['Age'] == '(59, 83]'], columns=columns, groupby='CRF_age_cat', pval=True)

display(dem_table)

dem_table = TableOne(data=df_orig, columns=columns, groupby='CRF_age_cat', pval=True)

display(dem_table)
# %% Table 1 Figure

from itertools import combinations
from statannot import add_stat_annotation
plt.figure(figsize=(10,10))
x = 'CRF_age_cat'
y = 'baPWV'
order = ['Low', 'Moderate', 'High']
ax = sns.barplot(data=df_orig, x=x, y=y, order=order)

ax, test_results = add_stat_annotation(ax, data=df_orig, x=x, y=y, order=order,
                                   box_pairs=[('Low', 'Moderate'), ('Moderate', 'High'), ('Low', 'High')],
                                   test='t-test_ind', text_format='star', loc='outside', verbose=2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.ylim(0, 1700)
plt.xlabel("")
plt.ylabel('baPWV', fontsize=20)
sns.despine()
# %%
from itertools import combinations
from statannot import add_stat_annotation
plt.figure(figsize=(10,10))
x = 'CRF_age_cat'
y = 'AJ_130_Score'
order = ['Low', 'Moderate', 'High']
ax = sns.barplot(data=df_orig, x=x, y=y, order=order)

ax, test_results = add_stat_annotation(ax, data=df_orig, x=x, y=y, order=order,
                                   box_pairs=[('Low', 'Moderate'), ('Moderate', 'High'), ('Low', 'High')],
                                   test='t-test_ind', text_format='star', loc='outside', verbose=2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.ylim(0, 110)
plt.xlabel("")
plt.ylabel('CAC(AJ-130 Score)', fontsize=20)
sns.despine()
# %%
from itertools import combinations
from statannot import add_stat_annotation
plt.figure(figsize=(10,10))
x = 'CRF_age_cat'
y = 'mean_IMT'
order = ['Low', 'Moderate', 'High']
ax = sns.barplot(data=df_orig, x=x, y=y, order=order)

ax, test_results = add_stat_annotation(ax, data=df_orig, x=x, y=y, order=order,
                                   box_pairs=[('Low', 'Moderate'), ('Moderate', 'High'), ('Low', 'High')],
                                   test='t-test_ind', text_format='star', loc='outside', verbose=2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.ylim(0, 1)
plt.xlabel("")
plt.ylabel('CIMT', fontsize=20)
sns.despine()
# %%

# %%

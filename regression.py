import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os, cv2, chilicv
from scipy.stats import linregress
import math
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

"""IDENTIFY SAMPLES, INITIAL MOISTURE AND TEMPERATURE"""
sample = 'cmb'; temp = 60
dataset = f"d://z/master/RaspberryPi/program/dataset/"
    
"""LOAD DATASET"""
def load_data(data_path):
    dataset = pd.read_csv(data_path)
    return dataset

ds_MR_60 = load_data(f"{dataset}correlate/{sample}_MR_60_updated.csv")
ds_MR_70 = load_data(f"{dataset}correlate/{sample}_MR_70_updated.csv")
ds_MR_80 = load_data(f"{dataset}correlate/{sample}_MR_80_updated.csv")
ds_CLR_60 = load_data(f"{dataset}correlate/{sample}_CLR_60_updated.csv")
ds_CLR_70 = load_data(f"{dataset}correlate/{sample}_CLR_70_updated.csv")
ds_CLR_80 = load_data(f"{dataset}correlate/{sample}_CLR_80_updated.csv")

MR_60 = ds_MR_60["MR_obs"]; MR_70 = ds_MR_70["MR_obs"]; MR_80 = ds_MR_80["MR_obs"]
L_60 = ds_CLR_60['L_norm']; L_70 = ds_CLR_70["L_norm"]; L_80 = ds_CLR_80["L_norm"]
a_60 = ds_CLR_60["a_norm"]; a_70 = ds_CLR_70["a_norm"]; a_80 = ds_CLR_80["a_norm"]
b_60 = ds_CLR_60["b_norm"]; b_70 = ds_CLR_70["b_norm"]; b_80 = ds_CLR_80["b_norm"]
C_60 = ds_CLR_60["C_norm"]; C_70 = ds_CLR_70["C_norm"]; C_80 = ds_CLR_80["C_norm"]
BI_60 = ds_CLR_60["BI_norm"]; BI_70 = ds_CLR_70["BI_norm"]; BI_80 = ds_CLR_80["BI_norm"]
YI_60 = ds_CLR_60["YI_norm"]; YI_70 = ds_CLR_70["YI_norm"]; YI_80 = ds_CLR_80["YI_norm"]

# Save to dataframe
df_60 = pd.DataFrame({
    "time": ds_MR_60["time"], "MR": MR_60, "L": L_60, "a": a_60, "b": b_60, "C": C_60, "BI": BI_60, "YI": YI_60
})
df_70 = pd.DataFrame({
    "time": ds_MR_70["time"], "MR": MR_70, "L": L_70, "a": a_70, "b": b_70, "C": C_70, "BI": BI_70, "YI": YI_70
})
df_80 = pd.DataFrame({
    "time": ds_MR_80["time"], "MR": MR_80, "L": L_80, "a": a_80, "b": b_80, "C": C_80, "BI": BI_80, "YI": YI_80
})

df_60.to_csv(f"{dataset}correlate/{sample}_df_60.csv", index=False)
df_70.to_csv(f"{dataset}correlate/{sample}_df_70.csv", index=False)
df_80.to_csv(f"{dataset}correlate/{sample}_df_80.csv", index=False)

# Set independent and dependent variables
df_60 = pd.read_csv(f"{dataset}correlate/{sample}_df_60.csv", index_col='time')
df_70 = pd.read_csv(f"{dataset}correlate/{sample}_df_70.csv", index_col='time')
df_80 = pd.read_csv(f"{dataset}correlate/{sample}_df_80.csv", index_col='time')

print(df_60.head())
def variable_data(data):
  y = data.iloc[: , 0]
  x = data.iloc[: , 1:]
  column = list(data.columns.values.tolist())
  feature_name = [str(x) for x in column[1:]]
  return y, x, feature_name

y_60, x_60, feature_name = variable_data(df_60) 
n_samples_60, n_features_60 = x_60.shape
print('Number of samples:', n_samples_60)
print('Number of features:', n_features_60)
print(df_60.head())

y_60 = df_60['MR'].to_numpy()
x_60 = sm.add_constant(df_60[['L', 'a', 'b', 'C', 'BI', 'YI']].to_numpy())

y_70 = df_70['MR'].to_numpy()
x_70 = sm.add_constant(df_70[['L', 'a', 'b', 'C', 'BI', 'YI']].to_numpy())

y_80 = df_80['MR'].to_numpy()
x_80 = sm.add_constant(df_80[['L', 'a', 'b', 'C', 'BI', 'YI']].to_numpy())

# Visualize the data using scatter plot and histogram, color=black
# sns.set_palette(['black'])
# sns.pairplot(data=df_60, height=3, markers=['o'], diag_kind='hist')
# plt.savefig(f"{dataset}correlate/{sample}_{temp}_pairplot.png")

# Visualize the distribution of each feature using histograms.
# plt.figure(figsize=(12, 12))
# for i, feature in enumerate(feature_name):
#     plt.subplot(4, 5, i+1)
#     sns.histplot(data=df_60, x=feature, kde=True)
#     plt.title(f'{feature} Distribution')
#     plt.tight_layout()
#     plt.savefig(f"{dataset}correlate/{sample}_{temp}_histplot.png")

# Correlation Heatmap, exclude time column
correlation_matrix = df_60.drop(columns=['time'], axis=1, errors='ignore').corr(numeric_only= True)
# Create a mask to hide the upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
plt.figure(figsize=(6, 5))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='gray_r', fmt=".2f", linewidths=0.5)
plt.title("Drying Temperature = 60°C")
plt.savefig(f"{dataset}correlate/{sample}_60_corrplot.png")

correlation_matrix = df_70.drop(columns=['time'], axis=1, errors='ignore').corr(numeric_only= True)
# Create a mask to hide the upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
plt.figure(figsize=(6,5))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='gray_r', fmt=".2f", linewidths=0.5)
plt.title("Drying Temperature = 70°C")
plt.savefig(f"{dataset}correlate/{sample}_70_corrplot.png")

correlation_matrix = df_80.drop(columns=['time'], axis=1, errors='ignore').corr(numeric_only= True)
# Create a mask to hide the upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
plt.figure(figsize=(6,5))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='gray_r', fmt=".2f", linewidths=0.5)
plt.title("Drying Temperature = 80°C")
plt.savefig(f"{dataset}correlate/{sample}_80_corrplot.png")

# fig, axs = plt.subplots(2, 3, figsize=(10,10))
# axs[0, 0].plot(x_[:,0], y_, 'o', label=feature_name[0])
# axs[0, 1].plot(x_[:,1], y_, 'o', label=feature_name[1])
# axs[0, 2].plot(x_[:,2], y_, 'o', label=feature_name[2])
# axs[1, 0].plot(x_[:,3], y_, 'o', label=feature_name[3])
# axs[1, 1].plot(x_[:,4], y_, 'o', label=feature_name[4])
# axs[1, 2].plot(x_[:,5], y_, 'o', label=feature_name[5])
# plt.legend()
# plt.savefig(f"{dataset}correlate/{sample}_{temp}_scatterplot.png")

print("\n" + "*"*80)
print("MULTIPLE LINEAR REGRESSION MODEL")
print("*"*80 + "\n")

# Calculate coefficients
beta_60 = np.linalg.inv(x_60.T @ x_60) @ (x_60.T @ y_60)
beta_70 = np.linalg.inv(x_70.T @ x_70) @ (x_70.T @ y_70)
beta_80 = np.linalg.inv(x_80.T @ x_80) @ (x_80.T @ y_80)

# Print the equation
equation_60 = f"y = {beta_60[0]:.5f}"
for i in range(1, len(beta_60)):
    equation_60 += f" + {beta_60[i]:.5f}x{i}"
print("Multiple Linear Regression (60°C)")
print(equation_60 + "\n")

equation_70 = f"y = {beta_70[0]:.5f}"
for i in range(1, len(beta_70)):
    equation_70 += f" + {beta_70[i]:.5f}x{i}"
print("Multiple Linear Regression (70°C)")
print(equation_70 + "\n")

equation_80 = f"y = {beta_80[0]:.5f}"
for i in range(1, len(beta_80)):
    equation_80 += f" + {beta_80[i]:.5f}x{i}"
print("Multiple Linear Regression (80°C)")
print(equation_80 + "\n")

# Define prediction function
def prediction(x, beta):
    pred = np.dot(x, beta)
    return pred

# Predict the dependent variable
y_pred_mlr_60 = prediction(x_60, beta_60)
y_pred_mlr_70 = prediction(x_70, beta_70)
y_pred_mlr_80 = prediction(x_80, beta_80)

# Model evaluation
mae_mlr_60 = np.mean(np.abs(y_pred_mlr_60 - y_60))
mse_mlr_60 = np.square(np.subtract(y_60, y_pred_mlr_60)).mean()
rmse_mlr_60 = math.sqrt(mse_mlr_60)
RSS_mlr_60 = np.sum(np.square((y_60 - y_pred_mlr_60)))
y_mean_60 = np.mean(y_60)
TSS_mlr_60 = np.sum(np.square(y_60 - y_mean_60))
r2score_mlr_60 = 1 - (RSS_mlr_60 / TSS_mlr_60)

mae_mlr_70 = np.mean(np.abs(y_pred_mlr_70 - y_70))
mse_mlr_70 = np.square(np.subtract(y_70, y_pred_mlr_70)).mean()
rmse_mlr_70 = math.sqrt(mse_mlr_70)
RSS_mlr_70 = np.sum(np.square((y_70 - y_pred_mlr_70)))
y_mean_70 = np.mean(y_70)
TSS_mlr_70 = np.sum(np.square(y_70 - y_mean_70))
r2score_mlr_70 = 1 - (RSS_mlr_70 / TSS_mlr_70)

mse_mlr_80 = np.square(np.subtract(y_80, y_pred_mlr_80)).mean()
rmse_mlr_80 = math.sqrt(mse_mlr_80)
RSS_mlr_80 = np.sum(np.square((y_80 - y_pred_mlr_80)))
y_mean_80 = np.mean(y_80)
TSS_mlr_80 = np.sum(np.square(y_80 - y_mean_80))
r2score_mlr_80 = 1 - (RSS_mlr_80 / TSS_mlr_80)

print(f"Coefficient of determination (R^2): {r2score_mlr_60:.3f}")
print(f"Root mean squared error (RMSE): {rmse_mlr_60:.3f}")

print(f"Coefficient of determination (R^2): {r2score_mlr_70:.3f}")
print(f"Root mean squared error (RMSE): {rmse_mlr_70:.3f}")

print(f"Coefficient of determination (R^2): {r2score_mlr_80:.3f}")
print(f"Root mean squared error (RMSE): {rmse_mlr_80:.3f}")

# Plot observed data vs predicted data
z_60 = np.polyfit(y_60, y_pred_mlr_60, 1)
z_70 = np.polyfit(y_70, y_pred_mlr_70, 1)
z_80 = np.polyfit(y_80, y_pred_mlr_80, 1)

plt.figure(figsize=(6,5))
plt.scatter(y_60, y_pred_mlr_60, marker='o', facecolors='white', edgecolors='black', label='MRobs vs MRpred 60°C')
plt.scatter(y_70, y_pred_mlr_70, marker='s', facecolors='black', edgecolors='black', label='MRobs vs MRpred 70°C')
plt.scatter(y_80, y_pred_mlr_80, marker='^', facecolors='white', edgecolors='black', label='MRobs vs MRpred 80°C')

plt.plot(y_60, z_60[0]*y_60 + z_60[1], 'k-.', lw=1.5, label='regression line 60°C')
plt.plot(y_70, z_70[0]*y_70 + z_70[1], 'k--', lw=1.5, label='regression line 70°C')
plt.plot(y_80, z_80[0]*y_80 + z_80[1], 'k:', lw=2, label='regression line 80°C')

plt.text(np.min(y_80), np.max(y_pred_mlr_80) - 0.00, f"R-squared (60°C): {r2score_mlr_60:.3f}")
plt.text(np.min(y_80), np.max(y_pred_mlr_80) - 0.05, f"RMSE (60°C): {rmse_mlr_60:.3f}")
plt.text(np.min(y_80), np.max(y_pred_mlr_80) - 0.1, f"R-squared (70°C): {r2score_mlr_70:.3f}")
plt.text(np.min(y_80), np.max(y_pred_mlr_80) - 0.15, f"RMSE (70°C): {rmse_mlr_70:.3f}")
plt.text(np.min(y_80), np.max(y_pred_mlr_80) - 0.2, f"R-squared (80°C): {r2score_mlr_80:.3f}")
plt.text(np.min(y_80), np.max(y_pred_mlr_80) - 0.25, f"RMSE (80°C): {rmse_mlr_80:.3f}")

plt.title('MLR Prediction')
plt.xlabel('Observed Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.savefig(f"{dataset}correlate/{sample}_MLR_predictedplot.png")

# Ordinary Least Square - Linear Regression
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
X = sm.add_constant(x_60)
ols_model_60 = sm.OLS(y_60, X).fit()
print(ols_model_60.summary())
print('R2 score:', ols_model_60.rsquared)
print('It means {} % of our dependent (response) variable can be explained using our independent (predictor) variables'.format(round(ols_model_60.rsquared*100, 2)))
print('F-statistic:', ols_model_60.fvalue)
print('Probability of observing value at least as high as F-statistic:', ols_model_60.f_pvalue)
if ols_model_60.f_pvalue<0.05:
    print('Because our f_pvalue is lower than 0.05 we can conclude that our model performs better than other simpler model.')
else:
    print('Because our f_pvalue is higher than 0.05 we can conclude that our model performs worse than other simpler model.')

X = sm.add_constant(x_70)
ols_model_70 = sm.OLS(y_70, X).fit()
print(ols_model_70.summary())
print('R2 score:', ols_model_70.rsquared)
print('It means {} % of our dependent (response) variable can be explained using our independent (predictor) variables'.format(round(ols_model_70.rsquared*100, 2)))
print('F-statistic:', ols_model_70.fvalue)
print('Probability of observing value at least as high as F-statistic:', ols_model_70.f_pvalue)
if ols_model_70.f_pvalue<0.05:
    print('Because our f_pvalue is lower than 0.05 we can conclude that our model performs better than other simpler model.')
else:
    print('Because our f_pvalue is higher than 0.05 we can conclude that our model performs worse than other simpler model.')

X = sm.add_constant(x_80)
ols_model_80 = sm.OLS(y_80, X).fit()
print(ols_model_80.summary())
print('R2 score:', ols_model_80.rsquared)
print('It means {} % of our dependent (response) variable can be explained using our independent (predictor) variables'.format(round(ols_model_80.rsquared*100, 2)))
print('F-statistic:', ols_model_80.fvalue)
print('Probability of observing value at least as high as F-statistic:', ols_model_80.f_pvalue)
if ols_model_80.f_pvalue<0.05:
    print('Because our f_pvalue is lower than 0.05 we can conclude that our model performs better than other simpler model.')
else:
    print('Because our f_pvalue is higher than 0.05 we can conclude that our model performs worse than other simpler model.')

print("\n" + "*"*80)
print("PRINCIPAL COMPONENT REGRESSION MODEL")
print("*"*80 + "\n")

# Principal Component Regression - PLS Regression (aka Partial Least Squares)
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
n_comps = 6

pca = PCA()
x_pca_60 = pca.fit_transform(x_60)[:,:n_comps]
x_pca_70 = pca.fit_transform(x_70)[:,:n_comps]
x_pca_80 = pca.fit_transform(x_80)[:,:n_comps]

# Scree plot
var_expl_60 = pca.fit(x_60).explained_variance_
var_expl_70 = pca.fit(x_70).explained_variance_
var_expl_80 = pca.fit(x_80).explained_variance_
for a in var_expl_60:
    if(a>0.5):
        print('Sorted Eignevalues 60 : {}'.format(round(a, 3)))
for a in var_expl_70:
    if(a>0.5):
        print('Sorted Eignevalues 70 : {}'.format(round(a, 3)))
for a in var_expl_80:
    if(a>0.5):
        print('Sorted Eignevalues 80 : {}'.format(round(a, 3)))

var_expl_ratio_60 = pca.fit(x_60).explained_variance_ratio_
var_expl_ratio_70 = pca.fit(x_70).explained_variance_ratio_
var_expl_ratio_80 = pca.fit(x_80).explained_variance_ratio_
for b in var_expl_ratio_60:
    if (b>(0.1/100)):
        print('Explained Variance 60 : {}%'.format(round(b*100, 2)))
for b in var_expl_ratio_70:
    if (b>(0.1/100)):
        print('Explained Variance 70 : {}%'.format(round(b*100, 2)))
for b in var_expl_ratio_80:
    if (b>(0.1/100)):
        print('Explained Variance 80 : {}%'.format(round(b*100, 2)))

with plt.style.context(('ggplot')):
    fig, ax = plt.subplots(figsize=(5,4))
    fig.set_tight_layout(True)
    ax.plot(np.arange(1, len(var_expl_ratio_60)+1), var_expl_ratio_60, '-o', label = 'Explained Variance (%) at temp = 60 C')
    ax.plot(np.arange(1, len(var_expl_ratio_60)+1), np.cumsum(var_expl_ratio_60), '-o', label = 'Cumulative Variance (%) at temp = 60 C')
    ax.plot(np.arange(1, len(var_expl_ratio_70)+1), var_expl_ratio_70, '-o', label = 'Explained Variance (%) at temp = 70 C')
    ax.plot(np.arange(1, len(var_expl_ratio_70)+1), np.cumsum(var_expl_ratio_70), '-o', label = 'Cumulative Variance (%) at temp = 70 C')
    ax.plot(np.arange(1, len(var_expl_ratio_80)+1), var_expl_ratio_80, '-o', label = 'Explained Variance (%) at temp = 80 C')
    ax.plot(np.arange(1, len(var_expl_ratio_80)+1), np.cumsum(var_expl_ratio_80), '-o', label = 'Cumulative Variance (%) at temp = 80 C')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_xlabel('Number of Components')
    ax.set_yticks(np.arange(0.0, 1.1, step=0.1))
    ax.set_xticks(np.arange(1, len(var_expl_ratio_60)+1, step=1))
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.set_facecolor("white")
plt.legend()
plt.savefig(f"{dataset}correlate/{sample}_pca_screeplot.png")

pc_list_60 = ['PC'+ str(i) for i in list(range(1, n_comps+1))]
pc_list_70 = ['PC'+ str(i) for i in list(range(1, n_comps+1))]
pc_list_80 = ['PC'+ str(i) for i in list(range(1, n_comps+1))]

"""PCA Loading or correlation coefficients"""
df_xcomps_60 = pd.DataFrame.from_dict(dict(zip(pc_list_60, x_pca_60)))
df_xcomps_70 = pd.DataFrame.from_dict(dict(zip(pc_list_70, x_pca_70)))
df_xcomps_80 = pd.DataFrame.from_dict(dict(zip(pc_list_80, x_pca_80)))
print('Projected X data\n',df_xcomps_60.head())
print('Projected X data\n',df_xcomps_70.head())
print('Projected X data\n',df_xcomps_80.head())

from sklearn.linear_model import LinearRegression
n_comps_new = 3
reg_model_60 = LinearRegression().fit(x_pca_60[:,:n_comps_new], y_60)
reg_model_70 = LinearRegression().fit(x_pca_70[:,:n_comps_new], y_70)
reg_model_80 = LinearRegression().fit(x_pca_80[:,:n_comps_new], y_80)

print(f"intercept: {reg_model_60.intercept_}")
print(f"slope: {reg_model_60.coef_}")
print(f"intercept: {reg_model_70.intercept_}")
print(f"slope: {reg_model_70.coef_}")
print(f"intercept: {reg_model_80.intercept_}")
print(f"slope: {reg_model_80.coef_}")

# Predict the dependent variable
y_pred_pcr_60 = reg_model_60.intercept_ + np.sum(reg_model_60.coef_ * x_pca_60[:,:n_comps_new], axis=1)
y_pred_pcr_70 = reg_model_70.intercept_ + np.sum(reg_model_70.coef_ * x_pca_70[:,:n_comps_new], axis=1)
y_pred_pcr_80 = reg_model_80.intercept_ + np.sum(reg_model_80.coef_ * x_pca_80[:,:n_comps_new], axis=1)

# Model evaluation
mae_pcr_60 = np.mean(np.abs(y_pred_pcr_60 - y_60))
mse_pcr_60 = np.square(np.subtract(y_60, y_pred_pcr_60)).mean()
rmse_pcr_60 = math.sqrt(mse_pcr_60)
RSS_pcr_60 = np.sum(np.square((y_60 - y_pred_pcr_60)))
y_mean_60 = np.mean(y_60)
TSS_pcr_60 = np.sum(np.square(y_60 - y_mean_60))
r2score_pcr_60 = 1 - (RSS_pcr_60 / TSS_pcr_60)
print(f"Coefficient of determination (R^2): {r2score_pcr_60:.3f}")
print(f"Root mean squared error (RMSE): {rmse_pcr_60:.3f}")

mae_pcr_70 = np.mean(np.abs(y_pred_pcr_70 - y_70))
mse_pcr_70 = np.square(np.subtract(y_70, y_pred_pcr_70)).mean()
rmse_pcr_70 = math.sqrt(mse_pcr_70)
RSS_pcr_70 = np.sum(np.square((y_70 - y_pred_pcr_70)))
y_mean_70 = np.mean(y_70)
TSS_pcr_70 = np.sum(np.square(y_70 - y_mean_70))
r2score_pcr_70 = 1 - (RSS_pcr_70 / TSS_pcr_70)
print(f"Coefficient of determination (R^2): {r2score_pcr_70:.3f}")
print(f"Root mean squared error (RMSE): {rmse_pcr_70:.3f}")

mae_pcr_80 = np.mean(np.abs(y_pred_pcr_80 - y_80))
mse_pcr_80 = np.square(np.subtract(y_80, y_pred_pcr_80)).mean()
rmse_pcr_80 = math.sqrt(mse_pcr_80)
RSS_pcr_80 = np.sum(np.square((y_80 - y_pred_pcr_80)))
y_mean_80 = np.mean(y_80)
TSS_pcr_80 = np.sum(np.square(y_80 - y_mean_80))
r2score_pcr_80 = 1 - (RSS_pcr_80 / TSS_pcr_80)
print(f"Coefficient of determination (R^2): {r2score_pcr_80:.3f}")
print(f"Root mean squared error (RMSE): {rmse_pcr_80:.3f}")

# Regression plot for PCR
# Plot observed data vs predicted data
z_60 = np.polyfit(y_60, y_pred_pcr_60, 1)
z_70 = np.polyfit(y_70, y_pred_pcr_70, 1)
z_80 = np.polyfit(y_80, y_pred_pcr_80, 1)

plt.figure(figsize=(6,5))
plt.scatter(y_60, y_pred_pcr_60, marker='o', facecolors='white', edgecolors='black', label='MRobs vs MRpred 60°C')
plt.scatter(y_70, y_pred_pcr_70, marker='s', facecolors='black', edgecolors='black', label='MRobs vs MRpred 70°C')
plt.scatter(y_80, y_pred_pcr_80, marker='^', facecolors='white', edgecolors='black', label='MRobs vs MRpred 80°C')

plt.plot(y_60, z_60[0]*y_60 + z_60[1], 'k-.', lw=1.5, label='regression line 60°C')
plt.plot(y_70, z_70[0]*y_70 + z_70[1], 'k--', lw=1.5, label='regression line 70°C')
plt.plot(y_80, z_80[0]*y_80 + z_80[1], 'k:', lw=2, label='regression line 80°C')

plt.text(np.min(y_80), np.max(y_pred_pcr_80) - 0.00, f"R-squared (60°C): {r2score_pcr_60:.3f}")
plt.text(np.min(y_80), np.max(y_pred_pcr_80) - 0.05, f"RMSE (60°C): {rmse_pcr_60:.3f}")
plt.text(np.min(y_80), np.max(y_pred_pcr_80) - 0.1, f"R-squared (70°C): {r2score_pcr_70:.3f}")
plt.text(np.min(y_80), np.max(y_pred_pcr_80) - 0.15, f"RMSE (70°C): {rmse_pcr_70:.3f}")
plt.text(np.min(y_80), np.max(y_pred_pcr_80) - 0.2, f"R-squared (80°C): {r2score_pcr_80:.3f}")
plt.text(np.min(y_80), np.max(y_pred_pcr_80) - 0.25, f"RMSE (80°C): {rmse_pcr_80:.3f}")

plt.title('PCR Prediction')
plt.xlabel('Observed Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.savefig(f"{dataset}correlate/{sample}_PCR_predictedplot.png")

print('\n' + "*"*80)
print("PARTIAL LEAST SQUARE REGRESSION MODEL")
print("*"*80 + "\n")

plsr60 = PLSRegression(n_components=6)

# Find the optimal number of components for PLS using cross-validation
plsr60.fit(x_60, y_60)
mse_cv_60 = []
for i in range(1, 7):
    plsr60 = PLSRegression(n_components=i)
    y_pred_cv = cross_val_predict(plsr60, x_60, y_60, cv=20)
    mse_cv_60.append(np.square(np.subtract(y_60, y_pred_cv)).mean())
optimal_n_components_60 = np.argmin(mse_cv_60) + 1
print(f"Optimal number of components: {optimal_n_components_60}")

plsr60 = PLSRegression(n_components=optimal_n_components_60)
plsr60.fit(x_60, y_60)
print(plsr60.coef_)
y_pred_plsr_60 = cross_val_predict(plsr60, x_60, y_60, cv=20)

# Find the optimal number of components for PLS using cross-validation
plsr70 = PLSRegression(n_components=6)
plsr70.fit(x_70, y_70)
mse_cv_70 = []
for i in range(1, 7):
    plsr70 = PLSRegression(n_components=i)
    y_pred_cv = cross_val_predict(plsr70, x_70, y_70, cv=20)
    mse_cv_70.append(np.square(np.subtract(y_70, y_pred_cv)).mean())
optimal_n_components_70 = np.argmin(mse_cv_70) + 1
print(f"Optimal number of components: {optimal_n_components_60}")

plsr70 = PLSRegression(n_components=optimal_n_components_70)
plsr70.fit(x_70, y_70)
print(plsr70.coef_)
y_pred_plsr_70 = cross_val_predict(plsr70, x_70, y_70, cv=20)

# Find the optimal number of components for PLS using cross-validation
plsr80 = PLSRegression(n_components=6)
plsr80.fit(x_80, y_80)
mse_cv_80 = []
for i in range(1, 7):
    plsr80 = PLSRegression(n_components=i)
    y_pred_cv = cross_val_predict(plsr80, x_80, y_80, cv=20)
    mse_cv_80.append(np.square(np.subtract(y_80, y_pred_cv)).mean())
optimal_n_components_80 = np.argmin(mse_cv_80) + 1
print(f"Optimal number of components: {optimal_n_components_60}")

plsr80 = PLSRegression(n_components=optimal_n_components_80)
plsr80.fit(x_80, y_80)
print(plsr80.coef_)
y_pred_plsr_80 = cross_val_predict(plsr80, x_80, y_80, cv=20)

# Model evaluation
mae_plsr_60 = np.mean(np.abs(y_pred_plsr_60 - y_60))
mse_plsr_60 = np.square(np.subtract(y_60, y_pred_plsr_60)).mean()
rmse_plsr_60 = math.sqrt(mse_plsr_60)
RSS_plsr_60 = np.sum(np.square((y_60 - y_pred_plsr_60)))
y_mean_60 = np.mean(y_60)
TSS_plsr_60 = np.sum(np.square(y_60 - y_mean_60))
r2score_plsr_60 = 1 - (RSS_plsr_60 / TSS_plsr_60)
print(f"Coefficient of determination (R^2): {r2score_plsr_60:.3f}")
print(f"Root mean squared error (RMSE): {rmse_plsr_60:.3f}")

mae_plsr_70 = np.mean(np.abs(y_pred_plsr_70 - y_70))
mse_plsr_70 = np.square(np.subtract(y_70, y_pred_plsr_70)).mean()
rmse_plsr_70 = math.sqrt(mse_plsr_70)
RSS_plsr_70 = np.sum(np.square((y_70 - y_pred_plsr_70)))
y_mean_70 = np.mean(y_70)
TSS_plsr_70 = np.sum(np.square(y_70 - y_mean_70))
r2score_plsr_70 = 1 - (RSS_plsr_70 / TSS_plsr_70)
print(f"Coefficient of determination (R^2): {r2score_plsr_70:.3f}")
print(f"Root mean squared error (RMSE): {rmse_plsr_70:.3f}")

mae_plsr_80 = np.mean(np.abs(y_pred_plsr_80 - y_80))
mse_plsr_80 = np.square(np.subtract(y_80, y_pred_plsr_80)).mean()
rmse_plsr_80 = math.sqrt(mse_plsr_80)
RSS_plsr_80 = np.sum(np.square((y_80 - y_pred_plsr_80)))
y_mean_80 = np.mean(y_80)
TSS_plsr_80 = np.sum(np.square(y_80 - y_mean_80))
r2score_plsr_80 = 1 - (RSS_plsr_80 / TSS_plsr_80)
print(f"Coefficient of determination (R^2): {r2score_plsr_80:.3f}")
print(f"Root mean squared error (RMSE): {rmse_plsr_80:.3f}")

# Regression plot for PLS
z_60 = np.polyfit(y_60, y_pred_plsr_60, 1)
z_70 = np.polyfit(y_70, y_pred_plsr_70, 1)
z_80 = np.polyfit(y_80, y_pred_plsr_80, 1)

plt.figure(figsize=(6,5))
plt.scatter(y_60, y_pred_plsr_60, marker='o', facecolors='white', edgecolors='black', label='MRobs vs MRpred 60°C')
plt.scatter(y_70, y_pred_plsr_70, marker='s', facecolors='black', edgecolors='black', label='MRobs vs MRpred 70°C')
plt.scatter(y_80, y_pred_plsr_80, marker='^', facecolors='white', edgecolors='black', label='MRobs vs MRpred 80°C')

plt.plot(y_60, z_60[0]*y_60 + z_60[1], 'k-.', lw=1.5, label='regression line 60°C')
plt.plot(y_70, z_70[0]*y_70 + z_70[1], 'k--', lw=1.5, label='regression line 70°C')
plt.plot(y_80, z_80[0]*y_80 + z_80[1], 'k:', lw=2, label='regression line 80°C')

plt.text(np.min(y_80), np.max(y_pred_plsr_80) - 0.00, f"R-squared (60°C): {r2score_plsr_60:.3f}")
plt.text(np.min(y_80), np.max(y_pred_plsr_80) - 0.05, f"RMSE (60°C): {rmse_plsr_60:.3f}")
plt.text(np.min(y_80), np.max(y_pred_plsr_80) - 0.1, f"R-squared (70°C): {r2score_plsr_70:.3f}")
plt.text(np.min(y_80), np.max(y_pred_plsr_80) - 0.15, f"RMSE (70°C): {rmse_plsr_70:.3f}")
plt.text(np.min(y_80), np.max(y_pred_plsr_80) - 0.2, f"R-squared (80°C): {r2score_plsr_80:.3f}")
plt.text(np.min(y_80), np.max(y_pred_plsr_80) - 0.25, f"RMSE (80°C): {rmse_plsr_80:.3f}")

plt.title('PLS Prediction')
plt.xlabel('Observed Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.savefig(f"{dataset}correlate/{sample}_PLSR_predictedplot.png")

print('\n' + "*"*80)
print("SUPPORT VECTOR REGRESSION MODEL")
print("*"*80 + "\n")
from sklearn.svm import SVR
# Support vector machine (SVM) regression
svr_60 = SVR(kernel='rbf', C=1000, epsilon=0.001)
svr_70 = SVR(kernel='rbf', C=1000, epsilon=0.001)
svr_80 = SVR(kernel='rbf', C=1000, epsilon=0.001)

svr_60.fit(x_60, y_60)
y_pred_svr_60 = svr_60.predict(x_60)

svr_70.fit(x_70, y_70)
y_pred_svr_70 = svr_70.predict(x_70)

svr_80.fit(x_80, y_80)
y_pred_svr_80 = svr_80.predict(x_80)

# Model evaluation
mae_svr_60 = np.mean(np.abs(y_pred_svr_60 - y_60))
mse_svr_60 = np.square(np.subtract(y_60, y_pred_svr_60)).mean()
rmse_svr_60 = math.sqrt(mse_svr_60)
RSS_svr_60 = np.sum(np.square((y_60 - y_pred_svr_60)))
y_mean_60 = np.mean(y_60)
TSS_svr_60 = np.sum(np.square(y_60 - y_mean_60))
r2score_svr_60 = 1 - (RSS_svr_60 / TSS_svr_60)
print(f"Coefficient of determination (R^2): {r2score_svr_60:.3f}")
print(f"Root mean squared error (RMSE): {rmse_svr_60:.3f}")

mae_svr_70 = np.mean(np.abs(y_pred_svr_70 - y_70))
mse_svr_70 = np.square(np.subtract(y_70, y_pred_svr_70)).mean()
rmse_svr_70 = math.sqrt(mse_svr_70)
RSS_svr_70 = np.sum(np.square((y_70 - y_pred_svr_70)))
y_mean_70 = np.mean(y_70)
TSS_svr_70 = np.sum(np.square(y_70 - y_mean_70))
r2score_svr_70 = 1 - (RSS_svr_70 / TSS_svr_70)
print(f"Coefficient of determination (R^2): {r2score_svr_70:.3f}")
print(f"Root mean squared error (RMSE): {rmse_svr_70:.3f}")

mae_svr_80 = np.mean(np.abs(y_pred_svr_80 - y_80))
mse_svr_80 = np.square(np.subtract(y_80, y_pred_svr_80)).mean()
rmse_svr_80 = math.sqrt(mse_svr_80)
RSS_svr_80 = np.sum(np.square((y_80 - y_pred_svr_80)))
y_mean_80 = np.mean(y_80)
TSS_svr_80 = np.sum(np.square(y_80 - y_mean_80))
r2score_svr_80 = 1 - (RSS_svr_80 / TSS_svr_80)
print(f"Coefficient of determination (R^2): {r2score_svr_80:.3f}")
print(f"Root mean squared error (RMSE): {rmse_svr_80:.3f}")

# Regression plot for SVR
z_60 = np.polyfit(y_60, y_pred_svr_60, 1)
z_70 = np.polyfit(y_70, y_pred_svr_70, 1)
z_80 = np.polyfit(y_80, y_pred_svr_80, 1)

plt.figure(figsize=(6,5))
plt.scatter(y_60, y_pred_svr_60, marker='o', facecolors='white', edgecolors='black', label='MRobs vs MRpred 60°C')
plt.scatter(y_70, y_pred_svr_70, marker='s', facecolors='black', edgecolors='black', label='MRobs vs MRpred 70°C')
plt.scatter(y_80, y_pred_svr_80, marker='^', facecolors='white', edgecolors='black', label='MRobs vs MRpred 80°C')

plt.plot(y_60, z_60[0]*y_60 + z_60[1], 'k-.', lw=1.5, label='regression line 60°C')
plt.plot(y_70, z_70[0]*y_70 + z_70[1], 'k--', lw=1.5, label='regression line 70°C')
plt.plot(y_80, z_80[0]*y_80 + z_80[1], 'k:', lw=2, label='regression line 80°C')

plt.text(np.min(y_80), np.max(y_pred_svr_80) - 0.00, f"R-squared (60°C): {r2score_svr_60:.3f}")
plt.text(np.min(y_80), np.max(y_pred_svr_80) - 0.05, f"RMSE (60°C): {rmse_svr_60:.3f}")
plt.text(np.min(y_80), np.max(y_pred_svr_80) - 0.1, f"R-squared (70°C): {r2score_svr_70:.3f}")
plt.text(np.min(y_80), np.max(y_pred_svr_80) - 0.15, f"RMSE (70°C): {rmse_svr_70:.3f}")
plt.text(np.min(y_80), np.max(y_pred_svr_80) - 0.2, f"R-squared (80°C): {r2score_svr_80:.3f}")
plt.text(np.min(y_80), np.max(y_pred_svr_80) - 0.25, f"RMSE (80°C): {rmse_svr_80:.3f}")

plt.title('SVR Prediction')
plt.xlabel('Observed Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.savefig(f"{dataset}correlate/{sample}_SVR_predictedplot.png")

# Print overall results shown in terminal and save to txt
with open(f"{dataset}correlate/{sample}_regression_results.txt", 'w') as f:
    f.write("MULTIPLE LINEAR REGRESSION MODEL\n")
    f.write("="*80 + "\n\n")
    f.write(f"Multiple Linear Regression\n")
    f.write(f"R-squared (60°C): {r2score_mlr_60:.3f}\n")
    f.write(f"RMSE (60°C): {rmse_mlr_60:.3f}\n")
    f.write(f"R-squared (70°C): {r2score_mlr_70:.3f}\n")
    f.write(f"RMSE (70°C): {rmse_mlr_70:.3f}\n")
    f.write(f"R-squared (80°C): {r2score_mlr_80:.3f}\n")
    f.write(f"RMSE (80°C): {rmse_mlr_80:.3f}\n\n")

    f.write("PRINCIPAL COMPONENT REGRESSION MODEL\n")
    f.write("="*80 + "\n\n")
    f.write(f"Principal Component Regression\n")
    f.write(f"R-squared (60°C): {r2score_pcr_60:.3f}\n")
    f.write(f"RMSE (60°C): {rmse_pcr_60:.3f}\n")
    f.write(f"R-squared (70°C): {r2score_pcr_70:.3f}\n")
    f.write(f"RMSE (70°C): {rmse_pcr_70:.3f}\n")
    f.write(f"R-squared (80°C): {r2score_pcr_80:.3f}\n")
    f.write(f"RMSE (80°C): {rmse_pcr_80:.3f}\n\n")

    f.write("PARTIAL LEAST SQUARE REGRESSION MODEL\n")
    f.write("="*80 + "\n\n")
    f.write(f"Partial Least Square Regression\n")
    f.write(f"R-squared (60°C): {r2score_plsr_60:.3f}\n")
    f.write(f"RMSE (60°C): {rmse_plsr_60:.3f}\n")
    f.write(f"R-squared (70°C): {r2score_plsr_70:.3f}\n")
    f.write(f"RMSE (70°C): {rmse_plsr_70:.3f}\n")
    f.write(f"R-squared (80°C): {r2score_plsr_80:.3f}\n")
    f.write(f"RMSE (80°C): {rmse_plsr_80:.3f}\n\n")

    f.write("SUPPORT VECTOR REGRESSION MODEL\n")
    f.write("="*80 + "\n\n")
    f.write(f"Support Vector Regression\n")
    f.write(f"R-squared (60°C): {r2score_svr_60:.3f}\n")
    f.write(f"RMSE (60°C): {rmse_svr_60:.3f}\n")
    f.write(f"R-squared (70°C): {r2score_svr_70:.3f}\n")
    f.write(f"RMSE (70°C): {rmse_svr_70:.3f}\n")
    f.write(f"R-squared (80°C): {r2score_svr_80:.3f}\n")
    f.write(f"RMSE (80°C): {rmse_svr_80:.3f}\n\n")

    f.write("MODEL COMPARISON\n")
    f.write("="*80 + "\n\n")
    f.write(f"Temperature (°C)\tMLR\tPCR\tPLSR\tSVR\n")
    f.write(f"60\t{r2score_mlr_60:.3f}\t{r2score_pcr_60:.3f}\t{r2score_plsr_60:.3f}\t{r2score_svr_60:.3f}\n")
    f.write(f"70\t{r2score_mlr_70:.3f}\t{r2score_pcr_70:.3f}\t{r2score_plsr_70:.3f}\t{r2score_svr_70:.3f}\n")
    f.write(f"80\t{r2score_mlr_80:.3f}\t{r2score_pcr_80:.3f}\t{r2score_plsr_80:.3f}\t{r2score_svr_80:.3f}\n")

    f.write(f"Temperature (°C)\tMLR\tPCR\tPLSR\tSVR\n")
    f.write(f"60\t{rmse_mlr_60:.3f}\t{rmse_pcr_60:.3f}\t{rmse_plsr_60:.3f}\t{rmse_svr_60:.3f}\n")
    f.write(f"70\t{rmse_mlr_70:.3f}\t{rmse_pcr_70:.3f}\t{rmse_plsr_70:.3f}\t{rmse_svr_70:.3f}\n")
    f.write(f"80\t{rmse_mlr_80:.3f}\t{rmse_pcr_80:.3f}\t{rmse_plsr_80:.3f}\t{rmse_svr_80:.3f}\n")

    f.write("\n\n" + "*"*80 + "\n")
    f.write("DATA VISUALIZATION\n")
    f.write("*"*80 + "\n\n")
    f.write("Results saved to:")
    f.write(f"\n60°C: {dataset}correlate/{sample}_regression_predictedplot.png")
    f.write(f"\n70°C: {dataset}correlate/{sample}_regression_predictedplot.png")
    f.write(f"\n80°C: {dataset}correlate/{sample}_regression_predictedplot.png")
    f.write(f"\nComparison: {dataset}correlate/{sample}_regression_comparisonplot.png")
    f.write(f"\nAll plots saved in {dataset}correlate/\n")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 13:22:17 2017

@author: Andreas Georgopoulos
"""
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd
import numpy as np
import math



# Import dataframe (produced at preprocessing.py)
movies_data_modelling = pd.read_csv('final_data/movies_data_df_final_fillna_modelling.csv')

# Check if NA's
movies_data_modelling.isnull().sum()
movies_data_modelling.isnull().sum().sum()


# Remove movies with box office > 320m
#movies_data_modelling = movies_data_modelling[(movies_data_modelling.Domestic_Box_Office > 10000) & (movies_data_modelling.Domestic_Box_Office < 320000000)]
movies_data_modelling = movies_data_modelling[(movies_data_modelling.Domestic_Box_Office < 320000000)].reset_index(drop = True)


# Exclude target variable (Domestic Box Office) from regressors df
target = movies_data_modelling.pop('Domestic_Box_Office')
regressors = movies_data_modelling




"""
    ###########################################################################
    ######################## Multicollienarity (VIF) ##########################
    ###########################################################################
"""

regressors = regressors.T.drop_duplicates(keep='last').T

# Compute VIF
#vif = pd.DataFrame()
#vif["VIF Factor"] = [variance_inflation_factor(regressors.values, i) for i in range(regressors.shape[1])]
#vif["features"] = regressors.columns
#vif.round(1)
                                  
#vif.loc[vif["VIF Factor"] >10]

# Remove Features with VIF>10
#regressors.drop(['Year', 'Multiple_Genres','ambush','gore'], axis = 1, inplace = True)
regressors.drop(['Year', 'Multiple_Genres'], axis = 1, inplace = True)



"""
    ###########################################################################
    ########################## Outlier Detection ##############################
    ###########################################################################
"""


# Distribution of Box Office --------------------------------------------------

sns.set(style="white", color_codes=True, font_scale=1.2)
sns.distplot(target)
sns.despine(offset=10, trim=True)
plt.xlabel('Domestic Box Office (USD)')
plt.ylabel('Number of observed movies')
plt.title("Distribution of Box Office", fontweight='bold')
plt.savefig('Plots_Final/plt_boxplot_suc_genre_comb_top30', bbox_inches='tight') 
plt.show()


# Log Transformed Distribution of Box Office
sns.set_style("white")
sns.distplot(np.log(target))
sns.despine(offset=10, trim=True)
plt.xlabel('Domestic Box Office (USD)')
plt.ylabel('Number of observed movies')
plt.title("Distribution of Box Office", fontweight='bold')


def mad_based_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def percentile_based_outlier(data, threshold=99):
    diff = (100 - threshold) / 2.0
    minval, maxval = np.percentile(data, [diff, 100 - diff])
    return (data < minval) | (data > maxval)

def plot(x):
    fig, axes = plt.subplots(nrows=2)
    for ax, func in zip(axes, [percentile_based_outlier, mad_based_outlier]):
        #sns.distplot(x, ax=ax, rug=True, hist=False)
        sns.distplot(x, ax = ax)
        outliers = x[func(x)]
        ax.plot(outliers, np.zeros_like(outliers), 'ro', clip_on=False)

    kwargs = dict(y=0.95, x=0.05, ha='left', va='top')
    axes[0].set_title('Percentile-based Outliers', **kwargs)
    axes[1].set_title('MAD-based Outliers', **kwargs)
    fig.suptitle('Comparing Outlier Tests with n={}'.format(len(x)), size=14)

plot(np.log(target))
plot(target)

# Plot MAD OUTLIERS
sns.set(style="white", color_codes=True, font_scale=1.2)
fig, ax = plt.subplots()
x = target

g = sns.distplot(x)
outliers = x[mad_based_outlier(x, 65)]
ax.plot(outliers, np.zeros_like(outliers), 'ro', clip_on=False)
        
sns.despine(offset=10, trim=True)
plt.xlabel('Box Office (USD)')
plt.ylabel('Number of observed movies')
plt.xticks(g.get_xticks(), ["0","100m","200m", "300m", "400m", "500m","600m",'700m','800m'])
plt.yticks(g.get_yticks(), ["0","2,000","4,000", "6,000", "8,000"])
plt.title("MAD-based Outliers Detection", fontweight='bold')
plt.savefig('Plots_Final/plt_box_ofice_outliers', bbox_inches='tight') 
plt.show()


def is_outlier(points, thresh=3.5):

    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def is_outlier_perc(data, threshold=99):
    diff = (100 - threshold) / 2.0
    minval, maxval = np.percentile(data, [diff, 100 - diff])
    return (data < minval) | (data > maxval)


movies_data_modelling['Outlier'] = list(is_outlier(movies_data_modelling.Domestic_Box_Office)) # outliers from np.log distribution

# Remove Outliers
movies_data_modelling = movies_data_modelling[movies_data_modelling.Outlier == False]

movies_data_modelling.drop('Outlier', axis = 1, inplace = True)



"""
    Skew - Kurtosis -------------------------------------------------------------------------------
"""

from scipy.stats import skew, kurtosis
# Target Variable
skew(target.Domestic_Box_Office)
kurtosis(target.Domestic_Box_Office)

skew(np.log(target.Domestic_Box_Office))
kurtosis(np.log(target.Domestic_Box_Office))

# Regressors
df_skew = pd.DataFrame({'Regressor':list(regressors.columns), 'Skew': -100, 'Kurtosis': -100})
df_skew.Skew = list(skew(regressors))
df_skew.Kurtosis = list(kurtosis(regressors))

# Transform Budget
regressors.Budget = np.log(regressors.Budget)


# Distribution of Budget
sns.set(style="white", color_codes=True, font_scale=1.2)
fig = sns.distplot(regressors.Budget, kde=True, norm_hist=False, hist = True)
sns.despine(offset=10, trim=True)
plt.xlabel('Production Budget (USD)')
plt.ylabel('Number of observed movies')
plt.xticks(fig.get_xticks(), ["0","250m","500m", "750m", "1b", "1.25b","1.5b",'1.75b'])
plt.yticks(fig.get_yticks(), ["0","2,000","4,000", "6,000", "8,000", "11,000"])
plt.title("Distribution of Production Budget", fontweight='bold')
plt.savefig('Plots_Final/plt_budget', bbox_inches='tight') 
plt.show()

# Distribution of log budget
sns.set(style="white", color_codes=True, font_scale=1.2)
fig = sns.distplot(np.log(regressors.Budget), kde=True, norm_hist=False, hist = True)
sns.despine(offset=10, trim=True)
plt.xlabel('Logarithm of Production Budget')
plt.ylabel('Distribution (%)')
#plt.xticks(fig.get_xticks(), ["0","250m","500m", "750m", "1b", "1.25b","1.5b",'1.75b'])
#plt.yticks(fig.get_yticks(), ["0","2,000","4,000", "6,000", "8,000", "11,000"])
plt.title("Distribution of Production Budget in Logarithm Form", fontweight='bold')
plt.savefig('Plots_Final/plt_budget_log', bbox_inches='tight') 
plt.show()





"""
    Randomly Split train (80%) and test (20%) set ---------------------------------------
"""

# Random split regressors and corresponding target
regressors_train, regressors_test, target_train, target_test = train_test_split(
        regressors, target, test_size=0.2, random_state = 1991)


"""
    Scale Regressors ----------------------------------------------------------------------
"""
# Min-max standardisation --------------------------------------------
# Initialise Scaler
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
# Train scaler on train data 
regressors_train = min_max_scaler.fit_transform(regressors_train)
# Transform validation and test set based on train data transformations
#regressors_validation = min_max_scaler.transform(regressors_validation)   
regressors_test = min_max_scaler.transform(regressors_test)




"""
    Explained Variance Measure (identify No of PCA components) ------------------------------
"""

# Eigenvectors and eigenvalues of Covariance matirx ---------------
mean_vec = np.mean(regressors_train, axis=0)
cov_mat = np.cov(regressors_train.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# List of (eigenvalue, eigenvector) tuples
eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort from high to low
eig_pairs.sort(key = lambda x: x[0], reverse= True)

# Calculation of Explained Variance from the eigenvalues
tot = sum(eig_vals)
var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance
cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance


# Plot --------------------------------------------------------------
sns.set(style="white", color_codes=True, font_scale=1.2)
fig, ax = plt.subplots()
#ax.set_facecolor('white')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.bar(range(len(regressors.columns)), var_exp, alpha=0.3333, align='center', label='Individual explained variance', color = 'darkred')
#plt.step(range(len(regressors.columns)), cum_var_exp, where='mid',label='Cumulative explained variance')
plt.plot(range(len(regressors.columns)), cum_var_exp,label='Cumulative explained variance')
plt.axhline(y=100, xmax =0.9, color='grey', linestyle='-.', linewidth = 1.2)
plt.axhline(y=98.5, xmax =0.52, color='grey', linestyle='-.', linewidth = 1.2)
plt.axvline(x=150, ymin=0, ymax = 0.94, color='grey', linestyle='-.', linewidth = 1.2)

plt.ylabel('Explained variance Ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.title('Explained Variance of Principal Components', fontweight='bold')
plt.savefig('Plots_Final/plt_modelling_variance', bbox_inches='tight')
plt.show()


pca_components = 150

"""
    Principal Component Analysis -------------------------------------------------------------
"""

def pca(train, test, n):
    pca = PCA(n)
    pca.fit(train)
    train_pca = pca.transform(train)
    #validation_pca = pca.transform(validation)
    test_pca = pca.transform(test)    
    return train_pca, test_pca
   
regressors_train_pca, regressors_test_pca = pca(regressors_train, regressors_test, pca_components)





"""
    Decision Tree Regression --------------------------------------------------
"""

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.grid_search import GridSearchCV

# Tune Hyperparameters of DecisionTreeClassifier
parameters = {'max_depth' : [2,3,4, 5,10], 
              'criterion': ["mse"],
              'min_samples_split' : [2,3,5],
              'min_samples_leaf' : [1, 5],
              'max_leaf_nodes' : [5, 7, 10, 12, 15]}
grid_search_tree = GridSearchCV(DecisionTreeRegressor(), parameters, n_jobs=4)

grid_search_tree.fit(regressors_train_pca, target_train)
print (grid_search_tree.best_score_, grid_search_tree.best_params_) 

# Train Best Model
regr_tree = DecisionTreeRegressor(max_depth=2, min_samples_leaf =1, criterion = 'mse', min_samples_split = 2, max_leaf_nodes = 10)
regr_tree.fit(regressors_train_pca, target_train)
predicted_tree = regr_tree.predict(regressors_test_pca)
# RMSE
math.sqrt(mean_squared_error(target_test, predicted_tree))


# Add AdaBoost Regression ------------------------------------------------
# Tune learning rate and n_estimators
parameters = {'n_estimators' : range(2,50,2), 
              'learning_rate' : [0.001,0.05,0.1,0.3,0.8,1,1.5,2,4,5]
              }
grid_search_boost_tree = GridSearchCV(AdaBoostRegressor(DecisionTreeRegressor(max_depth=2, min_samples_leaf =1, criterion = 'mse', min_samples_split = 2, max_leaf_nodes = 10)), parameters, n_jobs=4)
grid_search_boost_tree.fit(regressors_train_pca, target_train)
print (grid_search_boost_tree.best_score_, grid_search_boost_tree.best_params_) 

# Train Best Model 
regr_tree_boost = AdaBoostRegressor(DecisionTreeRegressor(max_depth=2, min_samples_leaf =1, criterion = 'mse', min_samples_split = 2, max_leaf_nodes = 10),
                          n_estimators=42, learning_rate = 0.05, random_state=1991)

regr_tree_boost.fit(regressors_train_pca, target_train)
# Predict
predicted_tree_boost = regr_tree_boost.predict(regressors_test_pca)
# RMSE
math.sqrt(mean_squared_error(target_test, predicted_tree_boost))




"""
    Gradient Boosting Regression ----------------------------------------------------

"""

# Tune Hyperparameters of DecisionTreeClassifier
parameters = {'max_depth' : [2,3,4,5,10], 
              'learning_rate': [0.001,0.05,0.1,0.3,0.8,1,1.5,2,4,5],                           
              'n_estimators' : range(2,50,5),
              'min_samples_leaf' : [1, 2, 3,5],
              'max_leaf_nodes' : [5, 7, 10, 15]}
grid_search_gradientboost = GridSearchCV(GradientBoostingRegressor(), parameters, n_jobs=4)

grid_search_gradientboost.fit(regressors_train_pca, target_train)
print (grid_search_gradientboost.best_score_, grid_search_gradientboost.best_params_) 


# Train Best Model 
regr_gradientboost = GradientBoostingRegressor(n_estimators = 85, max_depth = 5, min_samples_split = 2, max_leaf_nodes = 14, min_samples_leaf = 4, learning_rate = 0.15, loss = 'ls')
regr_gradientboost.fit(regressors_train_pca, target_train)
# Predict
predicted_gradientboost = regr_gradientboost.predict(regressors_test_pca).clip(min=0)
# RMSE
math.sqrt(mean_squared_error(target_test, predicted_gradientboost))






"""
    Random Forest Regressor --------------------------------------------------
"""


# Tune Hyperparameters of DecisionTreeClassifier
parameters = {'max_depth' : [15,20,50], 
              'criterion': ["mse"],
              'min_samples_split' : [2,5],
              'min_samples_leaf' : [20,40],
              'max_leaf_nodes' : [60,100,200]}
grid_search_rf = GridSearchCV(RandomForestRegressor(), parameters, n_jobs=4)

grid_search_rf.fit(regressors_train_pca, target_train)
print (grid_search_rf.best_score_, grid_search_rf.best_params_) 

# Train Best Model
regr_rf = RandomForestRegressor(max_depth=150, min_samples_leaf =30, criterion = 'mse', min_samples_split = 2, max_leaf_nodes = 150, random_state = 1991)
regr_rf.fit(regressors_train_pca, target_train)
predicted_rf = regr_rf.predict(regressors_test_pca)
# RMSE
math.sqrt(mean_squared_error(target_test, predicted_rf))





"""
    LASSO -----------------------------------------
"""
from sklearn.linear_model import LassoCV, Lasso, RidgeCV, Ridge

# Get different alphas to test
alphas = 10**np.linspace(10,-5,100)*0.5

lasso = Lasso(max_iter=10000, normalize=False)
coefs = []
for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(regressors_train_pca, target_train)
    coefs.append(lasso.coef_)   
ax = plt.gca()
ax.plot(alphas*2, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')


# Tune parameter alpha with Cross Validation
lassocv = LassoCV(alphas=None, cv=10, max_iter=100000, normalize=False, random_state = 1991, positive = True)
lassocv.fit(regressors_train_pca, target_train)

# Fit Lasso model with best alpha
lasso = Lasso(max_iter=10000, normalize=False, alpha = lassocv.alpha_, positive = True) # a = 17671.398612860448
lasso.fit(regressors_train_pca, target_train)
# Predict on test set
predicted_lasso = lasso.predict(regressors_test_pca)

# RMSE
math.sqrt(mean_squared_error(target_test, predicted_lasso))
# MAE
mean_absolute_error(predicted_lasso, target_test)
#MAPE
np.mean(np.abs((target_test - predicted_lasso) / target_test))


predicted_df_lasso = pd.DataFrame({'Predicted_Values':list(predicted_lasso.flatten().astype(int)), 'Actual_Values': list(target_test)}).set_index(target_test.index) # Set index of initial movies to search for them (from validation target set)

predicted_df_lasso['SE'] = (predicted_df_lasso.Predicted_Values - predicted_df_lasso.Actual_Values) **2
predicted_df_lasso['Error'] = predicted_df_lasso.Predicted_Values - predicted_df_lasso.Actual_Values
predicted_df_lasso['Absolute_Error'] = predicted_df_lasso.Predicted_Values - predicted_df_lasso.Actual_Values





"""

    RIDGE ------------------------------------------------------
    
"""                   
                        
# Tune parameter alpha with Cross Validation                     
ridgecv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', normalize=False)
ridgecv.fit(regressors_train_pca, target_train)
ridgecv.alpha_

# Fit Ridge model with best alpha
ridge = Ridge(alpha=ridgecv.alpha_, normalize=False, max_iter=10000)
ridge.fit(regressors_train_pca, target_train)
# Predict on test set
predicted_ridge = ridge.predict(regressors_test_pca).clip(min=0)


# RMSE
math.sqrt(mean_squared_error(target_test, predicted_ridge))
# MAE
mean_absolute_error(predicted_ridge, target_test)
#MAPE
np.mean(np.abs((target_test - predicted_ridge) / target_test))
                        




"""
    Baseline Linear Regression --------------------------------------------------------------

"""
import math
from sklearn import linear_model

# baseline Regression (no preprocessing)
regr_base = linear_model.LinearRegression()
regr_base.fit(regressors_train, target_train)       
predicted_linear = regr_base.predict(regressors_test)        
# RMSE
math.sqrt(mean_squared_error(predicted_linear, target_test))


# Fit LR model
regr = linear_model.LinearRegression()
regr.fit(regressors_train_pca, target_train)       
# Predict on test set
predicted_linear = regr.predict(regressors_test_pca)
            
# RMSE
rmse_baseline = math.sqrt(mean_squared_error(predicted_linear, target_test))
# MAE
mae_baseline = mean_absolute_error(predicted_linear, target_test)
#MAPE
mape_baseline = np.mean(np.abs((target_test - predicted_linear) / target_test))
      
#plot residuals
residuals = target_test - predicted_linear
plt.scatter(x = predicted_linear, y = residuals)




"""
    Baseline Linear Regression Logarithm --------------------------------------------------------------

"""

regr_log = linear_model.LinearRegression()           
regr_log.fit(regressors_train_pca, target_train_log)     
predicted_linear_log = regr_log.predict(regressors_test_pca)
 
# RMSE           
rmse_baseline_log = math.sqrt(mean_squared_error(predicted_linear_log, target_test_log))
#MAE
mae_baseline_log = mean_absolute_error(predicted_linear_log, target_test_log)
#MAPE
mape_baseline_log = np.mean(np.abs((target_test_log - predicted_linear_log) / target_test_log))
#plot residuals
residuals_log = target_test_log - predicted_linear_log
plt.scatter(x = predicted_linear_log, y = residuals_log)



# Convert log transformation scaling up by an estimation of Σ(exp(u_i)) / n   
predicted_linear_transf = np.exp(predicted_linear_log)  
# rmse
math.sqrt(mean_squared_error(predicted_linear_transf, target_test))
# mape
np.mean(np.abs((target_test - predicted_linear_transf) / target_test))


"""
    Baseline Linear Regression Root of 10 --------------------------------------------------------------

"""

regr_10 = linear_model.LinearRegression()           
regr_10.fit(regressors_train_pca, target_train_10)     

# Predict and Transform back to base 
predicted_10 = regr_10.predict(regressors_test_pca) ** 10

# RMSE           
math.sqrt(mean_squared_error(predicted_10, target_test))
# MAPE
np.mean(np.abs((target_test - predicted_10) / target_test))





"""
    SVR  --------------------------------------------------------------------------------------------
"""
from sklearn.svm import SVR

# RBF ----------------------------
svr_rbf = SVR(kernel='rbf', C=1e8, gamma=0.001)
predicted_rbf = svr_rbf.fit(regressors_train_pca, target_train).predict(regressors_test_pca)
# RMSE
math.sqrt(mean_squared_error(predicted_rbf, target_test))


# POLYNOMIAL KERNEL --------------
svr_poly = SVR(kernel='poly', C=1e3, degree=3)
predicted_poly = svr_poly.fit(regressors_train_pca, target_train).predict(regressors_test_pca)
math.sqrt(mean_squared_error(predicted_poly, target_test))






"""
    ANN  --------------------------------------------------------------------------------------------

"""

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from sklearn.cross_validation import train_test_split


# Model 2 -----------------------------------------------------------
model = Sequential()
# Input layer with dimension 1 and hidden layer i with 128 neurons. 
model.add(Dense(128, input_dim=150, init='normal', activation='relu'))
# Dropout of 20% of the neurons and activation layer.
model.add(Dropout(.2))
model.add(Activation("linear"))
# Hidden layer j with 64 neurons plus activation layer.
model.add(Dense(64, activation='relu'))
model.add(Activation("linear"))
# Hidden layer j with 64 neurons plus activation layer.
model.add(Dense(64, activation='relu'))
model.add(Activation("linear"))
# Hidden layer k with 64 neurons.
model.add(Dense(64, activation='relu'))
# Output Layer.
model.add(Dense(1))

# Model is derived and compiled using mean square error as loss
# function, accuracy as metric and gradient descent optimizer.
model.compile(loss='mse', optimizer='rmsprop')

# Training model with train data. Fixed random seed:
np.random.seed(1991)

# Fit model
history = model.fit(regressors_train_pca, target_train, validation_split=0.2, epochs=60, batch_size=16, verbose=2)



# Model 3 -----------------------------------------------------------
model = Sequential()
# Input layer with dimension 1 and hidden layer i with 128 neurons. 
model.add(Dense(128, input_dim=pca_components, kernel_initializer='he_normal', activation='relu'))
# Dropout of 20% of the neurons and activation layer.
model.add(Dropout(.3))
# Hidden layer j with 256 neurons plus activation layer.
model.add(Dense(256, activation='relu'))
# Hidden layer j with 128 neurons plus activation layer.
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
# Output Layer
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop')

# Train NN
history = model.fit(regressors_train_pca, target_train, validation_split=0.2, epochs=20, batch_size=16, verbose=2)

# Save model
#model.save('model_3.h5')  # creates a HDF5 file 'my_model.h5'
# Load model
#model3_validation = load_model('model_3.h5')
#model_3_test = load_model('model_3_test.h5')

# Predict
predicted_nn = model.predict(regressors_test_pca)
# Inverse predictions
#predicted = target_scaler.inverse_transform(predicted)

# Create df of predicted and actual values
predicted_df_nn = pd.DataFrame({'Predicted_Values':list(predicted_nn.flatten().astype(int)), 'Actual_Values': list(target_test)}).set_index(target_test.index) # Set index of initial movies to search for them (from validation target set)

# RMSE
math.sqrt(mean_squared_error(predicted_df_nn.Predicted_Values, predicted_df_nn.Actual_Values))

# MAE
mean_absolute_error(predicted_df_nn.Predicted_Values, predicted_df_nn.Actual_Values)
# MAPE
np.mean(np.abs((predicted_df_nn.Actual_Values - predicted_df_nn.Predicted_Values) / predicted_df_nn.Actual_Values))

# Residuals plot
predicted_df_nn["residuals"] = predicted_df_nn["Actual_Values"] - predicted_df_nn["Predicted_Values"]
predicted_df_nn.plot(x = "Predicted_Values", y = "residuals",kind = "scatter")





"""
    ###########################################################################
    ##################### Ensemble Stacking Regression ########################
    ###########################################################################
    
    Level 1: train and predict each model on training set with NFold CV
    Level 2: train and predict(test) based on the predictions of Level 1 (training set)
"""


def model_ANN(no_features):
    model = Sequential()
    # Input layer with dimension 1 and hidden layer i with 128 neurons. 
    model.add(Dense(128, input_dim=no_features, init='normal', activation='relu'))
    # Dropout of 20% of the neurons and activation layer.
    model.add(Dropout(.2))
    model.add(Activation("linear"))
    # Hidden layer j with 64 neurons plus activation layer.
    model.add(Dense(64, activation='relu'))
    model.add(Activation("linear"))
    # Hidden layer j with 64 neurons plus activation layer.
    model.add(Dense(64, activation='relu'))
    model.add(Activation("linear"))
    # Hidden layer k with 64 neurons.
    model.add(Dense(64, activation='relu'))
    # Output Layer.
    model.add(Dense(1))

    # Model is derived and compiled using mean square error as loss
    # function, accuracy as metric and gradient descent optimizer.
    model.compile(loss='mse', optimizer='rmsprop')
    return model

def model_ANN_2(no_features):
    model = Sequential()
    # Input layer with dimension 1 and hidden layer i with 128 neurons. 
    model.add(Dense(128, input_dim=no_features, kernel_initializer='he_normal', activation='relu'))
    # Dropout of 20% of the neurons and activation layer.
    model.add(Dropout(.3))
    # Hidden layer j with 64 neurons plus activation layer.
    model.add(Dense(256, activation='relu'))
    # Hidden layer j with 64 neurons plus activation layer.
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    # Hidden layer k with 64 neurons.
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    # Output Layer.
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='rmsprop')
    return model

from sklearn.cross_validation import KFold


def stacking_regression(folds, models):    
    # Level 1 regression models
    regrs = models

    # 5-fold cross validation
    kf = list(KFold(len(target_train), n_folds=folds, shuffle = True, random_state = 1991))

   
    # Pre-allocate the data
    blend_train = np.zeros((regressors_train_pca.shape[0], len(regrs)))     # Number of training data x Number of classifiers
    blend_test = np.zeros((regressors_test_pca.shape[0], len(regrs)))       # Number of testing data x Number of classifiers
                  
    
    # For each classifier, we train the number of fold times (=len(kf))
    for j, clf in enumerate(regrs):
        print('Training Regression Model [{}] - {}'.format(j, clf))
        blend_test_j = np.zeros((regressors_test_pca.shape[0], len(kf))) # Number of testing data x Number of folds , we will take the mean of the predictions later
        for i, (train_index, cv_index) in enumerate(kf):
            print('Fold [{}]'.format(i))
            
            # This is the training and validation set
            X_train = regressors_train_pca[train_index]
            Y_train = target_train.iloc[train_index]
            X_cv = regressors_train_pca[cv_index]
        
            if(j == 0):
                clf.fit(X_train, Y_train, epochs=20, batch_size=16, verbose = 0)
            else:  
                clf.fit(X_train, Y_train)
            
            # This output will be the basis for our blended classifier to train against,
            # which is also the output of level 1 Regressors
            blend_train[cv_index, j] = clf.predict(X_cv).clip(min=0).flatten()
            blend_test_j[:, i] = clf.predict(regressors_test_pca).clip(min=0).flatten() # make negative predictions 0
           
        # Take the mean of the predictions of the cross validation set
        blend_test[:, j] = blend_test_j.mean(1)
    
    
    # Blending (predict Level 2 based on predictions on the train set)

    bclf = model_ANN(len(models)) # Number of faetures on new training set at Level 2 = Number of Models at Level 1
    bclf.fit(blend_train, target_train, epochs=20, batch_size=16, verbose = 2)
    # Predict
    predicted_level2 = bclf.predict(blend_test)
    score = math.sqrt(mean_squared_error(target_test, predicted_level2))
    return score, predicted_level2


rmse_2, predicted_stacking_2 = stacking_regression(
        5, [model_ANN(150), 
#            Lasso(max_iter=10000, normalize=False, alpha = 17671.4, positive = True),
#            Ridge(alpha=23, normalize=False, max_iter=10000),
            SVR(kernel='rbf', C = 1000000000, gamma = 0.08)
#            RandomForestRegressor(max_depth=150, min_samples_leaf =30, criterion = 'mse', \
 #                         min_samples_split = 2, max_leaf_nodes = 150, random_state = 1991)
            ])



predicted_df = pd.DataFrame({'Predicted_Values':list(predicted_stacking_2.flatten().astype(int)), 'Actual_Values': list(target_test)}).set_index(target_test.index) # Set index of initial movies to search for them (from validation target set)

# RMSE's ON DIFFERENT RANGES
predicted_df['SE'] = (predicted_df.Predicted_Values - predicted_df.Actual_Values) **2
predicted_df['Absolute_Error'] = abs(predicted_df.Predicted_Values - predicted_df.Actual_Values)


# RMSE in Movies < 1m
math.sqrt(np.mean(predicted_df.loc[predicted_df.Actual_Values < 1000000,'SE']))
len(predicted_df.loc[(predicted_df.Actual_Values <= 1000000) & (predicted_df.Absolute_Error <1000000)])
len(predicted_df.loc[(predicted_df.Actual_Values <= 1000000) & (predicted_df.Absolute_Error >=1000000) & (predicted_df.Absolute_Error <10000000)])
len(predicted_df.loc[(predicted_df.Actual_Values <= 1000000) & (predicted_df.Absolute_Error >=10000000) & (predicted_df.Absolute_Error <50000000)])
len(predicted_df.loc[(predicted_df.Actual_Values <= 1000000) & (predicted_df.Absolute_Error >=50000000) & (predicted_df.Absolute_Error <100000000)])
len(predicted_df.loc[(predicted_df.Actual_Values <= 1000000) & (predicted_df.Absolute_Error >=100000000)])

# Range classification
len(set(list(predicted_df.loc[(predicted_df.Predicted_Values < 1000000)].index)).intersection(
        set(list(predicted_df.loc[(predicted_df.Actual_Values <= 1000000)].index))))
len(set(list(predicted_df.loc[(predicted_df.Predicted_Values > 1000000) & (predicted_df.Predicted_Values < 10000000)].index)).intersection(
        set(list(predicted_df.loc[(predicted_df.Actual_Values <= 1000000)].index))))
len(set(list(predicted_df.loc[(predicted_df.Predicted_Values > 10000000) & (predicted_df.Predicted_Values < 50000000)].index)).intersection(
        set(list(predicted_df.loc[(predicted_df.Actual_Values <= 1000000)].index))))
len(set(list(predicted_df.loc[(predicted_df.Predicted_Values > 50000000) & (predicted_df.Predicted_Values < 100000000)].index)).intersection(
        set(list(predicted_df.loc[(predicted_df.Actual_Values <= 1000000)].index))))
len(set(list(predicted_df.loc[(predicted_df.Predicted_Values > 100000000)].index)).intersection(
        set(list(predicted_df.loc[(predicted_df.Actual_Values <= 1000000)].index))))



# RMSE in Movies <10m
math.sqrt(np.mean(predicted_df.loc[((predicted_df.Actual_Values < 10000000) &  (predicted_df.Actual_Values > 1000000)),'SE']))
len(predicted_df.loc[(predicted_df.Actual_Values > 1000000) & (predicted_df.Actual_Values <= 10000000) & (predicted_df.Absolute_Error <1000000)])
len(predicted_df.loc[(predicted_df.Actual_Values > 1000000) & (predicted_df.Actual_Values <= 10000000) & (predicted_df.Absolute_Error >=1000000) & (predicted_df.Absolute_Error <10000000)])
len(predicted_df.loc[(predicted_df.Actual_Values > 1000000) & (predicted_df.Actual_Values <= 10000000) & (predicted_df.Absolute_Error >=10000000) & (predicted_df.Absolute_Error <50000000)])
len(predicted_df.loc[(predicted_df.Actual_Values > 1000000) & (predicted_df.Actual_Values <= 10000000) & (predicted_df.Absolute_Error >=50000000) & (predicted_df.Absolute_Error <100000000)])
len(predicted_df.loc[(predicted_df.Actual_Values > 1000000) & (predicted_df.Actual_Values <= 10000000) & (predicted_df.Absolute_Error >=100000000)])


# Number of movies predicted at each range with actual box office <10m
len(set(list(predicted_df.loc[(predicted_df.Predicted_Values < 1000000)].index)).intersection(
        set(list(predicted_df.loc[(predicted_df.Actual_Values > 1000000) & (predicted_df.Actual_Values <= 10000000)].index))))
len(set(list(predicted_df.loc[(predicted_df.Predicted_Values > 1000000) & (predicted_df.Predicted_Values < 10000000)].index)).intersection(
        set(list(predicted_df.loc[(predicted_df.Actual_Values > 1000000) & (predicted_df.Actual_Values <= 10000000)].index))))
len(set(list(predicted_df.loc[(predicted_df.Predicted_Values > 10000000) & (predicted_df.Predicted_Values < 50000000)].index)).intersection(
        set(list(predicted_df.loc[(predicted_df.Actual_Values > 1000000) & (predicted_df.Actual_Values <= 10000000)].index))))
len(set(list(predicted_df.loc[(predicted_df.Predicted_Values > 50000000) & (predicted_df.Predicted_Values < 100000000)].index)).intersection(
        set(list(predicted_df.loc[(predicted_df.Actual_Values > 1000000) & (predicted_df.Actual_Values <= 10000000)].index))))
len(set(list(predicted_df.loc[(predicted_df.Predicted_Values > 100000000)].index)).intersection(
        set(list(predicted_df.loc[(predicted_df.Actual_Values > 1000000) & (predicted_df.Actual_Values <= 10000000)].index))))


# RMSE in Movies >10m - <50m
math.sqrt(np.mean(predicted_df.loc[((predicted_df.Actual_Values < 50000000) &  (predicted_df.Actual_Values > 10000000)),'SE']))
len(predicted_df.loc[(predicted_df.Actual_Values > 10000000) & (predicted_df.Actual_Values < 50000000) & (predicted_df.Absolute_Error <1000000)])
len(predicted_df.loc[(predicted_df.Actual_Values > 10000000) & (predicted_df.Actual_Values < 50000000) & (predicted_df.Absolute_Error >=1000000) & (predicted_df.Absolute_Error <10000000)])
# >10m - < 50 m
len(predicted_df.loc[(predicted_df.Actual_Values > 10000000) & (predicted_df.Actual_Values < 50000000) & (predicted_df.Absolute_Error >=10000000) & (predicted_df.Absolute_Error <50000000)])
# 50 - 100m
len(predicted_df.loc[(predicted_df.Actual_Values > 10000000) & (predicted_df.Actual_Values < 50000000) & (predicted_df.Absolute_Error >=50000000) & (predicted_df.Absolute_Error <100000000)])
len(predicted_df.loc[(predicted_df.Actual_Values > 10000000) & (predicted_df.Actual_Values < 50000000) & (predicted_df.Absolute_Error >=100000000)])


# Number of movies predicted at each range with actual box office <50m
len(set(list(predicted_df.loc[(predicted_df.Predicted_Values < 1000000)].index)).intersection(
        set(list(predicted_df.loc[(predicted_df.Actual_Values > 10000000) & (predicted_df.Actual_Values < 50000000)].index))))
len(set(list(predicted_df.loc[(predicted_df.Predicted_Values > 1000000) & (predicted_df.Predicted_Values < 10000000)].index)).intersection(
        set(list(predicted_df.loc[(predicted_df.Actual_Values > 10000000) & (predicted_df.Actual_Values < 50000000)].index))))
len(set(list(predicted_df.loc[(predicted_df.Predicted_Values > 10000000) & (predicted_df.Predicted_Values < 50000000)].index)).intersection(
        set(list(predicted_df.loc[(predicted_df.Actual_Values > 10000000) & (predicted_df.Actual_Values < 50000000)].index))))
len(set(list(predicted_df.loc[(predicted_df.Predicted_Values > 50000000) & (predicted_df.Predicted_Values < 100000000)].index)).intersection(
        set(list(predicted_df.loc[(predicted_df.Actual_Values > 10000000) & (predicted_df.Actual_Values < 50000000)].index))))
len(set(list(predicted_df.loc[(predicted_df.Predicted_Values > 100000000)].index)).intersection(
        set(list(predicted_df.loc[(predicted_df.Actual_Values > 10000000) & (predicted_df.Actual_Values < 50000000)].index))))



# RMSE in Movies >50m - <100m
math.sqrt(np.mean(predicted_df.loc[((predicted_df.Actual_Values < 100000000) &  (predicted_df.Actual_Values > 50000000)),'SE']))
len(predicted_df.loc[(predicted_df.Actual_Values > 50000000) & (predicted_df.Actual_Values < 100000000) & (predicted_df.Absolute_Error <1000000)])
len(predicted_df.loc[(predicted_df.Actual_Values > 50000000) & (predicted_df.Actual_Values < 100000000) & (predicted_df.Absolute_Error >=1000000) & (predicted_df.Absolute_Error <10000000)])
# >10m - < 50 m
len(predicted_df.loc[(predicted_df.Actual_Values > 50000000) & (predicted_df.Actual_Values < 100000000) & (predicted_df.Absolute_Error >=10000000) & (predicted_df.Absolute_Error <50000000)])
# 50 - 100m
len(predicted_df.loc[(predicted_df.Actual_Values > 50000000) & (predicted_df.Actual_Values < 100000000) & (predicted_df.Absolute_Error >=50000000) & (predicted_df.Absolute_Error <100000000)])
len(predicted_df.loc[(predicted_df.Actual_Values > 50000000) & (predicted_df.Actual_Values < 100000000) & (predicted_df.Absolute_Error >=100000000)])



# Number of movies predicted at each range with actual box office <100m
len(set(list(predicted_df.loc[(predicted_df.Predicted_Values < 1000000)].index)).intersection(
        set(list(predicted_df.loc[(predicted_df.Actual_Values > 50000000) & (predicted_df.Actual_Values < 100000000)].index))))
len(set(list(predicted_df.loc[(predicted_df.Predicted_Values > 1000000) & (predicted_df.Predicted_Values < 10000000)].index)).intersection(
        set(list(predicted_df.loc[(predicted_df.Actual_Values > 50000000) & (predicted_df.Actual_Values < 100000000)].index))))
len(set(list(predicted_df.loc[(predicted_df.Predicted_Values > 10000000) & (predicted_df.Predicted_Values < 50000000)].index)).intersection(
        set(list(predicted_df.loc[(predicted_df.Actual_Values > 50000000) & (predicted_df.Actual_Values < 100000000)].index))))
len(set(list(predicted_df.loc[(predicted_df.Predicted_Values > 50000000) & (predicted_df.Predicted_Values < 100000000)].index)).intersection(
        set(list(predicted_df.loc[(predicted_df.Actual_Values > 50000000) & (predicted_df.Actual_Values < 100000000)].index))))
len(set(list(predicted_df.loc[(predicted_df.Predicted_Values > 100000000)].index)).intersection(
        set(list(predicted_df.loc[(predicted_df.Actual_Values > 50000000) & (predicted_df.Actual_Values < 100000000)].index))))


# RMSE in Movies >100m
math.sqrt(np.mean(predicted_df.loc[(predicted_df.Actual_Values > 100000000),'SE']))

len(predicted_df.loc[(predicted_df.Actual_Values > 100000000) & (predicted_df.Absolute_Error <1000000)])
len(predicted_df.loc[(predicted_df.Actual_Values > 100000000) & (predicted_df.Absolute_Error >=1000000) & (predicted_df.Absolute_Error <10000000)])
len(predicted_df.loc[(predicted_df.Actual_Values > 100000000) & (predicted_df.Absolute_Error >=10000000) & (predicted_df.Absolute_Error <50000000)])
len(predicted_df.loc[(predicted_df.Actual_Values > 100000000) & (predicted_df.Absolute_Error >=50000000) & (predicted_df.Absolute_Error <100000000)])
len(predicted_df.loc[(predicted_df.Actual_Values > 100000000) & (predicted_df.Absolute_Error >=100000000)])


# Number of movies predicted at each range with actual box office >100m
len(set(list(predicted_df.loc[(predicted_df.Predicted_Values < 1000000)].index)).intersection(
        set(list(predicted_df.loc[(predicted_df.Actual_Values > 100000000)].index))))
len(set(list(predicted_df.loc[(predicted_df.Predicted_Values > 1000000) & (predicted_df.Predicted_Values < 10000000)].index)).intersection(
        set(list(predicted_df.loc[(predicted_df.Actual_Values > 100000000)].index))))
len(set(list(predicted_df.loc[(predicted_df.Predicted_Values > 10000000) & (predicted_df.Predicted_Values < 50000000)].index)).intersection(
        set(list(predicted_df.loc[(predicted_df.Actual_Values > 100000000)].index))))
len(set(list(predicted_df.loc[(predicted_df.Predicted_Values > 50000000) & (predicted_df.Predicted_Values < 100000000)].index)).intersection(
        set(list(predicted_df.loc[(predicted_df.Actual_Values > 100000000)].index))))
len(set(list(predicted_df.loc[(predicted_df.Predicted_Values > 100000000)].index)).intersection(
        set(list(predicted_df.loc[(predicted_df.Actual_Values > 100000000)].index))))


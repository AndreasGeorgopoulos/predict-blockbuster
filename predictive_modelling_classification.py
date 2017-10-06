#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 12:04:03 2017

@author: Andreas Georgopoulos
"""
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt  
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import math



# Import dataframe (produced at preprocessing.py)
movies_data_modelling_class = pd.read_csv('final_data/movies_data_df_final_fillna_modelling.csv')

# Check if NA's
movies_data_modelling_class.isnull().sum().sum()


# Remove outliers (MAD-based)
movies_data_modelling_class = movies_data_modelling_class[(movies_data_modelling_class.Domestic_Box_Office < 320000000)].reset_index(drop = True)

# Create Bins of Ranges of Domestic Box Office
bins = [min(movies_data_modelling_class.Domestic_Box_Office)-1, 1000000, 10000000, 100000000, max(movies_data_modelling_class.Domestic_Box_Office)+1]
bins_names = list(range(len(bins)-1))
movies_data_modelling_class['Box_Office_Range_Bins'] = pd.cut(movies_data_modelling_class.Domestic_Box_Office, bins, labels=bins_names)

# Exclude target variable (Domestic Box Office) from regressors df
target = movies_data_modelling_class[['Box_Office_Range_Bins','Domestic_Box_Office']]
regressors = movies_data_modelling_class.drop(['Box_Office_Range_Bins','Domestic_Box_Office'], axis = 1)




"""
    VIF -------------------------------------------------------------------------------------
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




"""
    Randomly Split train (80%) and test (20%) set -----------------------
"""

# Random split regressors and corresponding target
regressors_train, regressors_test, target_train, target_test = train_test_split(
        regressors, target, test_size=0.2, random_state = 1991)


# Extract Target in RangeBins and Continuous Box Office
target_train_bin = target_train.Box_Office_Range_Bins
target_train_continuous = target_train.Domestic_Box_Office
target_validation_bin = target_test.Box_Office_Range_Bins
target_validation_continuous = target_test.Domestic_Box_Office




"""
    Oversampling SMOTE ----------------------------------------------------------------------------
"""
from imblearn.over_sampling import SMOTE

#regressors_train, regressors_validation_smote, target_train_bin, target_train_bin_smote = train_test_split(regressors_train, target_train_bin,
#                                                  test_size = .1,
#                                                  random_state=1991)
sm = SMOTE(random_state=1991, ratio = "auto")
regressors_train, target_train_bin = sm.fit_sample(regressors_train, target_train_bin)
regressors_train, target_train_bin = sm.fit_sample(regressors_train, target_train_bin) #resample second lowest class as well
regressors_train, target_train_bin = sm.fit_sample(regressors_train, target_train_bin)

plt.hist(target_train_bin)  
plt.title("Histogram")
plt.show()




"""
    Scale Regressors ----------------------------------------------------------------------
"""
# Min-max standardisation --------------------------------------------
# Initialise Scaler
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
# Train scaler on train data 
regressors_train = pd.DataFrame(min_max_scaler.fit_transform(regressors_train), index=regressors_train.index, columns=regressors_train.columns)
# Transform validation set based on train data transformations
regressors_validation = pd.DataFrame(min_max_scaler.transform(regressors_test), index=regressors_test.index, columns=regressors_test.columns)  

# For SMOTE --------------------
min_max_scaler = MinMaxScaler(feature_range=(0, 1))

regressors_train =min_max_scaler.fit_transform(regressors_train)
# Transform validation set based on train data transformations
regressors_validation = min_max_scaler.transform(regressors_test) 




"""
    Principal Component Analysis -----------------------------------------------------------
"""

def pca(train, test, n):
    pca = PCA(n)
    pca.fit(train)
    train_pca = pca.transform(train)
    test_pca = pca.transform(test)    
    return train_pca, test_pca
   
regressors_train_pca, regressors_validation_pca = pca(regressors_train, regressors_validation, 150)

pca = PCA(150)
pca.fit(regressors_train)
regressors_train_pca_df = pd.DataFrame(pca.transform(regressors_train), index=regressors_train.index)
regressors_validation_pca_df = pd.DataFrame(pca.transform(regressors_validation))



"""
    Random Forest -------------------------------------------------------------------
"""
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

model = RandomForestClassifier(random_state = 1991)
# Tune FR hyper parameters 'n_estimators' & 'min_samples_leaf'
param_grid = { 
    #'n_estimators': [1, 5, 10, 20, 50, 60, 70, 80, 90, 100, 150, 200],
    'n_estimators': [100, 200, 400],
    'min_samples_leaf': [1, 2, 5]
}
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = 5, verbose = 2) 
grid.fit(regressors_train_pca, target_train_bin) 
# Optimal Hyper-parameters based on GridSearch for RF
optimal_trees = grid.best_estimator_.n_estimators    
optimal_leaf = grid.best_estimator_.min_samples_leaf 
#clf = RandomForestClassifier(n_estimators = optimal_trees, min_samples_leaf = 2, random_state = 1991)


clf = RandomForestClassifier(n_estimators = 400, min_samples_leaf = 2, random_state = 1991)
clf.fit(regressors_train_pca, target_train_bin)           

target_validation_bin_predicted = clf.predict(regressors_validation_pca)


print(classification_report(target_validation_bin, target_validation_bin_predicted))
# Accuracy of Predictions
accuracy_score(target_validation_bin, target_validation_bin_predicted)

# Plot Confusion Matrix (check bin ranges that are missclassified with others)
def plot_confusion_matrix(y_true,y_pred):
    cm_array = confusion_matrix(y_true,y_pred)
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    plt.imshow(cm_array, interpolation='nearest', cmap=plt.cm.jet) # plt.cm.Blues jet
    plt.title("Confusion matrix", fontsize=16)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Number of Occurences', rotation=270, labelpad=30, fontsize=12)
    xtick_marks = np.arange(len(true_labels))
    ytick_marks = np.arange(len(pred_labels))
    plt.xticks(xtick_marks, true_labels)
    plt.yticks(ytick_marks,pred_labels)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 12
    #plt.savefig('confusionn_matrix')
    plt.rcParams["figure.figsize"] = fig_size
    plt.show()

print(confusion_matrix(target_test.Box_Office_Range_Bins, target_validation_bin_predicted))
plot_confusion_matrix(target_test.Box_Office_Range_Bins, target_validation_bin_predicted)

def classifaction_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[1]
        row['precision'] = float(row_data[2])
        row['recall'] = float(row_data[3])
        row['f1_score'] = float(row_data[4])
        row['support'] = float(row_data[5])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    return dataframe
    
print(classification_report(target_test.Box_Office_Range_Bins, target_validation_bin_predicted))
# Classification report
class_report_1_df = classifaction_report_csv(classification_report(target_test.Box_Office_Range_Bins, target_validation_bin_predicted))
class_report_1_df.to_csv('classification_report_model_1.csv', index = False)



# Predict Avg Box Office based on Bin ---------------------------------------------------------------

avg_boxoffice_bin = (target_train.groupby(['Box_Office_Range_Bins'], as_index=False).mean()).Domestic_Box_Office
# Predict Avg Box Office based on Bin prediction (avg value of movies in corresponding bin at training data)
target_validation_continuous_predicted = []

for i in range(len(target_validation_bin_predicted)):
    target_validation_continuous_predicted.append(avg_boxoffice_bin[int(target_validation_bin_predicted[i])])

# RMSE
math.sqrt(mean_squared_error(target_validation_continuous, target_validation_continuous_predicted))

# MAE
mean_absolute_error(target_validation_continuous_predicted, target_validation_continuous)
#MAPE
np.mean(np.abs((target_validation_continuous - target_validation_continuous_predicted) / target_validation_continuous))

# RMSE's ON DIFFERENT RANGES
predicted_df_continuous = pd.DataFrame({'Predicted_Values':target_validation_continuous_predicted, 'Actual_Values': target_validation_continuous})

predicted_df_continuous['SE'] = (predicted_df_continuous.Predicted_Values - predicted_df_continuous.Actual_Values) **2

# RMSE in Movies < 1m
math.sqrt(np.mean(predicted_df_continuous.loc[predicted_df_continuous.Actual_Values < 1000000,'SE']))

# RMSE in Movies <10m
math.sqrt(np.mean(predicted_df_continuous.loc[((predicted_df_continuous.Actual_Values < 10000000) &  (predicted_df_continuous.Actual_Values > 1000000)),'SE']))

# RMSE in Movies <100m
math.sqrt(np.mean(predicted_df_continuous.loc[((predicted_df_continuous.Actual_Values < 100000000) &  (predicted_df_continuous.Actual_Values > 10000000)),'SE']))

# RMSE in Movies >100m
math.sqrt(np.mean(predicted_df_continuous.loc[(predicted_df_continuous.Actual_Values > 100000000),'SE']))



# Predict Avg Box Office of ANN on corresponding bin ---------------------------------------------------------------
from sklearn.neighbors import LSHForest
from sklearn.neighbors import NearestNeighbors

import random

random.seed(1991)

def find_budget(train, train_y, test):
    
    # Get exact neighbors
    k = 5
    nbrs = NearestNeighbors(n_neighbors = k, algorithm = 'brute',
                        metric = 'cosine').fit(train)
    neighbors_exact = nbrs.kneighbors(test, return_distance=False)
    
    # Find the approximate 5 nearest neighbours of our test set based on LHSForest
    n_candidates = [10, 20, 50, 80, 100]
    n_estimators = [5, 10, 20, 30]
    accuracies = np.zeros((len(n_estimators), len(n_candidates)), dtype = float)
    for j in range(len(n_candidates)):
        for i in range(len(n_estimators)):
            lshf = LSHForest(n_estimators = n_estimators[i],
                 n_candidates = n_candidates[j], n_neighbors = k,
                 random_state = 1991)
            # Build the LSH Forest index
            lshf.fit(train)
            # Get neighbors
            neighbors_approx = lshf.kneighbors(test, return_distance=False)
            # Find accuracy
            array = np.equal(neighbors_approx,neighbors_exact)
            ac = np.mean(np.sum(array,axis=1)/k)
            accuracies[i, j] = ac
    max_ac = np.amax(accuracies)
    for i in range(accuracies.shape[0]):
        for j in range(accuracies.shape[1]):
            if accuracies[i, j] == max_ac:
                optimal_candidates = n_candidates[j]
                optimal_estimators = n_estimators[i]
    lshf = LSHForest(n_estimators = optimal_estimators,
                 n_candidates = optimal_candidates, n_neighbors = k,
                 random_state = 1991)
    # Build the LSH Forest index
    lshf.fit(train)
    # Get neighbors
    neighbors_approx = lshf.kneighbors(test, return_distance=False)
    # Return the average budget of the 5 nearest neighbors
    avg_budget = np.mean(train_y.loc[neighbors_approx.tolist()[0]])
    return avg_budget 


# i is corresponding predicted bin index of movie 
find_budget(regressors_train_pca_df.loc[target_train_bin[target_train_bin == target_validation_bin_predicted[i]].index.tolist()].as_matrix(), 
                                        target_train_continuous.loc[target_train_bin[target_train_bin == target_validation_bin_predicted[i]].index.tolist()].reset_index(drop = True),
                                                                    np.reshape(regressors_validation_pca_df.loc[i].as_matrix(), (1,-1)))


# Predict Avg Box Office based on Bin prediction (avg value of movies in corresponding bin at training data)
target_validation_continuous_predicted_LHS = []
for i in tqdm(range(len(target_validation_bin_predicted))):
    target_validation_continuous_predicted_LHS.append(find_budget(regressors_train_pca_df.loc[target_train_bin[target_train_bin == target_validation_bin_predicted[i]].index.tolist()].as_matrix(), 
                                        target_train_continuous.loc[target_train_bin[target_train_bin == target_validation_bin_predicted[i]].index.tolist()].reset_index(drop = True),
                                                                    np.reshape(regressors_validation_pca_df.loc[i].as_matrix(), (1,-1))))
    time.sleep(0.01)

# RMSE
math.sqrt(mean_squared_error(target_validation_continuous, target_validation_continuous_predicted_LHS))




"""
    SVM Classification ----------------------------------------------------------

"""
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 100, 1000]},
                    {'kernel': ['poly'], 'degree': [3,4] , 'coef0': [2,4,6], 'C': [1, 5,8,10,100]}
                     ]

""" Gaussian kernel grid search """

# Grid Search
tuned_parameters_rbf = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}]

clf_grid_rbf = GridSearchCV(SVC(), tuned_parameters_rbf, scoring = 'accuracy')
clf_grid_rbf.fit(regressors_train_pca, target_train_bin)

clf_grid_rbf.grid_scores_
clf_grid_rbf.best_params_


# Train Best Model
clf_svc_rbf = SVC(kernel = 'rbf', gamma = 0.1, C = 1000, class_weight = 'balanced') #0.001,1000
clf_svc_rbf.fit(regressors_train_pca, target_train_bin)           

target_validation_bin_predicted_rbf = clf_svc_rbf.predict(regressors_validation_pca)

# Accuracy of Predictions
accuracy_score(target_validation_bin, target_validation_bin_predicted_rbf)
# Confusion Matrix
print(confusion_matrix(target_test.Box_Office_Range_Bins, target_validation_bin_predicted_rbf))

# Classification Report
print(classification_report(target_test.Box_Office_Range_Bins, target_validation_bin_predicted_rbf))


# Find Box Office by LHS   (for SMOTE !!!!!)
# Inverse from smote
regressors_train_pca = regressors_train_pca[:8902]
target_train_bin = target_train_bin[:8902]

find_budget(regressors_train_pca[(np.where(target_train_bin == target_validation_bin_predicted_rbf[i])[0].tolist())], 
                                        target_train_continuous.iloc[np.where(target_train_bin == target_validation_bin_predicted_rbf[i])[0].tolist()].reset_index(drop = True),
                                                                    np.reshape(regressors_validation_pca_df.loc[i].as_matrix(), (1,-1)))

# Predict Avg Box Office based on Bin prediction (avg value of movies in corresponding bin at training data)
target_validation_continuous_predicted_rbf_LHS = []
for i in tqdm(range(len(target_validation_bin_predicted_rbf))):
    target_validation_continuous_predicted_rbf_LHS.append(find_budget(regressors_train_pca[(np.where(target_train_bin == target_validation_bin_predicted_rbf[i])[0].tolist())], 
                                        target_train_continuous.iloc[np.where(target_train_bin == target_validation_bin_predicted_rbf[i])[0].tolist()].reset_index(drop = True),
                                                                    np.reshape(regressors_validation_pca_df.loc[i].as_matrix(), (1,-1))))
    time.sleep(0.01)

# RMSE
math.sqrt(mean_squared_error(target_validation_continuous, target_validation_continuous_predicted_rbf_LHS)) # rmse 34658435

         
# Plot Tuning parameters ---------------------------         

table = pd.read_csv('tableofclassifiers.csv')

table.set_index(table['Unnamed: 0'], inplace = True, drop = True)
table.drop('Unnamed: 0', axis = 1, inplace = True)
#Create a colormap  
import seaborn as sns

sns.set(style="white", color_codes=True, font_scale=1.2)
f2=plt.figure(figsize=(12,8))
mask = np.zeros_like(table)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    sns.heatmap(table, mask=mask, vmax=.65, square=True, cmap="RdBu_r")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.xlabel('Gamma')
plt.ylabel('C')
plt.title("Heatmap of SVC parameters", fontweight='bold')
plt.savefig('plt_heatmap_svc.png', bbox_inches='tight')
plt.show() 
         
sns.heatmap(table, mask=mask, vmax=.65, square=True, cmap="RdBu_r")       




"""
    Gradient Boosting ---------------------------------------------------------

"""

from sklearn.ensemble import GradientBoostingClassifier

clf_fb = GradientBoostingClassifier(n_estimators = 200) #0.001,1000
clf_fb.fit(regressors_train_pca, target_train_bin)           

target_validation_bin_predicted_gb = clf_fb.predict(regressors_validation_pca)

# Accuracy of Predictions
accuracy_score(target_validation_bin, target_validation_bin_predicted_gb)
# Confusion Matrix
print(confusion_matrix(target_test.Box_Office_Range_Bins, target_validation_bin_predicted_gb))




"""

    Neural Network --------------------------------------------------------------
"""

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation


from sklearn.cross_validation import train_test_split
from keras.utils.np_utils import to_categorical


target_train_bin_cat = to_categorical(target_train_bin)

def NN_CLF_model(features):
	# create model
    model = Sequential()
    model.add(Dense(128, input_dim=features, kernel_initializer='he_normal', activation='relu'))
    # Dropout of 20% of the neurons and activation layer.
#    model.add(Dropout(.25))
#    model.add(Dense(128, activation='relu'))
#    model.add(Dropout(.3))
    # Hidden layer k with 64 neurons.
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.3))
    model.add(Dense(32, activation='relu'))
    # Output Layer.
    model.add(Dropout(.25))
    model.add(Dense(4, activation='softmax'))

	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model_clf_nn = NN_CLF_model(150)
history_csf_nn = model_clf_nn.fit(regressors_train_pca, target_train_bin_cat, validation_split=0.2, epochs=5, batch_size=16, verbose=2)

# Predict
predicted_clf_nn = model_clf_nn.predict_classes(regressors_validation_pca)

# Accuracy of Predictions
accuracy_score(target_validation_bin, predicted_clf_nn)
# Confusion Matrix
print(confusion_matrix(target_test.Box_Office_Range_Bins, predicted_clf_nn))

# Classification Report
print(classification_report(target_test.Box_Office_Range_Bins, predicted_clf_nn))







"""
    Ensemble Stacking Classifiers -----------------------------------------------
"""



from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold

def stacking_classifier(folds, models):    
    # Level 1 regression models
    regrs = models

    # 5-fold cross validation
    kf = list(KFold(len(target_train_bin), n_folds=folds, shuffle = True, random_state = 1991))

   
    # Pre-allocate the data
    blend_train = np.zeros((regressors_train_pca.shape[0], len(regrs)))     # Number of training data x Number of classifiers
    blend_test = np.zeros((regressors_validation_pca.shape[0], len(regrs)))       # Number of testing data x Number of classifiers
                  
    
    # For each classifier, we train the number of fold times (=len(kf))
    for j, clf in enumerate(regrs):
        print('Training Regression Model [{}] - {}'.format(j, clf))
        blend_test_j = np.zeros((regressors_validation_pca.shape[0], len(kf))) # Number of testing data x Number of folds , we will take the mean of the predictions later
        for i, (train_index, cv_index) in enumerate(kf):
            print('Fold [{}]'.format(i))
            
            # This is the training and validation set
            X_train = regressors_train_pca[train_index]
            #Y_train = target_train_bin.iloc[train_index]
            Y_train = target_train_bin[train_index]
            X_cv = regressors_train_pca[cv_index]
            
            if(j == 0):
                # ANN
                Y_train = to_categorical(Y_train)
                clf.fit(X_train, Y_train, validation_split=0.2, epochs=5, batch_size=16, verbose=2)
            else:  
                clf.fit(X_train, Y_train)

            
            # This output will be the basis for our blended classifier to train against,
            # which is also the output of level 1 Regressors
            if(j==0):
                blend_train[cv_index, j] = clf.predict_classes(X_cv).flatten()
                blend_test_j[:, i] = clf.predict_classes(regressors_validation_pca).flatten()
            else:
                blend_train[cv_index, j] = clf.predict(X_cv).flatten()
                blend_test_j[:, i] = clf.predict(regressors_validation_pca).flatten()
           
        # Take the mean of the predictions of the cross validation set
        blend_test[:, j] = blend_test_j.mean(1)
    
    
    # Blending (predict Level 2 based on predictions on the train set)
 
#    ridgecv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', normalize=False)
#    ridgecv.fit(blend_train, target_train)
#    ridgecv.alpha_
    # Fit Ridge model with best alpha

#    bclf = Ridge(alpha=ridgecv.alpha_, normalize=False, max_iter=10000)
#    bclf.fit(blend_train, target_train)

    bclf =  LogisticRegression()
    bclf.fit(blend_train, target_train_bin)
    #bclf =NN_CLF_model(len(regrs))
    #bclf.fit(blend_train, to_categorical(target_train_bin), validation_split=0.2, epochs=5, batch_size=16, verbose=2)
    # Predict now
    predicted_level2_bin = bclf.predict(blend_test)
    #predicted_level2_bin = bclf.predict_classes(blend_test)
    score = accuracy_score(target_validation_bin, predicted_level2_bin)
    return score, predicted_level2_bin


accuracy, predicted_stacking_bin = stacking_classifier(5, [NN_CLF_model(150),
        
        RandomForestClassifier(n_estimators = 400, min_samples_leaf = 2, random_state = 1991),
                                                       #ExtraTreesClassifier(n_estimators = 200, criterion = 'gini'),
                                                        GradientBoostingClassifier(n_estimators = 200),
                                                        SVC(kernel = 'rbf', gamma = 0.001, C = 1000)
                                                   ])


# Plot Confusion Matrix
plot_confusion_matrix(target_test.Box_Office_Range_Bins, predicted_stacking_bin)
print(confusion_matrix(target_test.Box_Office_Range_Bins, predicted_stacking_bin))
# Classification Report
print(classification_report(target_test.Box_Office_Range_Bins, predicted_stacking_bin))
# Classification report
class_report_1_df = classifaction_report_csv(classification_report(target_test.Box_Office_Range_Bins, predicted_stacking_bin))
class_report_1_df.to_csv('classification_report_model_1.csv', index = False)


# Predict Avg Box Office based on Bin ---------------------------------------------------------------
avg_boxoffice_bin_stack = (target_train.groupby(['Box_Office_Range_Bins'], as_index=False).mean()).Domestic_Box_Office
# Predict Avg Box Office based on Bin prediction (avg value of movies in corresponding bin at training data)
target_validation_continuous_predicted_stack = []
for i in range(len(predicted_stacking_bin)):
    target_validation_continuous_predicted_stack.append(avg_boxoffice_bin_stack[int(predicted_stacking_bin[i])])

# RMSE
math.sqrt(mean_squared_error(target_validation_continuous, target_validation_continuous_predicted_stack))




# Predict Box Office based on predicted Bins and LHS ----------------------------------
target_validation_continuous_predicted_LHS_stack = []
for i in tqdm(range(len(predicted_stacking_bin))):
    target_validation_continuous_predicted_LHS_stack.append(find_budget(regressors_train_pca_df.loc[target_train_bin[target_train_bin == predicted_stacking_bin[i]].index.tolist()].as_matrix(), 
                                        target_train_continuous.loc[target_train_bin[target_train_bin == predicted_stacking_bin[i]].index.tolist()].reset_index(drop = True),
                                                                    np.reshape(regressors_validation_pca_df.loc[i].as_matrix(), (1,-1))))
    time.sleep(0.01)

# RMSE
math.sqrt(mean_squared_error(target_validation_continuous, target_validation_continuous_predicted_LHS_stack))




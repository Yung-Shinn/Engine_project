#%% Import Package
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from sklearn.model_selection import  RandomizedSearchCV
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout,BatchNormalization, LSTM ,GRU ,SimpleRNN
from tensorflow.keras.optimizers import Adam,SGD,RMSprop,Adagrad,Adadelta,Adamax,Nadam
from tensorflow.python.keras import backend as K
from tensorflow import keras 
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,InputLayer,Reshape
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')
random.seed(0)
#function of MAAPE(Mean Arctangent Absolute Percentage Error)
def MAAPE(y_true,y_pred):
    return np.mean(np.arctan(np.abs((y_true-y_pred)/y_true)))

#%% Load data
train = pd.read_csv('Train.csv').astype('float32')
test = pd.read_csv('Test.csv').astype('float32')

x_train = train.drop(columns=['Unnamed: 0', 'RUL'])
y_train = train.RUL.reset_index(drop=True)


x_test = test.drop(columns=['Unnamed: 0', 'RUL'])
y_test = test.RUL.reset_index(drop=True)
print(x_train.shape, x_test.shape)
#%%標準化
x_scaler = StandardScaler().fit(x_train)
x_train = x_scaler.transform(x_train)
x_test = x_scaler.transform(x_test)

#%% PCA
pca = PCA().fit(x_train)
##作圖
plt.rcParams["figure.figsize"] = (12,6)
fig, ax = plt.subplots()
xi = np.arange(1, x_train.shape[1]+1, step=1)
y_for_pca = np.cumsum(pca.explained_variance_ratio_)
print(y_for_pca)
def find_best_Number_of_Components():
    for i in xi:
        if y_for_pca[i-1] > 0.9:
            best_n = i
            return best_n
plt.ylim(0.0,1.1)
plt.plot(xi, y_for_pca[:], marker='o', linestyle='--', color='b')
plt.xlabel('Number of Components')
plt.xticks(np.arange(0, x_train.shape[1]+1, step=1)) 
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')
plt.axhline(y=0.9, color='r', linestyle='-')
plt.text(0.5, 1.05, '90% cut-off threshold', color = 'red', fontsize=16)
ax.grid(axis='x')
plt.show()

#設定可解釋程度>90%所需的主成分數目
best_comp = find_best_Number_of_Components()
print(best_comp)

## PCA select
pca = PCA(n_components = best_comp).fit(x_train)
pca_x_train, pca_x_test = pca.transform(x_train),pca.transform(x_test)

#%% CART參數
tree_param_grid = {         
                'max_depth': [3,5,7,9],
                'max_features': ['auto', 0.9, 0.8, 0.7, 0.6, 0.5],
                'min_samples_split': [4,8,16,32], 
                'min_samples_leaf': [4,8,16,32]
                }
#%% Original CART Tuning
DT_grid_ori = GridSearchCV(DecisionTreeRegressor(random_state=3),param_grid=tree_param_grid, scoring='neg_mean_absolute_error', cv=5)
DT_grid_ori.fit(x_train, y_train)
DT_grid_ori.cv_results_, DT_grid_ori.best_params_, DT_grid_ori.best_score_

DT_ori = DT_grid_ori.best_estimator_  
DT_ori.fit(x_train, y_train)          
print(DT_grid_ori.best_params_)       

#%% Original CART Performance
DT_ori_train = DT_ori.predict(x_train)
DT_ori_test = DT_ori.predict(x_test)
DT_ori_pred = pd.concat([pd.DataFrame(DT_ori_train), pd.DataFrame(DT_ori_test)])

DT_RMSEr = np.sqrt(MSE(y_train, DT_ori_train))
print('training RMSE:', DT_RMSEr)
DT_MAEr = MAE(y_train, DT_ori_train)
print('training MAE:', DT_MAEr)
DT_MAAPEr = MAAPE(y_train, DT_ori_train)
print('training MAAPE:', DT_MAAPEr)

DT_RMSEs = np.sqrt(MSE(y_test, DT_ori_test))
print('testing RMSE:', DT_RMSEs)
DT_MAEs = MAE(y_test, DT_ori_test)
print('testing MAE:', DT_MAEs)
DT_MAAPEs = MAAPE(y_test, DT_ori_test)
print('testing MAAPE:', DT_MAAPEs)

perf_ori_DT = pd.DataFrame()
perf_ori_DT["DT_train"] = [DT_RMSEr, DT_MAEr, DT_MAAPEr]
perf_ori_DT["DT_test"] = [DT_RMSEs, DT_MAEs, DT_MAAPEs]
perf_ori_DT.index = ["RMSE", "MAE", "MAAPE"]

#%% PCA CART Tuning
DT_grid_pca = GridSearchCV(DecisionTreeRegressor(random_state=3),param_grid=tree_param_grid, scoring='neg_mean_absolute_error', cv=5)
DT_grid_pca.fit(pca_x_train, y_train)
DT_grid_pca.cv_results_, DT_grid_pca.best_params_, DT_grid_pca.best_score_

DT_pca = DT_grid_pca.best_estimator_
DT_pca.fit(pca_x_train, y_train)
print(DT_grid_pca.best_params_)

#%% PCA CART Performance
DT_pca_train = DT_pca.predict(pca_x_train)
DT_pca_test = DT_pca.predict(pca_x_test)
DT_pred_pca = pd.concat([pd.DataFrame(DT_pca_train), pd.DataFrame(DT_pca_test)])

pca_DT_RMSEr = np.sqrt(MSE(y_train, DT_pca_train))
print('training RMSE:', pca_DT_RMSEr)
pca_DT_MAEr = MAE(y_train, DT_pca_train)
print('training MAE:', pca_DT_MAEr)
pca_DT_MAAPEr = MAAPE(y_train, DT_pca_train)
print('training MAAPE:', pca_DT_MAAPEr)

pca_DT_RMSEs = np.sqrt(MSE(y_test, DT_pca_test))
print('testing RMSE:', pca_DT_RMSEs)
pca_DT_MAEs = MAE(y_test, DT_pca_test)
print('testing MAE:', pca_DT_MAEs)
pca_DT_MAAPEs = MAAPE(y_test, DT_pca_test)
print('testing MAAPE:', pca_DT_MAAPEs)

perf_pca_DT = pd.DataFrame()
perf_pca_DT["DT_train"] = [pca_DT_RMSEr, pca_DT_MAEr, pca_DT_MAAPEr]
perf_pca_DT["DT_test"] = [pca_DT_RMSEs, pca_DT_MAEs, pca_DT_MAAPEs]
perf_pca_DT.index = ["RMSE", "MAE", "MAAPE"]


#%% RF參數
RF_params = {
        'n_estimators': [4,8,12,16,32],
        'max_depth': [3,5,7,9],
        'max_features': ['auto', 0.9, 0.8, 0.7, 0.6, 0.5],
        'min_samples_split': [4,8,16,32], 
        'min_samples_leaf': [4,8,16,32] 
        }
#%% Original RF Tuning
rf_RS_ori = RandomizedSearchCV(RandomForestRegressor(random_state=516), param_distributions=RF_params, scoring='neg_mean_absolute_error', cv=3)
rf_RS_ori.fit(x_train, y_train)
rf_RS_ori.cv_results_, rf_RS_ori.best_params_, rf_RS_ori.best_score_

RF_ori = rf_RS_ori.best_estimator_
RF_ori.fit(x_train, y_train)
print(rf_RS_ori.best_params_)

#%% Original RF Performance
RF_ori_train = RF_ori.predict(x_train)
RF_ori_test = RF_ori.predict(x_test)
RF_ori_pred = pd.concat([pd.DataFrame(RF_ori_train), pd.DataFrame(RF_ori_test)])

RF_RMSEr = np.sqrt(MSE(y_train, RF_ori_train))
print('training RMSE:', RF_RMSEr)
RF_MAEr = MAE(y_train, RF_ori_train)
print('training MAE:', RF_MAEr)
RF_MAAPEr = MAAPE(y_train, RF_ori_train)
print('training MAAPE:', RF_MAAPEr)

RF_RMSEs = np.sqrt(MSE(y_test, RF_ori_test))
print('testing RMSE:', RF_RMSEs)
RF_MAEs = MAE(y_test, RF_ori_test)
print('testing MAE:', RF_MAEs)
RF_MAAPEs = MAAPE(y_test, RF_ori_test)
print('testing MAAPE:', RF_MAAPEs)

perf_ori_RF = pd.DataFrame()
perf_ori_RF["RF_train"] = [RF_RMSEr, RF_MAEr, RF_MAAPEr]
perf_ori_RF["RF_test"] = [RF_RMSEs, RF_MAEs, RF_MAAPEs]
perf_ori_RF.index = ["RMSE", "MAE", "MAAPE"]

#%% PCA RF Tuning
rf_RS_pca = RandomizedSearchCV(RandomForestRegressor(random_state=516), param_distributions=RF_params, scoring='neg_mean_absolute_error', cv=3)
rf_RS_pca.fit(pca_x_train, y_train)
rf_RS_pca.cv_results_, rf_RS_pca.best_params_, rf_RS_pca.best_score_

RF_pca = rf_RS_pca.best_estimator_
RF_pca.fit(pca_x_train, y_train)
print(rf_RS_pca.best_params_)

#%% PCA RF Performance
RF_pca_train = RF_pca.predict(pca_x_train)
RF_pca_test = RF_pca.predict(pca_x_test)
RF_pred_pca = pd.concat([pd.DataFrame(RF_pca_train), pd.DataFrame(RF_pca_test)])

pca_RF_RMSEr = np.sqrt(MSE(y_train, RF_pca_train))
print('training RMSE:', pca_RF_RMSEr)
pca_RF_MAEr = MAE(y_train, RF_pca_train)
print('training MAE:', pca_RF_MAEr)
pca_RF_MAAPEr = MAAPE(y_train, RF_pca_train)
print('training MAAPE:', pca_RF_MAAPEr)

pca_RF_RMSEs = np.sqrt(MSE(y_test, RF_pca_test))
print('testing RMSE:', pca_RF_RMSEs)
pca_RF_MAEs = MAE(y_test, RF_pca_test)
print('testing MAE:', pca_RF_MAEs)
pca_RF_MAAPEs = MAAPE(y_test, RF_pca_test)
print('testing MAAPE:', pca_RF_MAAPEs)

perf_pca_RF = pd.DataFrame()
perf_pca_RF["RF_train"] = [pca_RF_RMSEr, pca_RF_MAEr, pca_RF_MAAPEr]
perf_pca_RF["RF_test"] = [pca_RF_RMSEs, pca_RF_MAEs, pca_RF_MAAPEs]
perf_pca_RF.index = ["RMSE", "MAE", "MAAPE"]


#%%XGB參數
XGB_params = {'n_estimators':[10,50,100],
              'learning_rate':[0.005,0.08,0.01,0.02],
              'max_depth': range(3,10,1),
              'min_child_weight':range(1,6,1),
              'subsample': [0.6,0.7,0.8,0.9],
              'colsample_bytree': [0.6,0.7,0.8,0.9]
              }
#%%Ori XGB 調參
xgb_RS_ori = RandomizedSearchCV(XGBRegressor(random_state=516),param_distributions=XGB_params,scoring='neg_mean_absolute_error', cv=3)
xgb_RS_ori.fit(x_train, y_train)
xgb_RS_ori.cv_results_, xgb_RS_ori.best_params_, xgb_RS_ori.best_score_

XGB_ori = xgb_RS_ori.best_estimator_
XGB_ori.fit(x_train, y_train)
print(xgb_RS_ori.best_params_)
#%% Ori XGB Performance
XGB_ori_train = XGB_ori.predict(x_train)
XGB_ori_test = XGB_ori.predict(x_test)
XGB_ori_pred = pd.concat([pd.DataFrame(XGB_ori_train), pd.DataFrame(XGB_ori_test)])

XGB_RMSEr = np.sqrt(MSE(y_train, XGB_ori_train))
print('training RMSE:', XGB_RMSEr)
XGB_MAEr = MAE(y_train, XGB_ori_train)
print('training MAE:', XGB_MAEr)
XGB_MAAPEr = MAAPE(y_train, XGB_ori_train)
print('training MAAPE:', XGB_MAAPEr)

XGB_RMSEs = np.sqrt(MSE(y_test, XGB_ori_test))
print('testing RMSE:', XGB_RMSEs)
XGB_MAEs = MAE(y_test, XGB_ori_test)
print('testing MAE:', XGB_MAEs)
XGB_MAAPEs = MAAPE(y_test, XGB_ori_test)
print('testing MAAPE:', XGB_MAAPEs)

perf_ori_XGB = pd.DataFrame()
perf_ori_XGB["XGB_train"] = [XGB_RMSEr, XGB_MAEr, XGB_MAAPEr]
perf_ori_XGB["XGB_test"] = [XGB_RMSEs, XGB_MAEs, XGB_MAAPEs]
perf_ori_XGB.index = ["RMSE", "MAE", "MAAPE"]
#%%PCA XGB 調參
xgb_RS_pca = RandomizedSearchCV(XGBRegressor(random_state=516),param_distributions=XGB_params,scoring='neg_mean_absolute_error', cv=3)
xgb_RS_pca.fit(pca_x_train, y_train)
xgb_RS_pca.cv_results_, xgb_RS_pca.best_params_, xgb_RS_pca.best_score_

XGB_pca = xgb_RS_pca.best_estimator_
XGB_pca.fit(pca_x_train, y_train)
print(xgb_RS_pca.best_params_)
#%% PCA XGB Performance
XGB_pca_train = XGB_pca.predict(pca_x_train)
XGB_pca_test = XGB_pca.predict(pca_x_test)
XGB_pred_pca = pd.concat([pd.DataFrame(XGB_pca_train), pd.DataFrame(XGB_pca_test)])

pca_XGB_RMSEr = np.sqrt(MSE(y_train, XGB_pca_train))
print('training RMSE:', pca_XGB_RMSEr)
pca_XGB_MAEr = MAE(y_train, XGB_pca_train)
print('training MAE:', pca_XGB_MAEr)
pca_XGB_MAAPEr = MAAPE(y_train, XGB_pca_train)
print('training MAAPE:', pca_XGB_MAAPEr)

pca_XGB_RMSEs = np.sqrt(MSE(y_test, XGB_pca_test))
print('testing RMSE:', pca_XGB_RMSEs)
pca_XGB_MAEs = MAE(y_test, XGB_pca_test)
print('testing MAE:', pca_XGB_MAEs)
pca_XGB_MAAPEs = MAAPE(y_test, XGB_pca_test)
print('testing MAAPE:', pca_XGB_MAAPEs)

perf_pca_XGB = pd.DataFrame()
perf_pca_XGB["XGB_train"] = [pca_XGB_RMSEr, pca_XGB_MAEr, pca_XGB_MAAPEr]
perf_pca_XGB["XGB_test"] = [pca_XGB_RMSEs, pca_XGB_MAEs, pca_XGB_MAAPEs]
perf_pca_XGB.index = ["RMSE", "MAE", "MAAPE"]



#%%SVR參數

SVR_params = {'kernel':['rbf','sigmoid','linear','poly'],
              'C':[50,100], 
              'gamma':[pow(5,-3), pow(5,-2),pow(5,-1),1,5], 
              'epsilon':[0.05,0.1,0.15],    
              'degree':[2]}

#%%Ori SVR調參
SVR_RS_ori = RandomizedSearchCV(SVR(), param_distributions= SVR_params, scoring='neg_mean_absolute_error', n_jobs=-1, n_iter=15, cv=3)
SVR_RS_ori.fit(x_train, y_train)
SVR_RS_ori.cv_results_, SVR_RS_ori.best_params_, SVR_RS_ori.best_score_

SVR_ori = SVR_RS_ori.best_estimator_
SVR_ori.fit(x_train, y_train)
print(SVR_RS_ori.best_params_)

#%%Ori SVR Performance

SVR_ori_train = SVR_ori.predict(x_train)
SVR_ori_test = SVR_ori.predict(x_test)
SVR_ori_pred = pd.concat([pd.DataFrame(SVR_ori_train), pd.DataFrame(SVR_ori_test)])

SVR_RMSEr = np.sqrt(MSE(y_train, SVR_ori_train))
print('training RMSE:', SVR_RMSEr)
SVR_MAEr = MAE(y_train, SVR_ori_train)
print('training MAE:', SVR_MAEr)
SVR_MAAPEr = MAAPE(y_train, SVR_ori_train)
print('training MAAPE:', SVR_MAAPEr)

SVR_RMSEs = np.sqrt(MSE(y_test, SVR_ori_test))
print('testing RMSE:', SVR_RMSEs)
SVR_MAEs = MAE(y_test, SVR_ori_test)
print('testing MAE:',SVR_MAEs)
SVR_MAAPEs = MAAPE(y_test, SVR_ori_test)
print('testing MAAPE:', SVR_MAAPEs)

perf_ori_SVR = pd.DataFrame()
perf_ori_SVR["SVR_train"] = [SVR_RMSEr, SVR_MAEr, SVR_MAAPEr]
perf_ori_SVR["SVR_test"] = [SVR_RMSEs, SVR_MAEs, SVR_MAAPEs]
perf_ori_SVR.index = ["RMSE", "MAE", "MAAPE"]

#%%PCA SVR 調參
SVR_RS_pca = RandomizedSearchCV(SVR(), param_distributions= SVR_params, scoring='neg_mean_absolute_error', n_iter=15, cv=3)
SVR_RS_pca.fit(pca_x_train, y_train)
SVR_RS_pca.cv_results_, SVR_RS_pca.best_params_, SVR_RS_pca.best_score_

SVR_pca = SVR_RS_pca.best_estimator_
SVR_pca.fit(pca_x_train, y_train)
print(SVR_RS_pca.best_params_)

#%% PCA SVR Performance
SVR_pca_train = SVR_pca.predict(pca_x_train)
SVR_pca_test = SVR_pca.predict(pca_x_test)
SVR_pred_pca = pd.concat([pd.DataFrame(SVR_pca_train), pd.DataFrame(SVR_pca_test)])

pca_SVR_RMSEr = np.sqrt(MSE(y_train, SVR_pca_train))
print('training RMSE:', pca_SVR_RMSEr)
pca_SVR_MAEr = MAE(y_train, SVR_pca_train)
print('training MAE:', pca_SVR_MAEr)
pca_SVR_MAAPEr = MAAPE(y_train, SVR_pca_train)
print('training MAAPE:', pca_SVR_MAAPEr)

pca_SVR_RMSEs = np.sqrt(MSE(y_test, SVR_pca_test))
print('testing RMSE:', pca_SVR_RMSEs)
pca_SVR_MAEs = MAE(y_test, SVR_pca_test)
print('testing MAE:', pca_SVR_MAEs)
pca_SVR_MAAPEs = MAAPE(y_test, SVR_pca_test)
print('testing MAAPE:', pca_SVR_MAAPEs)

perf_pca_SVR = pd.DataFrame()
perf_pca_SVR["SVR_train"] = [pca_SVR_RMSEr, pca_SVR_MAEr, pca_SVR_MAAPEr]
perf_pca_SVR["SVR_test"] = [pca_SVR_RMSEs, pca_SVR_MAEs, pca_SVR_MAAPEs]
perf_pca_SVR.index = ["RMSE", "MAE", "MAAPE"]


#%% 深度學習參數   
from skopt.space import Categorical
DL_params ={                    
        'n_hidden':[1,4],
        'n_neurons': [6,128],   
        'activation':['relu', 'selu','tanh','softplus'],
        'select_optimizer':Categorical([optimizers.Adam, optimizers.RMSprop]), 
        'learning_rate':[0.0005, 0.025], 
        'n_batch_size':[8, 512],   
        'n_epochs':[100,200],  
        'n_dropout':[0.1,0.2],
        "kernel_initializer": ['glorot_uniform', 'he_normal', 'random_normal'] 
            }
#%%DNN Bulid the model
es = EarlyStopping(monitor='val_loss', mode='min', patience = 20, verbose=1)   
def build_model(n_dropout, n_hidden, n_neurons,
                learning_rate, activation, kernel_initializer,
                n_epochs, n_batch_size, select_optimizer):
    model = Sequential()   
    model.add(Dense(n_neurons, input_dim= len(x_train[0]), activation=activation, kernel_initializer=kernel_initializer))  
    for layer in range(n_hidden): 
        model.add(Dense(n_neurons, activation=activation, kernel_initializer=kernel_initializer))
        model.add(Dropout(n_dropout))
        
    model.add(Dense(1, activation='linear'))    
    
    optimizer = select_optimizer(lr=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)  
    
    model.fit(x_train, y_train, epochs=n_epochs, batch_size=n_batch_size, validation_split=0.2, callbacks=[es])
    return model

DL_keras_ori = KerasRegressor(build_model)
#%%Ori DNN 貝氏
BS_DL_ori= BayesSearchCV(DL_keras_ori, DL_params, n_iter=5, cv=5, random_state=0)
BS_DL_ori.fit(x_train,y_train)
print(BS_DL_ori.best_params_)
# Best model
DNN_ori=BS_DL_ori.best_estimator_.model
DNN_ori.summary()
#%% DNN-Performance
DNN_train= DNN_ori.predict(x_train).flatten()
DNN_test= DNN_ori.predict(x_test).flatten()
DNN_pred=pd.concat([pd.DataFrame(DNN_train),pd.DataFrame(DNN_test)])

#training
DNN_RMSEr =np.sqrt(MSE(y_train,DNN_train))
print('training RMSE:',DNN_RMSEr)

DNN_MAEr =MAE(y_train,DNN_train)
print('training MAE:',DNN_MAEr)

DNN_MAAPEr =MAAPE(y_train,DNN_train)
print('training MAAPE:',DNN_MAAPEr)

##testing
DNN_RMSEs =np.sqrt(MSE(y_test,DNN_test))
print('testing RMSE:',DNN_RMSEs)

DNN_MAEs =MAE(y_test,DNN_test)
print('testing MAE:',DNN_MAEs)

DNN_MAAPEs =MAAPE(y_test,DNN_test)
print('testing MAAPE:',DNN_MAAPEs)

perf_ori_DNN = pd.DataFrame()
perf_ori_DNN["DNN_train"] =[DNN_RMSEr,DNN_MAEr,DNN_MAAPEr]
perf_ori_DNN["DNN_test"] = [DNN_RMSEs,DNN_MAEs,DNN_MAAPEs]
perf_ori_DNN.index=["RMSE","MAE","MAPE"]
perf_ori_DNN
#%% PCA- DNN model
def build_pca_model(n_dropout, n_hidden, n_neurons,
                learning_rate, activation, kernel_initializer,
                n_epochs, n_batch_size, select_optimizer):
    model = Sequential()
    model.add(Dense(n_neurons, input_dim=len(pca_x_train[0]), activation=activation, kernel_initializer=kernel_initializer))
    for layer in range(n_hidden):
        model.add(Dense(n_neurons, activation=activation, kernel_initializer=kernel_initializer))
        model.add(Dropout(n_dropout))
        
    model.add(Dense(1, activation='linear'))
    
    optimizer = select_optimizer(lr=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    model.fit(pca_x_train, y_train, epochs=n_epochs, batch_size=n_batch_size, validation_split=0.2, callbacks=[es])
    return model

DL_keras_pca = KerasRegressor(build_pca_model)
#%%PCA DNN貝氏 
BS_DL_pca= BayesSearchCV(DL_keras_pca, DL_params, n_iter=5, cv=5, random_state=0)
BS_DL_pca.fit(pca_x_train,y_train)
print(BS_DL_pca.best_params_)
# Best model
DNN_pca=BS_DL_pca.best_estimator_.model
DNN_pca.summary()
#%% DNN-Performance
DNN_pca_train = DNN_pca.predict(pca_x_train).flatten()
DNN_pca_test = DNN_pca.predict(pca_x_test).flatten()
DNN_pca_pred=pd.concat([pd.DataFrame(DNN_pca_train),pd.DataFrame(DNN_pca_test)])

#training
pca_DNN_RMSEr =np.sqrt(MSE(y_train,DNN_pca_train))
print('training RMSE:',DNN_RMSEr)

pca_DNN_MAEr =MAE(y_train,DNN_pca_train)
print('training MAE:',pca_DNN_MAEr)

pca_DNN_MAAPEr =MAAPE(y_train,DNN_pca_train)
print('training MAAPE:',pca_DNN_MAAPEr)

##testing
pca_DNN_RMSEs =np.sqrt(MSE(y_test,DNN_pca_test))
print('testing RMSE:',pca_DNN_RMSEs)

pca_DNN_MAEs =MAE(y_test,DNN_pca_test)
print('testing MAE:',pca_DNN_MAEs)

pca_DNN_MAAPEs =MAAPE(y_test,DNN_pca_test)
print('testing MAAPE:',pca_DNN_MAAPEs)

perf_pca_DNN = pd.DataFrame()
perf_pca_DNN["DNN_train"] =[pca_DNN_RMSEr,pca_DNN_MAEr,pca_DNN_MAAPEr]
perf_pca_DNN["DNN_test"] = [pca_DNN_RMSEs,pca_DNN_MAEs,pca_DNN_MAAPEs]
perf_pca_DNN.index=["RMSE","MAE","MAPE"]
perf_pca_DNN


#%% Y標準化           
y_train, y_test = y_train.values.reshape(-1, 1), y_test.values.reshape(-1, 1)
#%%ori_X標準化
X_train = x_train.reshape((x_train.shape[0], 1, int(x_train.shape[1])))
X_test = x_test.reshape((x_test.shape[0], 1, int(x_test.shape[1])))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
#%% 深度學習參數
GRU_params ={
        'n_hidden':[1,3],
        'n_neurons': [6,128],
        'activation':['relu', 'selu','tanh','softplus'],
        'select_optimizer':Categorical([optimizers.Adam,optimizers.RMSprop]),
        'learning_rate':[0.0005, 0.025], 
        'n_batch_size':[8, 512],
        'n_epochs':[100,200],
        'n_dropout':[0.1,0.2]
            } 
#%%Ori_GRU Model
cell=eval('GRU') 
es = EarlyStopping(monitor='val_loss', mode='min', patience = 20, verbose=1)
def bulid_model(n_dropout, n_hidden,n_neurons,learning_rate, activation, n_epochs, n_batch_size, select_optimizer):
    
    model = Sequential()
    for i in range(n_hidden-1):
        model.add(cell(units=n_neurons,input_shape=(X_train.shape[1], X_train.shape[2]),activation=activation,return_sequences=True))
        
    model.add(cell(units=n_neurons,input_shape=(X_train.shape[1], X_train.shape[2]),activation=activation,return_sequences=False))    
    model.add(Dropout(n_dropout))  
    model.add(Dense(1,activation='linear'))

    optimizer = select_optimizer(lr=learning_rate)
    model.compile(loss='mean_squared_error',optimizer=optimizer)
    
    model.fit(X_train,y_train, epochs=n_epochs, batch_size=n_batch_size, validation_split=0.2, callbacks=[es])
    return model

GRU_keras_ori= KerasRegressor(bulid_model)
#%%Ori 貝氏 
BS_GRU_ori= BayesSearchCV(GRU_keras_ori, GRU_params, n_iter=5, cv=5, random_state=0)
BS_GRU_ori.fit(X_train,y_train)
print(BS_GRU_ori.best_params_)


#%% GRU-Performance
# Best model
GRU_ori= BS_GRU_ori.best_estimator_.model
GRU_ori.summary()

GRU_train = GRU_ori.predict(X_train)
GRU_test = GRU_ori.predict(X_test)
GRU_pred=pd.concat([pd.DataFrame(GRU_train),pd.DataFrame(GRU_test)])

#training
GRU_RMSEr = np.sqrt(MSE(y_train, GRU_train))
print('training RMSE:', GRU_RMSEr)
GRU_MAEr = MAE(y_train, GRU_train)
print('training MAE:', GRU_MAEr)
GRU_MAAPEr = MAAPE(y_train, GRU_train)
print('training MAAPE:', GRU_MAAPEr)


##testing
GRU_RMSEs = np.sqrt(MSE(y_test, GRU_test))
print('testing RMSE:', GRU_RMSEs)
GRU_MAEs = MAE(y_test, GRU_test)
print('testing MAE:', GRU_MAEs)
GRU_MAAPEs = MAAPE(y_test, GRU_test)
print('testing MAAPE:', GRU_MAAPEs)

perf_ori_GRU = pd.DataFrame()
perf_ori_GRU["GRU_train"] = [GRU_RMSEr, GRU_MAEr, GRU_MAAPEr]
perf_ori_GRU["GRU_test"] = [GRU_RMSEs, GRU_MAEs, GRU_MAAPEs]
perf_ori_GRU.index = ["RMSE", "MAE", "MAPE"]


#%%PCA X 標準化
pca_X_train = pca_x_train.reshape((pca_x_train.shape[0], 1, int(pca_x_train.shape[1])))
pca_X_test = pca_x_test.reshape((pca_x_test.shape[0], 1, int(pca_x_test.shape[1])))
print(pca_X_train.shape, y_train.shape, pca_X_test.shape, y_test.shape)
#%%PCA_LSTM GRU Model
cell=eval('GRU') 
es = EarlyStopping(monitor='val_loss', mode='min', patience = 20, verbose=1)
def bulid_pca_model(n_dropout, n_hidden,n_neurons,learning_rate, activation, n_epochs, n_batch_size, select_optimizer):
    
    model = Sequential()
    for i in range(n_hidden-1):
        model.add(cell(units=n_neurons,input_shape=(pca_X_train.shape[1], pca_X_train.shape[2]),activation=activation,return_sequences=True))
        
    model.add(cell(units=n_neurons,input_shape=(pca_X_train.shape[1], pca_X_train.shape[2]),activation=activation,return_sequences=False))    
    model.add(Dropout(n_dropout))  
    model.add(Dense(1,activation='linear'))

    optimizer = select_optimizer(lr=learning_rate)
    model.compile(loss='mean_squared_error',optimizer=optimizer)
    
    model.fit(pca_X_train,y_train, epochs=n_epochs, batch_size=n_batch_size, validation_split=0.2, callbacks=[es])
    return model

GRU_keras_pca= KerasRegressor(bulid_pca_model)
#%%PCA 貝氏 
BS_GRU_pca= BayesSearchCV(GRU_keras_pca, GRU_params, n_iter=5, cv=5, random_state=0)
BS_GRU_pca.fit(pca_X_train,y_train)
print(BS_GRU_pca.best_params_)

#%% GRU-Performance
# Best model
GRU_pca= BS_GRU_pca.best_estimator_.model
GRU_pca.summary()

pca_GRU_train = GRU_pca.predict(pca_X_train)
pca_GRU_test = GRU_pca.predict(pca_X_test)
pca_GRU_pred=pd.concat([pd.DataFrame(pca_GRU_train),pd.DataFrame(pca_GRU_test)])

#training
pca_GRU_RMSEr = np.sqrt(MSE(y_train, pca_GRU_train))
print('training RMSE:', pca_GRU_RMSEr)
pca_GRU_MAEr = MAE(y_train, pca_GRU_train)
print('training MAE:', pca_GRU_MAEr)
pca_GRU_MAAPEr = MAAPE(y_train, pca_GRU_train)
print('training MAAPE:', pca_GRU_MAAPEr)

##testing
pca_GRU_RMSEs = np.sqrt(MSE(y_test, pca_GRU_test))
print('testing RMSE:', pca_GRU_RMSEs)
pca_GRU_MAEs = MAE(y_test, pca_GRU_test)
print('testing MAE:', pca_GRU_MAEs)
pca_GRU_MAAPEs = MAAPE(y_test, pca_GRU_test)
print('testing MAAPE:', pca_GRU_MAAPEs)

perf_pca_GRU = pd.DataFrame()
perf_pca_GRU["GRU_train"] = [pca_GRU_RMSEr, pca_GRU_MAEr, pca_GRU_MAAPEr]
perf_pca_GRU["GRU_test"] = [pca_GRU_RMSEs, pca_GRU_MAEs, pca_GRU_MAAPEs]
perf_pca_GRU.index = ["RMSE", "MAE", "MAPE"]



#%% All the ori performane
perf_tr = pd.DataFrame({
    'Training':['RMSE','MAE','MAAPE'],
    'CART':[round(x, 3) for x in(DT_RMSEr,DT_MAEr,DT_MAAPEr)],
    'RF':[round(x, 3) for x in(RF_RMSEr,RF_MAEr,RF_MAAPEr)],
    'XGB':[round(x, 3) for x in(XGB_RMSEr,XGB_MAEr,XGB_MAAPEr)],
    'SVR':[round(x, 3) for x in(SVR_RMSEr,SVR_MAEr,SVR_MAAPEr)],
    'DNN':[round(x, 3) for x in (DNN_RMSEr,DNN_MAEr,DNN_MAAPEr)],
    'GRU':[round(x, 3) for x in (GRU_RMSEr,GRU_MAEr,GRU_MAAPEr)]
    })

perf_ts = pd.DataFrame({
    'Testing':['RMSE','MAE','MAAPE'],
    'CART':[round(x, 3) for x in(DT_RMSEs,DT_MAEs,DT_MAAPEs)],    
    'RF':[round(x, 3) for x in(RF_RMSEs,RF_MAEs,RF_MAAPEs)],
    'XGB':[round(x, 3) for x in(XGB_RMSEs,XGB_MAEs,XGB_MAAPEs)],
    'SVR':[round(x, 3) for x in(SVR_RMSEs,SVR_MAEs,SVR_MAAPEs)],
    'DNN':[round(x, 3) for x in (DNN_RMSEs,DNN_MAEs,DNN_MAAPEs)],
    'GRU':[round(x, 3) for x in (GRU_RMSEs,GRU_MAEs,GRU_MAAPEs)]
    })

print(perf_tr)
print('\n',perf_ts)

#%% All the pca performane
perf_tr = pd.DataFrame({
    'Training':['RMSE','MAE','MAAPE'],
    'CART':[round(x, 3) for x in(pca_DT_RMSEr,pca_DT_MAEr,pca_DT_MAAPEr)],    
    'RF':[round(x, 3) for x in(pca_RF_RMSEr,pca_RF_MAEr,pca_RF_MAAPEr)],
    'XGB':[round(x, 3) for x in(pca_XGB_RMSEr,pca_XGB_MAEr,pca_XGB_MAAPEr)],
    'SVR':[round(x, 3) for x in(pca_SVR_RMSEr,pca_SVR_MAEr,pca_SVR_MAAPEr)],
    'DNN':[round(x, 3) for x in (pca_DNN_RMSEr,pca_DNN_MAEr,pca_DNN_MAAPEr)],
    'GRU':[round(x, 3) for x in (pca_GRU_RMSEr,pca_GRU_MAEr,pca_GRU_MAAPEr)]
    })

perf_ts = pd.DataFrame({
    'Testing':['RMSE','MAE','MAAPE'],
    'CART':[round(x, 3) for x in(pca_DT_RMSEs,pca_DT_MAEs,pca_DT_MAAPEs)],    
    'RF':[round(x, 3) for x in(pca_RF_RMSEs,pca_RF_MAEs,pca_RF_MAAPEs)],
    'XGB':[round(x, 3) for x in(pca_XGB_RMSEs,pca_XGB_MAEs,pca_XGB_MAAPEs)],
    'SVR':[round(x, 3) for x in(pca_SVR_RMSEs,pca_SVR_MAEs,pca_SVR_MAAPEs)],
    'DNN':[round(x, 3) for x in (pca_DNN_RMSEs,pca_DNN_MAEs,pca_DNN_MAAPEs)],
    'GRU':[round(x, 3) for x in (pca_GRU_RMSEs,pca_GRU_MAEs,pca_GRU_MAAPEs)]
    })

print(perf_tr)
print('\n',perf_ts)



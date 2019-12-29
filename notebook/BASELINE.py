#!/usr/bin/env python
# coding: utf-8

# ## PERSISTENCE MODEL
# 
# This notebooks generates a simple persistence model and test it on 2017 annaul dataset.
# A more advanced model must be compared to this baseline to understand its perforance as a function of the time horizon of interest

# ### IMPORT LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt
from pandas.plotting import autocorrelation_plot
import pickle


# ### DEFINE CUSTOM FUNCTIONS
# 
# Basic function to run the model and make the code modular for a future object-oriented refactoring.

# In[2]:


def load_data(years = ['2013','2014','2015','2016','2017', '2018']):
    file_name_1 = 'elspot-prices_'
    file_name_2 = '_hourly_dkk.csv'
    data_folder = os.getcwd().replace('notebook','data/')
    all_data=[]
    for year in years:
        file_name_tot = data_folder + file_name_1 + year + file_name_2
        all_data.append(pd.read_csv(file_name_tot,encoding = "ISO-8859-1",sep=';',decimal=','))
    df = pd.concat(all_data, ignore_index=True,sort= True)
    return df
    


# In[3]:


def generate_shifted_features(df,time_shift,colname):
    pd.options.mode.chained_assignment = None
    df = df[['datetime','Hours','Oslo']]
    for t in time_shift:
        df.loc[:,colname+'_'+str(t)] = df.Oslo.shift(t)
    return df


# In[4]:


def create_train_test(df,test_size = 8000):
    df = df.dropna()
    train_size = df.shape[0]-test_size
    
    X_train = df.drop(['Oslo','Hours','datetime'],axis=1).head(train_size)
    y_train = df['Oslo'].head(train_size)
    X_test = df.drop(['Oslo','Hours','datetime'],axis=1).tail(test_size)
    y_test = df['Oslo'].tail(test_size)
    
    return X_train, y_train, X_test, y_test
    


# In[5]:


def baseline_model(X_test, y_test):
    y_pred = X_test
    return y_pred


# In[6]:


def predict_baseline_model(X_test,y_test):
    y_pred = baseline_model(X_test, y_test)
    mae = mean_absolute_error(y_test,y_pred)
    rmse = sqrt(mean_squared_error(y_test,y_pred))
    print('MAE = {:.2f}, RMSE = {:.2f}'.format(mae,rmse))
    return y_pred,mae,rmse
    


# In[7]:


def plot_predictions(y_test, y_pred ,plot_samples=100,size=(15,5)):
    plt.figure(figsize=size)
    plt.title('Example of model predictions')
    plt.scatter(range(plot_samples),y_test[:plot_samples],color='b')
    plt.scatter(range(plot_samples),y_pred[:plot_samples],color='r')
    plt.xlabel('time [h]')
    plt.ylabel('Hourly Electricity Price [DKK]')
    plt.show()
    return


# In[8]:


def calculate_multistep_performance(n_steps = 24):
    mae_list = list()
    rmse_list = list()
    df = load_data()
    time_shift = np.arange(1,n_steps+1)
    for t in time_shift: 
        t = [t]
        df_oslo = []
        y_pred = []
        df_oslo = generate_shifted_features(df,t,'Oslo')
        X_train, y_train, X_test, y_test = create_train_test(df_oslo,test_size = 8000)
        y_pred, mae, rmse = predict_baseline_model( X_test , y_test)
        mae_list.append(mae)
        rmse_list.append(rmse)
    return pd.DataFrame({'hour ahead' : time_shift,'mae' : mae_list,'rmse' : rmse_list})


# ### DATA MINING
# 
# For sake of simplicity, Bergan regional data are extracted and processed by default. If you are interested, you can analyse any geographic region contained in Nord Pool dataset.

# In[9]:


df = load_data()

plt.figure(figsize=(17,5))
plt.plot(range(df.shape[0]),df['Oslo'])
plt.show()


# ### PREDICT 1-HOUR STEP

# In[10]:


df = load_data()
time_shift = [1]
df_oslo = generate_shifted_features(df,time_shift,'Oslo')

X_train, y_train, X_test, y_test = create_train_test(df_oslo,test_size = 8000)

y_pred, mae, rmse = predict_baseline_model( X_test , y_test)


# In[26]:


plot_predictions(y_test, y_pred)


# ### HOW DISTANCE IN TIME AFFECTS PERFORMANCE 
# 
# Varying prediction time horizon, we expect persistence model to be dramatically affected by its extremely simplified approach. 

# In[11]:


df_baseline_performance = calculate_multistep_performance(n_steps = 24)


# In[12]:


df_baseline_performance.head()


# ### SAVE TO PICKLE
# For further comparison with other models

# In[13]:


results_pickle = '../saved_session/persistence_results_Oslo.pkl'
with open(results_pickle, "wb") as f:
    pickle.dump(df_baseline_performance, f)


# In[ ]:





# In[ ]:





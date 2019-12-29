#!/usr/bin/env python
# coding: utf-8

# ## DATA VISUALIZATION
# 
# This notebook collects plots and diagrams summarizing main outcomes from previous hourly price analysis

# ### IMPORT LIBRARIES AND FILES

# In[4]:


# import libraries

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle


# In[7]:


# Pickle files

saved_session_dir = '../saved_session/'
persistence_model_results = 'persistence_results.pkl'
lstm_model_results = 'lstm_results.pkl'


# In[20]:


# Load Pickle files

with open(saved_session_dir + persistence_model_results, 'rb') as handle:
    persistence_model_df = pickle.load(handle)
    
with open(saved_session_dir + lstm_model_results, 'rb') as handle:
    lstm_model_df = pickle.load(handle)


# ### PREPROCESSING
# 
# * Remove lag time / hours ahead columns, they are no longer required
# * Rename rmse/mae columns to recognize which model they belong to
# * Merge in a single dataframe

# In[21]:


# Preview
persistence_model_df.head()


# In[22]:


lstm_model_df.head()


# In[23]:


# Drop time column and merge dataframes on Index

lstm_model_df.drop('lag',axis=1,inplace=True)
lstm_model_df.columns = ['lstm_rmse','lstm_mae']
persistence_model_df.drop('hour ahead',axis=1,inplace=True)
persistence_model_df.columns = ['persistence_mae','persistence_rmse']

results_df = pd.concat([persistence_model_df,lstm_model_df], axis = 1)
results_df.head()


# ### PLOT RESULTS
# 
# Compares LSTM model with a a baseline model using two error metrics (MAE and RMSE), varying forecast time horizon. 

# In[44]:


bar_width = 0.3
bar_shift = 0.15


f, ax = plt.subplots(2,1,figsize=(15,10))

ax[0].bar(results_df.index-bar_shift, results_df.persistence_mae, width=bar_width, color='b', align='center')
ax[0].bar(results_df.index+bar_shift, results_df.lstm_mae, width=bar_width, color='r', align='center')
ax[0].legend(['persistence MAE','LSTM MAE'])
ax[0].set_xticks(range(results_df.shape[0]))
ax[0].set_xlabel('Forecast time horizon [h]')
ax[0].set_ylabel('Mean Absolute Error')

ax[1].bar(results_df.index-bar_shift, results_df.persistence_rmse, width=bar_width, color='b', align='center')
ax[1].bar(results_df.index+bar_shift, results_df.lstm_rmse, width=bar_width, color='r', align='center')
ax[1].legend(['persistence RMSE','LSTM RMSE'])
ax[1].set_xticks(range(results_df.shape[0]))
ax[1].set_xlabel('Forecast time horizon [h]')
ax[1].set_ylabel('Root Mean Square Error')

plt.show()


# In[ ]:





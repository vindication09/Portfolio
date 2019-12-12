#!/usr/bin/env python
# coding: utf-8

# # pull_data.py (5 points) 
# When this is called using python pull_data.py in the command line, this will go to the 2 Kaggle urls provided below, authenticate using your own Kaggle sign on, pull the two datasets, and save as .csv files in the current local directory. The authentication login details (aka secrets) need to be in a hidden folder (hint: use .gitignore). There must be a data check step to ensure the data has been pulled correctly and clear commenting and documentation for each step inside the .py file. Training dataset url: https://www.kaggle.com/c/titanic/download/train.csv Scoring dataset url: https://www.kaggle.com/c/titanic/download/test.csv

# In[3]:


import os
import zipfile
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi('{"username":"vindication09","key":"dd1b03f9210cced82665684af4672b29"}')
api.authenticate()
api.competition_download_files("titanic")

zf = zipfile.ZipFile('titanic.zip') # importing zip file from local path
train = pd.read_csv(zf.open('train.csv')) # open train.csv
test = pd.read_csv(zf.open('test.csv')) # open test.csv

if len(train) > 0:
    if len(test) > 0:
        print('train.csv & test.csv are saved as train and test dataframe')
    else:
        print('test.csv is empty')
else:
    print('train.csv is empty')

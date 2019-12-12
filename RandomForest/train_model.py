#!/usr/bin/env python
# coding: utf-8

# # train_model.py (5 points) 
# When this is called using python train_model.py in the command line, this will take in the training dataset csv, perform the necessary data cleaning and imputation, and fit a classification model to the dependent Y. There must be data check steps and clear commenting for each step inside the .py file. The output for running this file is the random forest model saved as a .pkl file in the local directory. Remember that the thought process and decision for why you chose the final model must be clearly documented in this section. eda.ipynb

# In[2]:


# Pandas is used for data manipulation
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import seaborn as sns
import numpy as np
import math
import zipfile
import pickle
# Read in data and display first 5 rows
zf = zipfile.ZipFile('titanic.zip') # importing zip file from local path
train = pd.read_csv(zf.open('train.csv')) # open train.csv
test = pd.read_csv(zf.open('test.csv')) # open test.csv


# In[3]:


# exclude PassengerId, Name
train = train.drop(['Name','PassengerId'], axis=1)


# In[4]:


train = train.drop(['Cabin'], axis=1)


# In[5]:


# Impute with Mode - Embarked
train['Embarked'] = train['Embarked'].fillna('S')

# Dummy variable - Gender and Embarked -- 1 - male, 0 - female
train['Gender'] = train['Sex'].map({'male': 1, 'female': 0})
#train["Gender"] = np.where(train["Sex"].str.contains("male"), 1, 0)
#train["Gender"] = np.where(train["Sex"].str.contains("female"), 0, 1)
train['Emb_code'] = np.where(train['Embarked'].str.contains('S'),1, (np.where(train['Embarked'].str.contains('C'), 2, 3)) )

# Drop Sex and Embarked
train = train.drop(['Embarked'], axis=1)
train = train.drop(['Sex'], axis=1)

# let's split tickets into prefix vs numbers
ticket_new = pd.DataFrame(train.Ticket.str.split(' ',1).tolist(), columns = ['first','Ticket'])
ticket_first = ticket_new['first'][~ticket_new.Ticket.isnull()]
#ticket_second = ticket_new[ticket_new.Ticket.isnull()]
ticket_second = ticket_new['first'][ticket_new.Ticket.isnull()]

for i in ticket_first.index:
    ticket_first[i] = ticket_first[i].replace('/','')
    ticket_first[i] = ticket_first[i].replace('.','')

df_1 = pd.DataFrame(ticket_first)
df_2 = pd.DataFrame(ticket_second)

# convert into binary - 1 - prefix, 0 - no-prefix
df_1['first'] = 1
df_2['first'] = 0

# concat ticket values
df_ticket = pd.concat([df_1,df_2]).sort_index()
train["ticket_prefix"] = df_ticket

# Drop Ticket
train = train.drop(['Ticket'], axis=1)

# Impute with mean - Age -- Let's keep original train dataset just in case.
train_mean = train.copy()
train_mean['Age'] = train_mean['Age'].fillna(train_mean.Age.mean())


# In[6]:


age_floor = list(map(lambda x: math.floor(x), train_mean.Age))
train_mean['Age'] = age_floor


# In[7]:


# Impute with KNN - Age
from missingpy import KNNImputer
imputer = KNNImputer()
X_imputed = imputer.fit_transform(train)

train_knn = pd.DataFrame(pd.DataFrame(X_imputed))
train_knn.columns = train.columns


# In[8]:


age_floor = list(map(lambda x: math.floor(x), train_knn.Age))
train_knn['Age'] = age_floor


# ## model 1: Random Forest using train_knn (imputation of Age with knn)

# In[9]:


x = train_knn.drop('Survived',axis=1)
y = train_knn.Survived


# In[10]:


from sklearn.model_selection import train_test_split
trainX, testX, trainY, testY = train_test_split(x,y,test_size=0.20, random_state=2019)


# ## Random Forest using train_knn with hyperparameters

# In[24]:


# Hyperparameters are configured using GridSearchCV. Please refer to eda.ipynb for more details.
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(bootstrap = True, criterion = 'entropy', max_depth = 10, n_estimators = 3000)


# In[33]:


# save the model to disk
filename = 'randomforest.pkl'
pickle.dump(model, open(filename, 'wb'))

# save the testset to disk
filename2 = 'testx.pkl'
pickle.dump(testX, open(filename2, 'wb'))
filename22 = 'testy.pkl'
pickle.dump(testY, open(filename22, 'wb'))

# save the trainset to disk
filename3 = 'trainx.pkl'
pickle.dump(trainX, open(filename3, 'wb'))
filename4 = 'trainy.pkl'
pickle.dump(trainY, open(filename4, 'wb'))


#!/usr/bin/env python
# coding: utf-8

# # score_model.py (2 points) 
# When this is called using python score_model.py in the command line, this will ingest the .pkl random forest file and apply the model to the locally saved scoring dataset csv. There must be data check steps and clear commenting for each step inside the .py file. The output for running this file is a csv file with the predicted score, as well as a png or text file output that contains the model accuracy report (e.g. sklearn's classification report or any other way of model evaluation).

# In[16]:


# load the .pkl files from disk
import pickle
import pandas as pd

filename = 'randomforest.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

filename2 = 'testx.pkl'
testX = pickle.load(open(filename2, 'rb'))
filename22 = 'testy.pkl'
testY = pickle.load(open(filename22, 'rb'))

filename3 = 'trainx.pkl'
trainX = pickle.load(open(filename3, 'rb'))
filename4 = 'trainy.pkl'
trainY = pickle.load(open(filename4, 'rb'))


# In[48]:


# load the model and produce prediction output
loaded_model.fit(trainX, trainY)
y_pred = loaded_model.predict(testX)

from sklearn.metrics import accuracy_score
res = pd.DataFrame({"Model":['RandomForestClassifier - Age with KNN (Hyperparameter)'],
                    "Accuracy Score": [accuracy_score(y_pred,testY)]})

# save output of y_prediction in .csv
pd.DataFrame(y_pred)[0].to_csv(r'predicted_score.csv')

# save output of accuracy in .csv
res.to_csv(r'accuracy_score.csv')

# save model accuracy report in .png
from sklearn.metrics import classification_report
#c_rpt = print(classification_report(testY, y_pred))
report = classification_report(testY, y_pred, output_dict = True)
f = open("classification_report.txt","w")
f.write( str(report) )
f.close()

#print(classification_report(testY, y_pred))

# I could also save classification report in .csv - this is just a bonus.
report = pd.DataFrame(report)
report.to_csv(r'classification_report.csv')


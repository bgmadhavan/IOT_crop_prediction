from flask import Flask, render_template, request
# Importing libraries
# from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
import serial
import random
warnings.filterwarnings('ignore')
PATH = 'C:/Users/bgmad/Desktop/new frontend/crop prediction/Crop_recommendation.csv'


df = pd.read_csv(PATH)



lst = []






app = Flask(__name__)
@app.route('/')
def index():
  global lst
  # n = random.randint(60.0,96.0)             #request.form['n']
  # p = random.randint(35.0,61.0) #request.form['p']
  # k = random.randint(19.0,46.0)            #request.form['k']
  # # t = request.form['t']
  # # h = request.form['h']
  # pp = random.uniform(5.3,7.6)           #request.form['ph']
  # # r = request.form['r']

  #----------------------------------new code----------------------------------
  ser = serial.Serial('COM6', 9600)
  x=5 
  with open('output.txt', 'a') as file:
    while (x):
      data = ser.readline().decode('utf-8').strip()
      gh=data.split(',')
      file.write(data + '\n')
      x-=1
  h,t,r=gh[0],gh[1],gh[2]

  lst.extend([t,h,r])
  return render_template('index.html',t=t,h=h,r=r)


@app.route('/',methods = ['POST'])
def getValue():
  global lst
#Separating features and target
  n = request.form['n']
  p = request.form['p']
  k = request.form['k']
  pp= request.form['ph']
  lst=[n,p,k,lst[0],lst[1],pp,lst[2]]


  features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
  target = df['label']
  labels = df['label']
  acc = []
  model = []
  from sklearn.model_selection import train_test_split
  Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)
  from sklearn.tree import DecisionTreeClassifier
  DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)
  DecisionTree.fit(Xtrain,Ytrain)
  predicted_values = DecisionTree.predict(Xtest)
  x = metrics.accuracy_score(Ytest, predicted_values)
  acc.append(x)
  model.append('Decision Tree')
  from sklearn.model_selection import cross_val_score
  score = cross_val_score(DecisionTree, features, target,cv=5)
  from sklearn.ensemble import RandomForestClassifier
  RF = RandomForestClassifier(n_estimators=20, random_state=0)
  RF.fit(Xtrain,Ytrain)
  predicted_values = RF.predict(Xtest)
  x = metrics.accuracy_score(Ytest, predicted_values)
  acc.append(x)
  model.append('RF')
  import pickle
  # Dump the trained Random forest classifier with Pickle
  DT_pkl_filename = 'DecisionTree.pkl'
  # Open the file to save as pkl file
  DT_Model_pkl = open(DT_pkl_filename, 'wb')
  pickle.dump(DecisionTree, DT_Model_pkl)
  # Close the pickle instances
  DT_Model_pkl.close()
  data = np.array([lst])
  prediction = RF.predict(data)

  return render_template('index.html',st = prediction[0],n=lst[0],p=lst[1],k=lst[2],ph=lst[5],t=lst[3],h=lst[4],r=lst[6])

if __name__ == '__main__':
  app.run(debug=True)
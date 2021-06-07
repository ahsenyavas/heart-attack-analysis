# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 16:40:58 2021

@author: ahsen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Metrics
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score

# Cross Validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

import warnings 
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)

df = pd.read_csv("heart.csv")


print("The shape of the dataset: ", df.shape)

print(df.head())

# categorical column 
cat_cols = ["sex", "exng", "cp", "fbs", "restecg", "thall", "slp", "caa"]

# continuous column
con_cols = ["age", "trtbps", "chol", "thalachh", "oldpeak"]

# target 
target_col = ["output"]

print(df[con_cols].describe().transpose()) # there are outliers in data

print(df[['age', 'chol']].plot(kind='scatter', x='age' ,y='chol'))

# NaN values 
print(df.isna().sum()) # there are no NaN values

# Visualization

plt.figure(figsize=(20, 10))
sns.displot(x = df["age"])
plt.title("Distribution of Age", fontsize=20)
#plt.show()

sns.displot(x = df["trtbps"])
plt.title("Distribution of Blood Pressure")
plt.xlabel("Blood Pressure", fontsize=10)
plt.ylabel("Count", fontsize=10)
#plt.show()

plt.figure(figsize=(10,6))
sns.histplot(data = df, x = 'age', hue = 'output')
plt.title("Does Age Effect The Heart Attack")
#plt.show()

plt.figure(figsize=(20,10))
sns.lineplot(y="trtbps",x="age",data=df)
plt.title("BLOOD PRESSURE WITH AGE",fontsize=20)
plt.xlabel("AGE",fontsize=20)
plt.ylabel("BLOOD PRESSURE",fontsize=20)
#plt.show()



# copy of df
df1 = df

df1 = pd.get_dummies(df1, columns = cat_cols, drop_first = True)

X = df1.drop(['output'], axis=1)
y = df1[['output']]

scaler = preprocessing.StandardScaler()
X[con_cols] = scaler.fit_transform(X[con_cols])

print(X.head())


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
print("The shape of X_train is      ", X_train.shape)
print("The shape of X_test is       ",X_test.shape)
print("The shape of y_train is      ",y_train.shape)
print("The shape of y_test is       ",y_test.shape)


score = []
models = [
          SVC(),
          LogisticRegression(),
          DecisionTreeClassifier(),
          RandomForestClassifier(),
          GradientBoostingClassifier()
          ]

for model in models:
    f = model.fit(X_train, y_train)
    y_pred = f.predict(X_test)
    score.append(accuracy_score(y_test, y_pred))
    
for i in range(len(score)):
    print(f"{models[i]}: {score[i]}")



logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_proba = logreg.predict_proba(X_test)

# calculating the probabilities
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# instantiating the roc_cruve
fpr,tpr,threshols=roc_curve(y_test,y_pred_prob)


logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistric Regression ROC Curve')
plt.legend(loc = 'lower right')
#plt.savefig('Log_ROC')
plt.show()

















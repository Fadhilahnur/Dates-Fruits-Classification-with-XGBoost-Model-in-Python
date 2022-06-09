
"""
Author : Fadhilah Nur Binti Ismail
Title : Dates Fruits Classification with XGBoost Model in Python
Objective: The aim of this study is to classify the types of date fruit, that are, Barhee, Deglet Nour, Sukkary, Rotab Mozafati, Ruthana, Safawi, and Sagai. 898 images of seven different date fruit types were obtained via the computer vision system (CVS). Through image processing techniques, a total of 34 features, including morphological features, shape, and color, were extracted from these images.
Date : June 2022
"""

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from xgboost import XGBClassifier

import warnings
import missingno as msno
from warnings import warn
warnings.filterwarnings('ignore')


data_path = r'''C:\Users\FadhilahNurIsmail\OneDrive - KDS Consulting Sdn Bhd\2022 May Work\Predictive Analytics\Testing Model Data\Date_Fruit_Datasets.xlsx'''
data=pd.read_excel(data_path)

df = data.copy()
df

def InfoData(dataframe,target_variable = None):
    
    print(f"""
== DATA INFO ==
* Shape: {dataframe.shape}
* Number of data = {dataframe.shape[0]}


== COLUMNS INFO ==
* Number of columns: {len(dataframe.columns)}
* Columns with dtype: 
{dataframe.dtypes}


== Missing / Nan Values ==
* Is there any missing value?: {dataframe.isnull().values.any()}
    """)
    
    if (target_variable != None) and (target_variable in dataframe.columns):    
        
        print(f"""
== TARGET VARIABLE ==

* Variable: {target_variable}
* Values of Variable: {" - ".join(df.Class.unique())}
* Count of Values: {len(df.Class.unique())}
""")
        
        
        
    elif (target_variable != None) and (target_variable not in dataframe.columns): 
        print("Please type correctly your target variable")
        
        
InfoData(df,"Class")

" ============GET MIN, MAX, STANDARD DEVIATION AND MEAN VALUES FOR ALL VARIABLES (EXCEPT OBJECT VARIABLE)"


for i in df.columns[:-1]: # I dont want to / can not see last column("Class") because it is target variable and it is an object
    print(f"{i}: | Min: {df[i].min():.4f} | Max: {df[i].max():.4f} | Std: {df[i].std():.4f} | Mean: {(df[i].mean()):.4f}")

"============ DATA PREPROCESSING==================================================="

le = LabelEncoder()
target = df['Class']
target = le.fit_transform(target)

X = df.drop("Class",axis=1)

train_test_split_params = {"test_size":0.33,
                        "random_state":1}

X_train, X_test, y_train, y_test = train_test_split(X, target,
                                                    test_size=train_test_split_params["test_size"],
                                                    random_state=train_test_split_params["random_state"],
                                                    shuffle=True)


print(f"""
X_train shape: {X_train.shape}
X_test shape: {X_test.shape}
y_train shape: {y_train.shape}
y_test shape: {y_test.shape}
""")

"=============== XGBOOST MODEL======================"
"Use RandomizeGridSearch to select parameters"
"Use StratifiedKFold because we are dealing with imbalanced class distributions as shown in barplot below."

sns.barplot(y=df["Class"].value_counts().index,x=df["Class"].value_counts().values);

params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'eval_metric': ["mlogloss"],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [0, 3, 4]
        }


"================= RANDOMIZED SEARCH FOR BEST PARAMETERS==========="

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1)

randomized_search = RandomizedSearchCV(XGBClassifier(), param_distributions=params, n_iter=5, cv=skf.split(X_train,y_train), verbose=3, random_state=1)

randomized_search.fit(X_train, y_train)

print('Best hyperparameters:', randomized_search.best_params_)

"===================CLASSIFY USING BEST PARAMETERS ======================"

xgb = XGBClassifier(subsample = randomized_search.best_params_["subsample"],
                      min_child_weight = randomized_search.best_params_["min_child_weight"],
                      max_depth = randomized_search.best_params_["max_depth"],
                      learning_rate = randomized_search.best_params_["learning_rate"],
                      gamma = randomized_search.best_params_["gamma"],
                      eval_metric = randomized_search.best_params_["eval_metric"],
                      colsample_bytree = randomized_search.best_params_["colsample_bytree"])

"=================== FIT TRAIN SET ====================================="
xgb.fit(X_train, y_train)

"XGBOOST PREDICTTION USING TRAIN SET AND COMPARE ACCURACY TO TEST SET"
train_pred = xgb.predict(X_train)
train_acc = accuracy_score(y_train,train_pred)
print('Train Accuracy: ', train_acc)
 
test_pred = xgb.predict(X_test)
test_acc = accuracy_score(y_test,test_pred)
print('Test Accuracy:', test_acc)
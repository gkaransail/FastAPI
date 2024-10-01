import os 
import sys 
import numpy as np
import pandas as pd 
import joblib

import sklearn 
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_curve, auc 
from category_encoders import OrdinalEncoder
import warnings 
warnings.filterwarnings("ignore")


## Files
data_file = "C:\\Users\\kgupta\\Downloads\\Placement_Data_Full_Class.csv"

# Load train loan dataset 
try:
    data = pd.read_csv(data_file)
    print("The dataset has {} samples with {} features.".format(*data.shape))
except:
    print("The dataset could not be loaded. Is the dataset missing?")


exclude_feature = ['sl_no', 'salary', 'status']
# Define Target columns
target = data['status'].map({"Placed": 0 , "Not Placed": 1})

# Define numeric and categorical features
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
numeric_features = [col for col in numeric_columns if col not in exclude_feature]
categorical_features = [col for col in categorical_columns if col not in exclude_feature]

# Define final feature list for training and validation
features = numeric_features + categorical_features
# Final data for training and validation
data = data[features]
data = data.fillna(0)

# Split data in train and vlaidation
X_train, X_valid, y_train, y_valid = train_test_split(data, target, test_size=0.15, random_state=10)
X_valid.to_json(path_or_buf='valid.json', orient='records', lines=True)

# Perform label encoding for categorical variable
le = OrdinalEncoder(cols=categorical_features)
le.fit(X_train[categorical_features])
X_train[categorical_features] = le.transform(X_train[categorical_features])
X_valid[categorical_features] = le.transform(X_valid[categorical_features])

print(X_train)


 # Perform model training
clf = LGBMClassifier(random_state=10)
clf.fit(X_train, y_train)

# Perform model evaluation 
valid_prediction = clf.predict_proba(X_valid)[:, 1]
fpr, tpr, thresholds = roc_curve(y_valid, valid_prediction)
roc_auc = auc(fpr, tpr) # compute area under the curve
print("=====================================")
print("Validation AUC:{}".format(roc_auc))
print("=====================================")


# Perform model evaluation 
print(classification_report(y_valid,clf.predict(X_valid)))


joblib.dump(le, 'label_encoder.joblib')
joblib.dump(clf, 'lgb_model.joblib')
joblib.dump(features, 'features.joblib')
joblib.dump(categorical_features, 'categorical_features.joblib')
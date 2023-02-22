import boto3
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from itertools import product
from sklearn.metrics import recall_score, accuracy_score

## Defining the s3 bucket
s3 = boto3.resource('s3')
bucket_name = 'ryan-greiner-bucket'
bucket = s3.Bucket(bucket_name)

## Defining the file to be read from s3 bucket
file_key80 = 'churn-bigml-80.csv'
bucket_object80 = bucket.Object(file_key80)
file_object80 = bucket_object80.get()
file_content_stream80 = file_object80.get('Body')

file_key20 = 'churn-bigml-20.csv'
bucket_object20 = bucket.Object(file_key20)
file_object20 = bucket_object20.get()
file_content_stream20 = file_object20.get('Body')

## Reading CSV file
telecom_train = pd.read_csv(file_content_stream80)
telecom_test = pd.read_csv(file_content_stream20)

## Churn number ##
telecom_train['Churn_numb'] = np.where(telecom_train['Churn'] == True, 1, 0)
telecom_test['Churn_numb'] = np.where(telecom_test['Churn'] == True, 1, 0)

## International plan number ##
telecom_train['International_plan'] = np.where(telecom_train['International_plan'] == 'Yes', 1, 0)
telecom_test['International_plan'] = np.where(telecom_test['International_plan'] == 'Yes', 1, 0)

## Voice mail plan number ##
telecom_train['Voice_mail_plan'] = np.where(telecom_train['Voice_mail_plan'] == 'Yes', 1, 0)
telecom_test['Voice_mail_plan'] = np.where(telecom_test['Voice_mail_plan'] == 'Yes', 1, 0)

## Total charge ##
telecom_train['total_charge'] = (telecom_train['Total_day_charge']+ 
                                 telecom_train['Total_eve_charge']+ 
                                 telecom_train['Total_night_charge']+ 
                                 telecom_train['Total_intl_charge'])
telecom_test['total_charge'] = (telecom_test['Total_day_charge']+ 
                                telecom_test['Total_eve_charge']+ 
                                telecom_test['Total_night_charge']+ 
                                telecom_test['Total_intl_charge'])

## Inputs and target ##
telecom_train = telecom_train[['Account_length',
                               'International_plan', 
                               'Voice_mail_plan', 
                               'total_charge', 
                               'Customer_service_calls', 
                               'Churn_numb']]
telecom_test = telecom_test[['Account_length', 
                             'International_plan', 
                             'Voice_mail_plan', 
                             'total_charge', 
                             'Customer_service_calls', 
                             'Churn_numb']]
X = telecom_train.drop(columns = ['Churn_numb'], axis = 1)
Y = telecom_train['Churn_numb']

## Define lists ##
rf_importance = list()
ada_importance = list()
gb_importance = list()

####################### IMPORTANCES #############################
for i in range(0, 1000):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y)
    ## Random Forest ##
    rf_md = RandomForestClassifier(n_estimators = 500, max_depth = 3).fit(X_train, Y_train)
    rf_importance.append(rf_md.feature_importances_)
    
    ## AdaBoost ##
    ada_md = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 3), n_estimators = 500, learning_rate = .01).fit(X_train, Y_train)
    ada_importance.append(ada_md.feature_importances_)
    
    ## Gradient Boosting ##
    gb_md = GradientBoostingClassifier(max_depth = 3, n_estimators = 500, learning_rate = .001).fit(X_train, Y_train)
    gb_importance.append(gb_md.feature_importances_)
        
## Define lists to calculate averages ##
rf_avg = list()
ada_avg = list()
gb_avg = list()

rf_avg.append(rf_importance[0])
ada_avg.append(ada_importance[0])
gb_avg.append(gb_importance[0])

## Calculate Averages ##
for i in range(1, len(rf_importance)):
    rf_avg = rf_avg + rf_importance[i]
    ada_avg = ada_avg + ada_importance[i]
    gb_avg = gb_avg + gb_importance[i]
    
rf_avg = rf_avg/len(rf_importance)
ada_avg = ada_avg/len(ada_importance)
gb_avg = gb_avg/len(gb_importance)



## Display Averages ##
importances = pd.DataFrame({'Input Variable': ['Account_length',
                                               'International_plan', 
                                               'Voice_mail_plan', 
                                               'total_charge', 
                                               'Customer_service_calls'],
                            'RF Importance': [rf_avg[0][0],
                                              rf_avg[0][1],
                                              rf_avg[0][2],
                                              rf_avg[0][3],
                                              rf_avg[0]][4],
                            'Ada Importance': [ada_avg[0][0],
                                               ada_avg[0][1],
                                               ada_avg[0][2],
                                               ada_avg[0][3],
                                               ada_avg[0]][4],
                            'GB Importance': [gb_avg[0][0],
                                              gb_avg[0][1],
                                              gb_avg[0][2],
                                              gb_avg[0][3],
                                              gb_avg[0][4]]})

importances.to_csv('Importances.csv', index = False)


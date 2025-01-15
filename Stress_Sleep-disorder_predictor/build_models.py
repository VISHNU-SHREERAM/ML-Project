import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# test_data is kept very low to maximize the usage of data for training. This is done after checking the performance of the model with different test sizes.


for i in range(1):
    df_part1 = pd.read_csv('./Sleep_health_and_lifestyle_dataset.csv')
    df_part2 = pd.read_csv('./Sleep_health_and_lifestyle_dataset_part_2.csv')
    df = pd.concat((df_part1, df_part2))
    df.drop('Person ID', axis=1, inplace=True)
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
    # In the BMI category, converting the 'Normal Weight' to 'Normal' as part of data cleaning
    df['BMI Category'] = df['BMI Category'].replace('Normal Weight', 'Normal')
    lowRow = []
    highRow = []
    for val in df['Blood Pressure']:
        sp = val.split('/')
        highRow.append(int(sp[0]))
        lowRow.append(int(sp[1]))
    df['Diastolic Pressure'] = np.array(lowRow)
    df['Systolic Pressure'] = np.array(highRow)
    df = df.drop('Blood Pressure', axis = 1)
    catData = [key for key in df if (df[key].dtype == object)]
    numData = [key for key in df if (df[key].dtype != object and key != 'Stress Level')]
    label_encoding_dict = {}
    scaler_dict = {}
    unique_values = {}
    for category in catData:
        unique_values[category] = df[category].unique()
        labelEncoder = LabelEncoder()
        labelEncoder.fit(df[category])
        df[category] = labelEncoder.transform(df[category])
        label_encoding_dict[category] = labelEncoder
        
    for f in numData:
        scaler = StandardScaler()
        df[f] = scaler.fit_transform(df[[f]])
        scaler_dict[f] = scaler

# Regression
tgt1 = df['Stress Level']
feat1 = df.drop(columns = ['Stress Level','Sleep Disorder'])

# X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
#     feat1, tgt1, test_size=0.01, random_state=42)

from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
regr = BaggingRegressor(estimator=SVR(),
                        n_estimators=80).fit(feat1, tgt1) #random_state=0
# y_pred_reg = regr.predict(X_test_reg)

# from sklearn.metrics import r2_score
# print('r2 score' , r2_score(y_test_reg,y_pred_reg))
# Classification
from sklearn.ensemble import RandomForestClassifier
tgt2 = df['Sleep Disorder']
feat2 = df.drop(columns = ['Sleep Disorder','Stress Level'])
# X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(feat2, tgt2, test_size=0.01, ) #random_state=42
clf = RandomForestClassifier(n_estimators=150, random_state=0)
clf.fit(feat2, tgt2)
# y_pred_clf = clf.predict(X_test_clf)
# from sklearn.metrics import accuracy_score
# print('accuracy score' , accuracy_score(y_test_clf,y_pred_clf))

# Save the model
import joblib
joblib.dump(regr, './regr.pkl')
joblib.dump(clf, './clf.pkl')
joblib.dump(label_encoding_dict, './label_encoding_dict.pkl')
feature_columns = list(df.columns)
joblib.dump(feature_columns, './feature_columns.pkl')
joblib.dump(scaler_dict, './scaler_dict.pkl')
joblib.dump(catData, './catData.pkl')
joblib.dump(numData, './numData.pkl')
joblib.dump(unique_values, './unique_values.pkl')


# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
df = pd.read_csv("bmi.csv")  
print("Original Dataset:")
print(df.head())
```
<img width="352" height="166" alt="image" src="https://github.com/user-attachments/assets/2220076d-558e-4700-9d28-e6d03bb2e8a9" />

```
df = df.dropna()
df_std = df.copy()
scaler_std = StandardScaler()
df_std[['Height', 'Weight']] = scaler_std.fit_transform(df_std[['Height', 'Weight']])
print("\nStandard Scaled Data:")
print(df_std.head())
```
<img width="451" height="188" alt="image" src="https://github.com/user-attachments/assets/9ef34945-deae-4de0-9442-64c6200c6c17" />

```

df_minmax = df.copy()
scaler_minmax = MinMaxScaler()
df_minmax[['Height', 'Weight']] = scaler_minmax.fit_transform(df_minmax[['Height', 'Weight']])

print("\nMin-Max Scaled Data:")
print(df_minmax.head())
```
<img width="402" height="177" alt="image" src="https://github.com/user-attachments/assets/cf6cf80d-9aee-41ee-bcb0-3980a4c01fe9" />

```
df_maxabs = df.copy()
scaler_maxabs = MaxAbsScaler()
df_maxabs[['Height', 'Weight']] = scaler_maxabs.fit_transform(df_maxabs[['Height', 'Weight']])
print("\nMaxAbs Scaled Data:")
print(df_maxabs.head())
```
<img width="402" height="182" alt="image" src="https://github.com/user-attachments/assets/e910b69c-ee4b-430d-a24c-8feb782c15bd" />

```
df_robust = df.copy()
scaler_robust = RobustScaler()
df_robust[['Height', 'Weight']] = scaler_robust.fit_transform(df_robust[['Height', 'Weight']])

print("\nRobust Scaled Data:")
print(df_robust.head())
print("\nFeature Scaling Completed Successfully.")
```
<img width="442" height="217" alt="image" src="https://github.com/user-attachments/assets/6bf1f8c8-2c06-4367-b91c-837ea96f3149" />

```
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE, SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
df = pd.read_csv("income(1) (1).csv")
print("Dataset Preview:")
print(df.head())
```
<img width="782" height="465" alt="image" src="https://github.com/user-attachments/assets/a5311719-9495-4aff-a315-fff6aaae9b7d" />

```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation',
                       'relationship', 'race', 'gender', 'nativecountry']

df[categorical_columns] = df[categorical_columns].astype('category').apply(lambda x: x.cat.codes)
if df['SalStat'].dtype == 'object':
    df['SalStat'] = df['SalStat'].astype('category').cat.codes
X = df.drop(columns=['SalStat'])
y = df['SalStat']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

selector_chi2 = SelectKBest(score_func=chi2, k=6)
selector_chi2.fit(X_scaled, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("\nChi-Square Selected:", list(selected_features_chi2))

selector_anova = SelectKBest(score_func=f_classif, k=5)
selector_anova.fit(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nANOVA Selected:", list(selected_features_anova))

logreg = LogisticRegression(max_iter=1000)
rfe = RFE(estimator=logreg, n_features_to_select=6)
rfe.fit(X, y)
selected_features_rfe = X.columns[rfe.support_]
print("\nRFE Selected:", list(selected_features_rfe))
```
<img width="1003" height="102" alt="image" src="https://github.com/user-attachments/assets/4f986ee6-c830-4c9a-86a0-dfd24b247538" />
<img width="902" height="47" alt="image" src="https://github.com/user-attachments/assets/1cede4fb-c14e-421e-81e5-ce2848586f05" />

```
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
embedded_selector = SelectFromModel(model, threshold='median')
embedded_selector.fit(X, y)
selected_features_embedded = X.columns[embedded_selector.get_support()]
print("\nEmbedded Method Selected:", list(selected_features_embedded))
X_train, X_test, y_train, y_test = train_test_split(
    X[selected_features_embedded], y,
    test_size=0.2,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy (Embedded Method):", accuracy)
```
<img width="1056" height="95" alt="image" src="https://github.com/user-attachments/assets/3c4d31d1-3412-46c9-8e65-1150a0f783ca" />


# RESULT:
Thus the Feature Scaling and selection Executed successfully.

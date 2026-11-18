import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV

df = sns.load_dataset("titanic")

df.info()


sns.barplot(data= df, x= "sex", y= "survived")
plt.show()

sns.catplot(data= df, x= "pclass", hue= "survived", kind= "count")
plt.show()

sns.histplot(data= df, hue= "survived", x= "age")


sns.countplot(data= df,x= "who", hue= "survived")

df.isnull().sum()

df = df.drop(["deck", "embark_town", "alive"], axis= 1)

df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

df.isnull().sum()

df["adult_male"] = df["adult_male"].astype(int)
df["alone"] = df["alone"].astype(int)

X = df.drop("survived", axis = 1)
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 15)

categorical_cols = ["sex", "class", "embarked", "who"]

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(drop= "first", handle_unknown= "ignore"), categorical_cols),
],
    remainder= "passthrough"
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

encoded_cols = preprocessor.get_feature_names_out()

X_train = pd.DataFrame(X_train, columns=encoded_cols)
X_test = pd.DataFrame(X_test, columns=encoded_cols)

clf = lgb.LGBMClassifier(verbosity = -1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

importances = clf.feature_importances_
feature_names = X_train.columns

feature_importances = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": importances,
    }
).sort_values(by= "importance", ascending= False)

feature_importances

lgb_model = lgb.LGBMClassifier(verbosity = -1)

param_grid = {
    "n_estimators": [100, 300, 500, 1000],
    "max_depth": [3, 5, 7,-1],
    "num_leaves" : [15, 31, 63, 127],
    "learning_rate" : [0.01, 0.05, 0.1, 0.3],
    "min_child_samples" : [10, 20, 30],
    "subsample" : [0.6, 0.8, 1.0],
    "colsample_bytree" : [0.6, 0.8, 1.0]
}

random_search = RandomizedSearchCV(estimator= lgb_model, param_distributions = param_grid, cv= 5, scoring= "accuracy", n_jobs= -1, verbose= 1)
random_search.fit(X_train, y_train)

random_search.best_params_

y_pred = random_search.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
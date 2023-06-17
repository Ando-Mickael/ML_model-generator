import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

data_frame = pd.read_csv('train.csv') # put your CSV file here
X = data_frame.iloc[:, 0:-1].values
y = data_frame.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
logistic_regression_model = LogisticRegression(random_state=42)
logistic_regression_model = logistic_regression_model.fit(X_train, y_train)
y_pred = logistic_regression_model.predict(X_test)
print(accuracy_score(y_test, y_pred))
joblib.dump(logistic_regression_model, 'logistic_regression_model.joblib')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

data_frame = pd.read_csv('train.csv') # put your CSV file here
X = data_frame.iloc[:, 0:-1].values
y = data_frame.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
random_forest_model = RandomForestClassifier(random_state=0)
random_forest_model = random_forest_model.fit(X_train, y_train)
y_pred = random_forest_model.predict(X_test)
print(accuracy_score(y_test, y_pred))
joblib.dump(random_forest_model, 'random_forest_model.joblib')
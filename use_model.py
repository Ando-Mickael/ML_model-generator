import joblib
import sys


def string_to_float_array(string):
    return [float(x) for x in string.split(",")]


X_test = [string_to_float_array(sys.argv[1])]
print(X_test)

model = joblib.load("model.joblib") # put your model here
prediction = model.predict(X_test)
print(prediction[0])

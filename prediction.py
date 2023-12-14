import joblib
def predict_data(data):
    clf = joblib.load("rf_model.sav")
    return clf.predict(data)
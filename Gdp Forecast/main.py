import joblib

Poly_regressor = joblib.load('Poly_regressor.joblib')
Model = joblib.load('Model.joblib')

for _ in range(5):
    User = int(input('Enter Year :'))

    Data = Model.predict(Poly_regressor.fit_transform([[User]]))
    print(Data[0] / 1000000000000, " trillion")
import joblib

Vectorizer = joblib.load('./Email_vectorizer.joblib')
Model = joblib.load('./Email_Spam.joblib')

for _ in range(3):
    User = input('Email Text :')
    convert = Vectorizer.transform([User])

    Prediction = Model.predict(convert)

    if Prediction == 0:
        print("Not Spam")

    else:
        print("Spam")

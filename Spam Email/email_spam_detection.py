# -*- coding: utf-8 -*-
"""Email_Spam_Detection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1m5TXxRzO2a4XXc2HAEG7cKg8jzAN1PGQ
"""

import pandas as pd

Data =pd.read_csv("combined_data.csv")
Data.head()

Data.columns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

Input_Data = Data['text']
Target = Data['label']

Input_Data_train, Input_Data_test, Target_train, Target_test = train_test_split(Input_Data, Target, test_size=0.2, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
Input_Data_Train = vectorizer.fit_transform(Input_Data_train)

Model = DecisionTreeClassifier()
Model.fit(Input_Data_Train, Target_train)

Model_Input_Data_Test = vectorizer.transform(Input_Data_test)
Prediction = Model.predict(Model_Input_Data_Test)

Accuracy = accuracy_score(Target_test, Prediction)
print("Accuracy:", Accuracy)

import joblib

joblib.dump(Model, 'Email_Spam.joblib')
joblib.dump(vectorizer, 'Email_vectorizer.joblib')

Model.predict(vectorizer.transform(["Hey how are you?"]))
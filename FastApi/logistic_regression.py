import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv("analysed_churn.csv")
y = df["Churn"]
X = df.drop(["Churn", "customerID"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

x_train , x_test , y_train , y_test = train_test_split(X,y,test_size=0.25, random_state = 35)
logreg_model = LogisticRegression()
logreg_model.fit(x_train, y_train)


# Modeli kaydet
pickle.dump(logreg_model,open("logistic_regression_model.pkl","wb"))

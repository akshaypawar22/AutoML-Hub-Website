# Importing the libraries
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('myNewModel.csv')

y = dataset.iloc[:, -1]
X = dataset.iloc[:, :16]


sc = StandardScaler()
X = sc.fit_transform(X)


# Splitting Training and Test Set
# Since we have a very small dataset, we will train our model with all availabe data.

# regressor = LinearRegression()
# lr = LogisticRegression()
svc = SVC(kernel='linear')

# Fitting model with trainig data
svc.fit(X, y)

# Saving model to disk
pickle.dump(svc, open('model.pkl', 'wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([7, 1, 12, 2, 0, 0, 0, 1, 2, 4, 23, 3, 2, 51590, 4, 2, 2]))

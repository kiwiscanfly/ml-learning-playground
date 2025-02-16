import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from matplotlib import pyplot as plt
import seaborn as sns

data = pd.read_csv('titanic.csv')
data.info()
print(data.isnull().sum())

def preprocess_data(df):
  df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], inplace=True)

  # Fill missing values
  df['Fare'].fillna(df['Fare'].mean(), inplace=True)

  fill_missing_ages(df)

  # Convert Sex to binary
  df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

  df['FamilySize'] = df['SibSp'] + df['Parch']
  df['IsAlone'] = np.where(df['FamilySize'] == 0, 1, 0)
  df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False)

  return df

def fill_missing_ages(df):
  age_fill_map = {}
  for pclass in df['Pclass'].unique():
    if pclass not in age_fill_map:
      age_fill_map[pclass] = df[df['Pclass'] == pclass]['Age'].median()

  df['Age'] = df.apply(lambda row: age_fill_map[row['Pclass']] if np.isnan(row['Age']) else row['Age'], axis=1)

data = preprocess_data(data)

print('After preprocessing')
print(data.isnull().sum())

X = data.drop(columns=['Survived'])
Y = data['Survived']

print('X')
print(X)
print('Y')
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# ML Preprocessing
scalar = MinMaxScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

# Hyperparameter tuning
def tune_model(X_train, Y_train):
  param_grid = {
    'n_neighbors': np.arange(1, 11),  # Reduced range for simplicity
    'metric': ['euclidean', 'manhattan'],
    'weights': ['uniform', 'distance']
  }

  model = KNeighborsClassifier()
  grid = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
  grid.fit(X_train, Y_train)
  return grid.best_estimator_

model = tune_model(X_train, Y_train)

def evaluate_model(model, X_test, Y_test):
  prediction = model.predict(X_test)
  accuracy = accuracy_score(Y_test, prediction)
  confusion = confusion_matrix(Y_test, prediction)
  return accuracy, confusion

accuracy, confusion = evaluate_model(model, X_test, Y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')
print('Confusion Matrix:', confusion)
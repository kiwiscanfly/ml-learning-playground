import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

class Sentiment:
  POSITIVE = 'positive'
  NEGATIVE = 'negative'
  # POSITIVE = 1
  # NEGATIVE = 0

file_path = 'imdb_sentiment_reduced.csv'
separator = ','

# file_path = 'imdb_sentiment_small.txt'
# separator = '\t'

column_names = ['review', 'sentiment']
data = pd.read_csv(file_path, sep=separator, names=column_names)

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

train_x = train_data['review']
train_y = train_data['sentiment']
test_x = test_data['review']
test_y = test_data['sentiment']

## Bag of Words Vectorisation
vectorizer = TfidfVectorizer() # CountVectorizer() does just a simple word count, TfidfVectorizer() does a word count with a weighting based on the frequency of the word in the document
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)

print(f"Training vectors shape: {train_x_vectors.shape}")
print(f"Testing vectors shape: {test_x_vectors.shape}")

print(f'Number of positive and negative reviews in the training set: {train_y.value_counts()}')
print(f'Number of positive and negative reviews in the testing set: {test_y.value_counts()}')

print('-' * 50)

current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f'Training Linear SVM... (Current time: {current_time})')

## Linear SVM
from sklearn import svm
clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)

# for i in range(30):  # Loop through the first 10 test samples
#   print(f"Review {i}: {test_x.iloc[i]}")
#   print(f"Actual Sentiment: {test_y.iloc[i]}")
#   print(f"Predicted Sentiment: {clf_svm.predict(test_x_vectors[i])}")
#   print()

current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f'Training Decision Tree... (Current time: {current_time})')

## Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vectors, train_y)

# for i in range(30):  # Loop through the first 10 test samples
#   print(f"Review {i}: {test_x.iloc[i]}")
#   print(f"Actual Sentiment: {test_y.iloc[i]}")
#   print(f"Predicted Sentiment: {clf_dec.predict(test_x_vectors[i])}")
#   print()

current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f'Training Naive Bayes... (Current time: {current_time})')

## Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf_nb = GaussianNB()
clf_nb.fit(train_x_vectors.toarray(), train_y)

# for i in range(30):  # Loop through the first 10 test samples
#   print(f"Review {i}: {test_x.iloc[i]}")
#   print(f"Actual Sentiment: {test_y.iloc[i]}")
#   print(f"Predicted Sentiment: {clf_nb.predict(test_x_vectors[i].toarray())}")
#   print()

current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f'Training Logistic Regression... (Current time: {current_time})')

from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression()
clf_lr.fit(train_x_vectors, train_y)

# for i in range(30):  # Loop through the first 10 test samples
#   print(f"Review {i}: {test_x.iloc[i]}")
#   print(f"Actual Sentiment: {test_y.iloc[i]}")
#   print(f"Predicted Sentiment: {clf_lr.predict(test_x_vectors[i])}")
#   print()

current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f'Tuning our model with Grid Search... (Current time: {current_time})')

from sklearn.model_selection import GridSearchCV

parameters = {'kernel': ('linear', 'rbf'), 'C': (1, 4, 8, 16)}

svc = svm.SVC()
clv = GridSearchCV(svc, parameters, cv=5)
clf_svm_grid = clv.fit(train_x_vectors, train_y)

current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f'Training complete. (Current time: {current_time})')

## Model Evaluation

### Mean accuracy of the model
print(f"** Mean accuracy of the models: **")
print(f"SVM: {clf_svm.score(test_x_vectors, test_y)}")
print(f"Decision Tree: {clf_dec.score(test_x_vectors, test_y)}")
print(f"Naive Bayes: {clf_nb.score(test_x_vectors.toarray(), test_y)}")
print(f"Logistic Regression: {clf_lr.score(test_x_vectors, test_y)}")
print(f"Grid Search SVM: {clf_svm_grid.score(test_x_vectors, test_y)}")

print('-' * 50)

### F1 Score
from sklearn.metrics import f1_score
print(f"** F1 Score of the models: **")
print(f"SVM: {f1_score(test_y, clf_svm.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE])}")
print(f"Decision Tree: {f1_score(test_y, clf_dec.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE])}")
print(f"Naive Bayes: {f1_score(test_y, clf_nb.predict(test_x_vectors.toarray()), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE])}")
print(f"Logistic Regression: {f1_score(test_y, clf_lr.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE])}")
print(f"Grid Search SVM: {f1_score(test_y, clf_svm_grid.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE])}")

## Saving the models
import pickle

print('Saving svm model...')
with open('sentiment_classifier_svm.pkl', 'wb') as f:
  pickle.dump(clf_svm, f)

print('Saving decision tree model...')
with open('sentiment_classifier_dec.pkl', 'wb') as f:
  pickle.dump(clf_dec, f)

print('Saving naive bayes model...')
with open('sentiment_classifier_nb.pkl', 'wb') as f:
  pickle.dump(clf_nb, f)

print('Saving logistic regression model...')
with open('sentiment_classifier_lr.pkl', 'wb') as f:
  pickle.dump(clf_lr, f)

print('Saving grid search svm model...')
with open('sentiment_classifier_svm_grid.pkl', 'wb') as f:
  pickle.dump(clf_svm_grid, f)

print('Models saved.')

## Loading the models
# print('Loading models...')
# with open('sentiment_classifier_svm.pkl', 'rb') as f:
#   loaded_clf_svm = pickle.load(f)

# with open('sentiment_classifier_dec.pkl', 'rb') as f:
#   loaded_clf_dec = pickle.load(f)

# with open('sentiment_classifier_nb.pkl', 'rb') as f:
#   loaded_clf_nb = pickle.load(f)

# with open('sentiment_classifier_lr.pkl', 'rb') as f:
#   loaded_clf_lr = pickle.load(f)
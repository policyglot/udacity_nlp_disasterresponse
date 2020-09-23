# import libraries
import pandas as pd
import time
from sqlalchemy import create_engine

# global for Random State 
seed = 2020

# Specific Machine Learning Algorithms
from sklearn.ensemble import RandomForestClassifier

#For Word Processing
import re
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
stop_words = stopwords.words("english")

#For the Machine Learning Model
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split 

# For Model Fit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# tokenize the data
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^A-Za-x0-9]', ' ', text)
    tokens = word_tokenizer(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

#Extract data from SQL
engine = create_engine('sqlite:///Messages.db')
df = pd.read_sql("SELECT * FROM Messages", engine)

X = df['message']
Y = df.iloc[:, -36:]

X_train, X_test, y_train, y_test = test_train_split(X, y, test_size=0.33, random_seed=2020)

#Machine Learning Pipeline
pipeline =  Pipeline([
	('vect', CountVectorizer(tokenizer=tokenize)),
	('tfidf', TfidfTransformer(smooth_idf=False)),
	('clf', RandomForestClassifier()) 
	])

pipeline.fit(X_train, y_train)
predicted = pipeline.predict(X_test)


# Complete necessary imports
import pandas as pd
import re
from sqlalchemy import create_engine
from scikitlearn.multioutput import MultipleOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
# To import from the database
from sqlalchemy import create_engine
# To set up NLP pipelines
from nltk import word_tokenizer
# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
# initialize count vectorizer object
vect = CountVectorizer(tokenizer=tokenize)



#Extract data from SQL
engine = create_engine('sqlite:///InsertDatabaseName.db')
df = pd.read_sql("SELECT * FROM population_data", engine)
#X = df[]
#Y = 

# tokenize the data
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^A-Za-x0-9]', ' ', text)
    tokens = word_tokenizer(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

    
X_train, X_test, y_train, y_test = test_train_split(X, y, test_size=0.33, random_seed=2020)

#Machine Learning Pipeline
pipeline =  Pipeline([
	('vect', CountVectorizer()),
	('tfidf', TfidfTransformer()),
	('clf', RandomForestClassifier()) 
	])

pipeline.fit(X_train)
predict = pipeline.predict(X_test)


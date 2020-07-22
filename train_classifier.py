# Complete necessary imports
import pandas as pd
import re
from sqlalchemy import create_engine
from scikitlearn.multioutput import MultipleOutputClassifier
# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])

#Extract data from SQL
engine = create_engine('sqlite:///InsertDatabaseName.db')
df = pd.read_sql("SELECT * FROM population_data", engine)
#X = df[]
#Y = 

# tokenize the data
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^A-Za-x0-9]', ' ', text)
    


#Machine Learning Pipeline
pipeline =  Pipeline([
	('vect', CountVectorizer()),
	('tfidf', TfidfTransformer()),
	('clf', RandomForestClassifier()) 
	])

pipeline.fit(Xtrain)
predict = pipeline.predict(Xtest)
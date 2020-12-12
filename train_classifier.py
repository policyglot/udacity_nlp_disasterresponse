import sys
import pandas as pd
from sqlalchemy import create_engine

# global for Random State
# seed = 2020

# Specific Machine Learning Algorithms
from sklearn.ensemble import RandomForestClassifier

# For Word Processing
import re
import pickle
import nltk

nltk.download(['punkt', 'wordnet',
               'averaged_perceptron_tagger', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords

stop_words = stopwords.words("english")

# For the Machine Learning Model
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# For Model Fit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    engine = create_engine(database_filepath)
    # check if this is correct
    df = pd.read_sql("SELECT * FROM Messages", engine)
    X = df['message']
    y = df.iloc[:, -36:]
    # To fix potential multi-output error for 'related' column
    y['related'].replace(to_replace=2, value=1, inplace=True)
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^A-Za-x0-9]', ' ', text)
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    final_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return final_tokens


def build_model():
    # Set up pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer(smooth_idf=False)),
        ('clf', RandomForestClassifier(random_state=seed,
                                       n_estimators=100,
                                       max_features=8))
    ])
    parameters = {  # 'scaler__with_mean':['True','False'],
        'clf__n_estimators': [100, 300],
        'clf__max_features': [5, 8]}
    cv = GridSearchCV(pipeline, parameters, n_jobs=4)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=category_names)
    for issue in category_names:
        print(classification_report(y_test[issue],
                                    y_pred_df[issue]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
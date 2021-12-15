# udacity_nlp_disasterresponse
My answers to the project on responding to SMS messages sourced from disaster response situations by Figure Eight so as to build ETL &amp; NLP pipelines. The end result is a supervised machine learning algorithm that helps classify messages both in terms of urgency and the specific domain (e.g. water, health, etc.) that the message pertains too. 

The process-data.py takes care of cleaning and splits the original data into the 36 categories for classification. It also exports an sqlite database file.
The train_classifier.py file then handles tasks such as tokenization of the message text before moving to a pipeline for multiclass classification. The machine learning algorithm of choice in this instance is random forests, which we then optimize using grid search. 


# import libraries
import pandas as pd
# load messages dataset
messages = pd.read_csv('messages.csv')
# load categories dataset
categories = pd.read_csv('categories.csv')

# merge datasets
df = pd.merge(right=messages, left=categories, on="id")
# create a dataframe of the 36 individual category columns
categories =  df['categories'].str.split(';', expand=True)
row = categories.iloc[0]
category_colnames = row.apply(lambda x: x[:-2])
categories.columns = category_colnames

for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].apply(lambda x: x[-1])
    
    # convert column from string to numeric
    categories[column] = categories[column].astype(int)

# drop the original categories column from `df`
df.drop('categories', axis=1, inplace=True)
df.head()

df = pd.concat([df,categories], axis=1)

# drop duplicates
df = df.drop_duplicates()

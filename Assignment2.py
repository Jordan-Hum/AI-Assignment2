# -------------------------------------------------------
# Assignment 2
# Written by Jordan Hum 400095876
# For COMP 472 Section IX – Summer 2020
# --------------------------------------------------------

#Library Imports
import pandas as pd
import nltk

#nltk.download()

#read csv files
df = pd.read_csv('hns_2018_2019.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')
year18 = []
title18 = []
for index, row in df.iterrows():
    if '2018' in row['created_at']:
        year18.append(row)
        title18.append(row['title'].lower())

title18 = [nltk.word_tokenize(words) for words in title18]
print(title18)


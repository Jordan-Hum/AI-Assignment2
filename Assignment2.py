# -------------------------------------------------------
# Assignment 2
# Written by Jordan Hum 400095876
# For COMP 472 Section IX – Summer 2020
# --------------------------------------------------------

# Library Imports
import pandas as pd
import nltk


# nltk.download()

# FUNCTIONS
# ----------------------------------------
def sortData():
    for index, row in df.iterrows():
        if '2018' in row['created_at']:
            year18.append(row)
            title18.append(row['title'].lower())
            postType18.append(row['post_type'])

    for i in range(len(postType18)):
        if postType18[i] == 'story':
            for word in nltk.word_tokenize(title18[i]):
                storyList.append(word)
                if word not in allWords:
                    allWords.append(word)
        if postType18[i] == 'ask_hn':
            for word in nltk.word_tokenize(title18[i]):
                askList.append(word)
                if word not in allWords:
                    allWords.append(word)
        if postType18[i] == 'show_hn':
            for word in nltk.word_tokenize(title18[i]):
                showList.append(word)
                if word not in allWords:
                    allWords.append(word)
        if postType18[i] == 'poll':
            for word in nltk.word_tokenize(title18[i]):
                pollList.append(word)
                if word not in allWords:
                    allWords.append(word)


def sortFreq():
    storyFreq = nltk.FreqDist(storyList)
    askFreq = nltk.FreqDist(askList)
    showFreq = nltk.FreqDist(showList)
    pollFreq = nltk.FreqDist(pollList)
    for word in allWords:
        # check frequency of word in story posts
        wordFreq = storyFreq[word]
        storyDict[word] = (wordFreq + 0.5) / len(storyList)
        # check frequency of word in ask_hn posts
        wordFreq = askFreq[word]
        askDict[word] = (wordFreq + 0.5) / len(askList)
        # check frequency of word in show_hn posts
        wordFreq = showFreq[word]
        showDict[word] = (wordFreq + 0.5) / len(showList)
        # check frequency of word in poll posts
        # wordFreq = pollFreq[word]
        # pollDict[word] = (wordFreq + 0.5) / len(pollList)
    file = open('model-2018.txt', 'w', encoding='utf-8')
    wordCounter = 0
    for word in sorted(allWords):
        if word.isalnum():
            wordCounter += 1
            file.write(str(wordCounter) + '  ' + word + '  ' + str(storyFreq[word]) + '  ' + str(storyDict[word]) + '  ' +
                       str(askFreq[word]) + '  ' + str(askDict[word]) +
                       '  ' + str(showFreq[word]) + '  ' + str(showDict[word]) + "\n")
    file.close()


# MAIN
# ----------------------------------------------

# read csv files
df = pd.read_csv('hns_2018_2019.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')

# initialize lists and dictionaries used
year18 = []
title18 = []
postType18 = []
storyList = []
askList = []
showList = []
pollList = []
allWords = []
storyDict = {}
askDict = {}
showDict = {}
pollDict = {}

# function calls
sortData()
sortFreq()

# PRINT CHECKS
# -------------------

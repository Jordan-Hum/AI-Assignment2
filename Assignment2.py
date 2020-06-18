# -------------------------------------------------------
# Assignment 2
# Written by Jordan Hum 40095876
# For COMP 472 Section IX – Summer 2020
# --------------------------------------------------------

# Library Imports
import pandas as pd
import nltk
import math
import matplotlib


# nltk.download()

#-----------------------------------------
# FUNCTIONS
# ----------------------------------------

#Task 1
#-----------------------------------------------------------------------------

#grabs all the rows with the year 2018 and tokenizes the titles into their appropriate lists
#-------------------------------------------------------------------------------------------
def sortTrainingData():
    for index, row in df.iterrows():
        if '2018' in row['created_at']:
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

#Find the frequencies of each word in each post type. Write all used words to appropriate text file, whether used or not
#-----------------------------------------------------------------------------------------------------------------------
def sortFreq():
    storyFreq = nltk.FreqDist(storyList)
    askFreq = nltk.FreqDist(askList)
    showFreq = nltk.FreqDist(showList)
    pollFreq = nltk.FreqDist(pollList)
    for word in allWords:
        # check frequency of word in story posts
        wordFreq = storyFreq[word]
        storyDict[word] = (wordFreq + delta) / (len(storyList) + delta * len(allWords))
        # check frequency of word in ask_hn posts
        wordFreq = askFreq[word]
        askDict[word] = (wordFreq + delta) / (len(askList) + delta * len(allWords))
        # check frequency of word in show_hn posts
        wordFreq = showFreq[word]
        showDict[word] = (wordFreq + delta) / (len(showList) + delta * len(allWords))
        # check frequency of word in poll posts
        wordFreq = pollFreq[word]
        pollDict[word] = (wordFreq + delta) / (len(pollList) + delta * len(allWords))
    modelFile = open('model-2018.txt', 'w', encoding='utf-8')
    vocabFile = open('vocabulary.txt', 'w', encoding='utf-8')
    removeFile = open('remove_word.txt', 'w', encoding='utf-8')
    wordCounter = 0
    for word in sorted(allWords):
        if word.isalnum():
            wordCounter += 1
            modelFile.write(str(wordCounter) + '  ' + word + '  ' + 'story: ' + str(storyFreq[word]) + '  ' + str(storyDict[word]) +
                       '  ' + 'ask:  ' + str(askFreq[word]) + '  ' + str(askDict[word]) +
                       '  ' + 'show:  ' + str(showFreq[word]) + '  ' + str(showDict[word]) + '  ' + 'poll' +
                       str(pollFreq[word]) + '  ' + str(pollDict[word]) + "\n")
            vocabFile.write(word + '\n')
        else:
            removeFile.write(word + "\n")
    modelFile.close()
    vocabFile.close()
    removeFile.close()

#count the amount of post type in each year
#-------------------------------------------
def postProb(post):
    return postType18.count(post)

#----------------------------------------------
#Task 2
#----------------------------------------------
def testingSet():
    for index, row in df.iterrows():
        if '2019' in row['created_at']:
            title19.append(row['title'].lower())
            postType19.append(row['post_type'])

    count = 0
    outputFile = open('baseline-result.txt', 'w', encoding='utf-8')

    for title in title19:
        words = nltk.word_tokenize(title)

        storyScore = math.log10(storyProb)
        showScore = math.log10(showProb)
        askScore = math.log10(askProb)
        pollScore = math.log10(0.000000000005) #TODO fix poll probability

        count += 1

        for word in words:
            if word in allWords:
                storyScore += math.log10(storyDict[word])
                showScore += math.log10(showDict[word])
                askScore += math.log10(askDict[word])
                pollScore += math.log10(pollDict[word])
        if storyScore > showScore and storyScore > askScore and storyScore > pollScore:
            if 'story' == postType19[count-1]:
                outputFile.write(str(count) + '  ' + title + '  story  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count-1] + '  ' + 'Right' + '\n')
            else:
                outputFile.write(str(count) + '  ' + title + '  story  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  ' + 'Wrong' + '\n')
        elif showScore > askScore and showScore > pollScore:
            if 'show_hn' == postType19[count - 1]:
                outputFile.write(str(count) + '  ' + title + '  show  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  ' + 'Right' + '\n')
            else:
                outputFile.write(str(count) + '  ' + title + '  show  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  ' + 'Wrong' + '\n')
        elif askScore > pollScore:
            if 'ask_hn' == postType19[count - 1]:
                outputFile.write(str(count) + '  ' + title + '  ask  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  ' + 'Right' + '\n')
            else:
                outputFile.write(str(count) + '  ' + title + '  ask  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  ' + 'Wrong' + '\n')
        else:
            if 'poll' == postType19[count - 1]:
                outputFile.write(str(count) + '  ' + title + '  poll  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  ' + 'Right' + '\n')
            else:
                outputFile.write(str(count) + '  ' + title + '  poll  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  ' + 'Wrong' + '\n')
    outputFile.close()

#--------------------------------------------
#Experiment 1
#--------------------------------------------
def sortExp1Data():
    for index, row in df.iterrows():
        if '2018' in row['created_at']:
            title18.append(row['title'].lower())
            postType18.append(row['post_type'])

    stopList = []
    stopword = open('stopwords.txt', 'r')
    for word in stopword:
        stopList.append(word.strip())


    for i in range(len(postType18)):
        if postType18[i] == 'story':
            for word in nltk.word_tokenize(title18[i]):
                storyList.append(word)
                if word not in allWords:
                    if word not in stopList:
                        allWords.append(word)
        if postType18[i] == 'ask_hn':
            for word in nltk.word_tokenize(title18[i]):
                askList.append(word)
                if word not in allWords:
                    if word not in stopList:
                        allWords.append(word)
        if postType18[i] == 'show_hn':
            for word in nltk.word_tokenize(title18[i]):
                showList.append(word)
                if word not in allWords:
                    if word not in stopList:
                        allWords.append(word)
        if postType18[i] == 'poll':
            for word in nltk.word_tokenize(title18[i]):
                pollList.append(word)
                if word not in allWords:
                    if word not in stopList:
                        allWords.append(word)

def sortExp1Freq():
    storyFreq = nltk.FreqDist(storyList)
    askFreq = nltk.FreqDist(askList)
    showFreq = nltk.FreqDist(showList)
    pollFreq = nltk.FreqDist(pollList)
    for word in allWords:
        # check frequency of word in story posts
        wordFreq = storyFreq[word]
        storyDict[word] = (wordFreq + delta) / (len(storyList) + delta * len(allWords))
        # check frequency of word in ask_hn posts
        wordFreq = askFreq[word]
        askDict[word] = (wordFreq + delta) / (len(askList) + delta * len(allWords))
        # check frequency of word in show_hn posts
        wordFreq = showFreq[word]
        showDict[word] = (wordFreq + delta) / (len(showList) + delta * len(allWords))
        # check frequency of word in poll posts
        wordFreq = pollFreq[word]
        pollDict[word] = (wordFreq + delta) / (len(pollList) + delta * len(allWords))
    modelFile = open('stopword-model.txt', 'w', encoding='utf-8')
    vocabFile = open('vocabulary.txt', 'w', encoding='utf-8')
    removeFile = open('remove_word.txt', 'w', encoding='utf-8')
    wordCounter = 0
    for word in sorted(allWords):
        if word.isalnum():
            wordCounter += 1
            modelFile.write(
                str(wordCounter) + '  ' + word + '  ' + 'story: ' + str(storyFreq[word]) + '  ' + str(storyDict[word]) +
                '  ' + 'ask:  ' + str(askFreq[word]) + '  ' + str(askDict[word]) +
                '  ' + 'show:  ' + str(showFreq[word]) + '  ' + str(showDict[word]) + '  ' + 'poll' +
                str(pollFreq[word]) + '  ' + str(pollDict[word]) + "\n")
            vocabFile.write(word + '\n')
        else:
            removeFile.write(word + "\n")
    modelFile.close()
    vocabFile.close()
    removeFile.close()

def testingExp1Set():
    for index, row in df.iterrows():
        if '2019' in row['created_at']:
            title19.append(row['title'].lower())
            postType19.append(row['post_type'])

    count = 0
    outputFile = open('stopword-result.txt', 'w', encoding='utf-8')

    for title in title19:
        words = nltk.word_tokenize(title)

        storyScore = math.log10(storyProb)
        showScore = math.log10(showProb)
        askScore = math.log10(askProb)
        pollScore = math.log10(0.000000000005) #TODO fix poll probability

        count += 1

        for word in words:
            if word in allWords:
                storyScore += math.log10(storyDict[word])
                showScore += math.log10(showDict[word])
                askScore += math.log10(askDict[word])
                pollScore += math.log10(pollDict[word])
        if storyScore > showScore and storyScore > askScore and storyScore > pollScore:
            if 'story' == postType19[count-1]:
                outputFile.write(str(count) + '  ' + title + '  story  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count-1] + '  ' + 'Right' + '\n')
            else:
                outputFile.write(str(count) + '  ' + title + '  story  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  ' + 'Wrong' + '\n')
        elif showScore > askScore and showScore > pollScore:
            if 'show_hn' == postType19[count - 1]:
                outputFile.write(str(count) + '  ' + title + '  show  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  ' + 'Right' + '\n')
            else:
                outputFile.write(str(count) + '  ' + title + '  show  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  ' + 'Wrong' + '\n')
        elif askScore > pollScore:
            if 'ask_hn' == postType19[count - 1]:
                outputFile.write(str(count) + '  ' + title + '  ask  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  ' + 'Right' + '\n')
            else:
                outputFile.write(str(count) + '  ' + title + '  ask  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  ' + 'Wrong' + '\n')
        else:
            if 'poll' == postType19[count - 1]:
                outputFile.write(str(count) + '  ' + title + '  poll  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  ' + 'Right' + '\n')
            else:
                outputFile.write(str(count) + '  ' + title + '  poll  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  ' + 'Wrong' + '\n')
    outputFile.close()

#---------------------------------------------
#Experiment 2
#---------------------------------------------
def sortExp2Data():
    for index, row in df.iterrows():
        if '2018' in row['created_at']:
            title18.append(row['title'].lower())
            postType18.append(row['post_type'])

    for i in range(len(postType18)):
        if postType18[i] == 'story':
            for word in nltk.word_tokenize(title18[i]):
                storyList.append(word)
                if word not in allWords:
                    if 2 < len(word) < 9:
                        allWords.append(word)
        if postType18[i] == 'ask_hn':
            for word in nltk.word_tokenize(title18[i]):
                askList.append(word)
                if word not in allWords:
                    if 2 < len(word) < 9:
                        allWords.append(word)
        if postType18[i] == 'show_hn':
            for word in nltk.word_tokenize(title18[i]):
                showList.append(word)
                if word not in allWords:
                    if 2 < len(word) < 9:
                        allWords.append(word)
        if postType18[i] == 'poll':
            for word in nltk.word_tokenize(title18[i]):
                pollList.append(word)
                if word not in allWords:
                    if 2 < len(word) < 9:
                        allWords.append(word)

def sortExp2Freq():
    storyFreq = nltk.FreqDist(storyList)
    askFreq = nltk.FreqDist(askList)
    showFreq = nltk.FreqDist(showList)
    pollFreq = nltk.FreqDist(pollList)
    for word in allWords:
        # check frequency of word in story posts
        wordFreq = storyFreq[word]
        storyDict[word] = (wordFreq + delta) / (len(storyList) + delta * len(allWords))
        # check frequency of word in ask_hn posts
        wordFreq = askFreq[word]
        askDict[word] = (wordFreq + delta) / (len(askList) + delta * len(allWords))
        # check frequency of word in show_hn posts
        wordFreq = showFreq[word]
        showDict[word] = (wordFreq + delta) / (len(showList) + delta * len(allWords))
        # check frequency of word in poll posts
        wordFreq = pollFreq[word]
        pollDict[word] = (wordFreq + delta) / (len(pollList) + delta * len(allWords))
    modelFile = open('wordlength-model.txt', 'w', encoding='utf-8')
    vocabFile = open('vocabulary.txt', 'w', encoding='utf-8')
    removeFile = open('remove_word.txt', 'w', encoding='utf-8')
    wordCounter = 0
    for word in sorted(allWords):
        if word.isalnum():
            wordCounter += 1
            modelFile.write(
                str(wordCounter) + '  ' + word + '  ' + 'story: ' + str(storyFreq[word]) + '  ' + str(storyDict[word]) +
                '  ' + 'ask:  ' + str(askFreq[word]) + '  ' + str(askDict[word]) +
                '  ' + 'show:  ' + str(showFreq[word]) + '  ' + str(showDict[word]) + '  ' + 'poll' +
                str(pollFreq[word]) + '  ' + str(pollDict[word]) + "\n")
            vocabFile.write(word + '\n')
        else:
            removeFile.write(word + "\n")
    modelFile.close()
    vocabFile.close()
    removeFile.close()

def testingExp2Set():
    for index, row in df.iterrows():
        if '2019' in row['created_at']:
            title19.append(row['title'].lower())
            postType19.append(row['post_type'])

    count = 0
    outputFile = open('wordlength-result.txt', 'w', encoding='utf-8')

    for title in title19:
        words = nltk.word_tokenize(title)

        storyScore = math.log10(storyProb)
        showScore = math.log10(showProb)
        askScore = math.log10(askProb)
        pollScore = math.log10(0.000000000005) #TODO fix poll probability

        count += 1

        for word in words:
            if word in allWords:
                storyScore += math.log10(storyDict[word])
                showScore += math.log10(showDict[word])
                askScore += math.log10(askDict[word])
                pollScore += math.log10(pollDict[word])
        if storyScore > showScore and storyScore > askScore and storyScore > pollScore:
            if 'story' == postType19[count-1]:
                outputFile.write(str(count) + '  ' + title + '  story  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count-1] + '  ' + 'Right' + '\n')
            else:
                outputFile.write(str(count) + '  ' + title + '  story  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  ' + 'Wrong' + '\n')
        elif showScore > askScore and showScore > pollScore:
            if 'show_hn' == postType19[count - 1]:
                outputFile.write(str(count) + '  ' + title + '  show  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  ' + 'Right' + '\n')
            else:
                outputFile.write(str(count) + '  ' + title + '  show  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  ' + 'Wrong' + '\n')
        elif askScore > pollScore:
            if 'ask_hn' == postType19[count - 1]:
                outputFile.write(str(count) + '  ' + title + '  ask  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  ' + 'Right' + '\n')
            else:
                outputFile.write(str(count) + '  ' + title + '  ask  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  ' + 'Wrong' + '\n')
        else:
            if 'poll' == postType19[count - 1]:
                outputFile.write(str(count) + '  ' + title + '  poll  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  ' + 'Right' + '\n')
            else:
                outputFile.write(str(count) + '  ' + title + '  poll  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  ' + 'Wrong' + '\n')
    outputFile.close()


# MAIN
# ----------------------------------------------

# read csv files
df = pd.read_csv('hns_2018_2019.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')

# initialize lists and dictionaries used
title18 = []
title19 = []
postType18 = []
postType19 = []
storyList = []
askList = []
showList = []
pollList = []
allWords = []
storyDict = {}
askDict = {}
showDict = {}
pollDict = {}
delta = 0.5


# function calls
#--------------------------------------------

#task 1 and 2, comment out for experiments
#sortTrainingData()
#sortFreq()

# uncomment below for experiment 1
#sortExp1Data()
#sortExp1Freq()

# uncomment below for experiment 2
sortExp2Data()
sortExp2Freq()

storyCount = postProb('story')
showCount = postProb('show_hn')
askCount = postProb('ask_hn')
pollCount = postProb('poll')
totalCount = len(postType18)
storyProb = storyCount / totalCount
showProb = showCount / totalCount
askProb = askCount / totalCount
pollProb = pollCount / totalCount

#task 1 and 2
#testingSet()

#uncomment below for experiment 1
#testingExp1Set()

#uncomment below for experiment 2
testingExp2Set()

# PRINT
# -------------------
print('The program is now finished!!')

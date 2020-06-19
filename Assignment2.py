# -------------------------------------------------------
# Assignment 2
# Written by Jordan Hum 40095876
# For COMP 472 Section IX – Summer 2020
# --------------------------------------------------------

# Library Imports
import pandas as pd
import nltk
import math
import matplotlib.pyplot as plt

#need to download before running the program by uncommenting
#nltk.download()

#-----------------------------------------
# FUNCTIONS
#----------------------------------------

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
            modelFile.write(str(wordCounter) + '  ' + word + '  story: ' + str(storyFreq[word]) + '  ' + str(storyDict[word]) +
                       '  ask:  ' + str(askFreq[word]) + '  ' + str(askDict[word]) +
                       '  show:  ' + str(showFreq[word]) + '  ' + str(showDict[word]) + '  poll  ' +
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
#Task 2- test the trained model with 2019 titles
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
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count-1] + ' Right' + '\n')
            else:
                outputFile.write(str(count) + '  ' + title + '  story  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + ' Wrong' + '\n')
        elif showScore > askScore and showScore > pollScore:
            if 'show_hn' == postType19[count - 1]:
                outputFile.write(str(count) + '  ' + title + '  show  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  Right' + '\n')
            else:
                outputFile.write(str(count) + '  ' + title + '  show  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  Wrong' + '\n')
        elif askScore > pollScore:
            if 'ask_hn' == postType19[count - 1]:
                outputFile.write(str(count) + '  ' + title + '  ask  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  Right' + '\n')
            else:
                outputFile.write(str(count) + '  ' + title + '  ask  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  Wrong' + '\n')
        else:
            if 'poll' == postType19[count - 1]:
                outputFile.write(str(count) + '  ' + title + '  poll  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  Right' + '\n')
            else:
                outputFile.write(str(count) + '  ' + title + '  poll  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + '  Wrong' + '\n')
    outputFile.close()

#--------------------------------------------
#Experiment 1
#--------------------------------------------
#Create lists for every post type while removing the words in stopwords
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

#Find the frequency of all the allowed words and output them to the appropriate text files
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

#Testing the model with the 2019 title set and outputting the result to a text file
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
#Create lists for every post type while removing words with lengths less than 2 and more than 9
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

#Find frequencies of all the remaining words and output them to the appropriate text files
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
                str(wordCounter) + '  ' + word + '  story:  ' + str(storyFreq[word]) + '  ' + str(storyDict[word]) +
                '  ask:  ' + str(askFreq[word]) + '  ' + str(askDict[word]) +
                '  show:  ' + str(showFreq[word]) + '  ' + str(showDict[word]) + '  poll:  ' +
                str(pollFreq[word]) + '  ' + str(pollDict[word]) + "\n")
            vocabFile.write(word + '\n')
        else:
            removeFile.write(word + "\n")
    modelFile.close()
    vocabFile.close()
    removeFile.close()

#Test the model with the 2019 titles and output the results to a text file
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
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count-1] + ' Right' + '\n')
            else:
                outputFile.write(str(count) + '  ' + title + '  story  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + ' Wrong' + '\n')
        elif showScore > askScore and showScore > pollScore:
            if 'show_hn' == postType19[count - 1]:
                outputFile.write(str(count) + '  ' + title + '  show  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + ' Right' + '\n')
            else:
                outputFile.write(str(count) + '  ' + title + '  show  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + ' Wrong' + '\n')
        elif askScore > pollScore:
            if 'ask_hn' == postType19[count - 1]:
                outputFile.write(str(count) + '  ' + title + '  ask  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + ' Right' + '\n')
            else:
                outputFile.write(str(count) + '  ' + title + '  ask  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + ' Wrong' + '\n')
        else:
            if 'poll' == postType19[count - 1]:
                outputFile.write(str(count) + '  ' + title + '  poll  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + ' Right' + '\n')
            else:
                outputFile.write(str(count) + '  ' + title + '  poll  ' + str(storyScore) + '  ' + str(askScore) + '  '
                                 + str(showCount) + '  ' + str(pollScore) + '  ' + postType19[count - 1] + ' Wrong' + '\n')
    outputFile.close()


#---------------------------------------------
#Experiment 3
#---------------------------------------------
#Create lists of every post type while removing the words of low freqency or top percentage
def sortExp3Data():
    for index, row in df.iterrows():
        if '2018' in row['created_at']:
            title18.append(row['title'].lower())
            postType18.append(row['post_type'])

    freqList = []

    for i in range(len(postType18)):

        if postType18[i] == 'story':
            for word in nltk.word_tokenize(title18[i]):
                storyList.append(word)
                freqList.append(word)
                if word not in allWords:
                    allWords.append(word)
        if postType18[i] == 'ask_hn':
            for word in nltk.word_tokenize(title18[i]):
                askList.append(word)
                freqList.append(word)
                if word not in allWords:
                    allWords.append(word)
        if postType18[i] == 'show_hn':
            for word in nltk.word_tokenize(title18[i]):
                showList.append(word)
                freqList.append(word)
                if word not in allWords:
                    allWords.append(word)
        if postType18[i] == 'poll':
            for word in nltk.word_tokenize(title18[i]):
                pollList.append(word)
                freqList.append(word)
                if word not in allWords:
                    allWords.append(word)

    oneList = allWords.copy()
    fiveList = allWords.copy()
    tenList = allWords.copy()
    fifteenList = allWords.copy()
    twentyList = allWords.copy()

    freqWord = nltk.FreqDist(freqList)
    fivepercent = []
    tenpercent = []
    fifteenpercent = []
    twentypercent = []
    twentyfivepercent = []
    for word, freq in freqWord.most_common(int(0.05 * len(allWords))):
        fivepercent.append(word)
    for word, freq in freqWord.most_common(int(0.1 * len(allWords))):
        tenpercent.append(word)
    for word, freq in freqWord.most_common(int(0.15 * len(allWords))):
        fifteenpercent.append(word)
    for word, freq in freqWord.most_common(int(0.2 * len(allWords))):
        twentypercent.append(word)
    for word, freq in freqWord.most_common(int(0.25 * len(allWords))):
        twentyfivepercent.append(word)
    for word in allWords:
        if freqWord[word] == 1:
            oneList.remove(word)
        if freqWord[word] <= 5:
            fiveList.remove(word)
        if freqWord[word] <= 10:
            tenList.remove(word)
        if freqWord[word] <= 15:
            fifteenList.remove(word)
        if freqWord[word] <= 20:
            twentyList.remove(word)
        if word not in fivepercent:
            fivePercentList.append(word)
        if word not in tenpercent:
            tenPercentList.append(word)
        if word not in fifteenpercent:
            fifteenPercentList.append(word)
        if word not in twentypercent:
            twentyPercentList.append(word)
        if word not in twentyfivepercent:
            twentyfivePercentList.append(word)
    return oneList, fiveList, tenList, fifteenList, twentyList

#Find frequency of all remaining words and output them to the appropriate text files
def sortExp3Freq(wordList):
    storyFreq = nltk.FreqDist(storyList)
    askFreq = nltk.FreqDist(askList)
    showFreq = nltk.FreqDist(showList)
    pollFreq = nltk.FreqDist(pollList)
    for word in allWords:
        if word in wordList:
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

#Test the model using the 2019 titles and show the accuracy with plots
def testingExp3Set(wordList):
    title19 = []
    postType19 = []
    for index, row in df.iterrows():
        if '2019' in row['created_at']:
            title19.append(row['title'].lower())
            postType19.append(row['post_type'])

    count = 0
    right = 0
    wrong = 0

    for title in title19:
        words = nltk.word_tokenize(title)

        storyScore = math.log10(storyProb)
        showScore = math.log10(showProb)
        askScore = math.log10(askProb)
        pollScore = math.log10(0.000000000005) #TODO fix poll probability

        count += 1

        for word in words:
            if word in wordList:
                storyScore += math.log10(storyDict[word])
                showScore += math.log10(showDict[word])
                askScore += math.log10(askDict[word])
                pollScore += math.log10(pollDict[word])
        if storyScore > showScore and storyScore > askScore and storyScore > pollScore:
            if 'story' == postType19[count-1]:
                right += 1
            else:
                wrong += 1
        elif showScore > askScore and showScore > pollScore:
            if 'show_hn' == postType19[count - 1]:
                right += 1
            else:
                wrong += 1
        elif askScore > pollScore:
            if 'ask_hn' == postType19[count - 1]:
                right += 1
            else:
                wrong += 1
        else:
            if 'poll' == postType19[count - 1]:
                right += 1
            else:
                wrong += 1
    accuracy = right / len(title19)
    performanceList.append(accuracy)
    vocabList.append(len(wordList))

# MAIN
# ----------------------------------------------

# read csv files and replace spaces with _ for ease of access
df = pd.read_csv('hns_2018_2019.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')

#ask user for input value to run specified task or experiment
task = int(input('Enter a task value (task 1 and 2: 0 | for experiment 1: 1 | for experiment 2: 2 | for experiment 3: 3) :  '))

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
oneList = []
fiveList = []
tenList = []
fifteenList = []
twentyList = []
performanceList = []
vocabList = []
fivePercentList = []
tenPercentList = []
fifteenPercentList = []
twentyPercentList = []
twentyfivePercentList = []
storyDict = {}
askDict = {}
showDict = {}
pollDict = {}
delta = 0.5


# function calls for each task
#--------------------------------------------

if task == 0:
#task 1 and 2
    sortTrainingData()
    sortFreq()

elif task == 1:
#experiment 1
    sortExp1Data()
    sortExp1Freq()

elif task == 2:
#experiment 2
    sortExp2Data()
    sortExp2Freq()

elif task == 3:
#experiment 3
    oneList, fiveList, tenList, fifteenList, twentyList = sortExp3Data()
    #call the function for each cutoff range
    sortExp3Freq(oneList)
    sortExp3Freq(fiveList)
    sortExp3Freq(tenList)
    sortExp3Freq(fifteenList)
    sortExp3Freq(twentyList)
    sortExp3Freq(fivePercentList)
    sortExp3Freq(tenPercentList)
    sortExp3Freq(fifteenPercentList)
    sortExp3Freq(twentyPercentList)
    sortExp3Freq(twentyfivePercentList)

storyCount = postProb('story')
showCount = postProb('show_hn')
askCount = postProb('ask_hn')
pollCount = postProb('poll')
totalCount = len(postType18)
storyProb = storyCount / totalCount
showProb = showCount / totalCount
askProb = askCount / totalCount
pollProb = pollCount / totalCount

if task == 0:
#task 1 and 2
   testingSet()

elif task == 1:
#experiment 1
    testingExp1Set()

elif task == 2:
#experiment 2
    testingExp2Set()

elif task == 3:
#experiment 3
    #call the function each time for each cutoff range
    testingExp3Set(oneList)
    testingExp3Set(fiveList)
    testingExp3Set(tenList)
    testingExp3Set(fifteenList)
    testingExp3Set(twentyList)
    plt.figure('1')
    plt.plot(vocabList, performanceList)
    plt.title('removed lower frequency words')
    plt.xlabel('number of words')
    plt.ylabel('accuracy')

    #reset the lists for different plots
    vocabList = []
    performanceList = []

    #call the function each time for each cutoff range
    testingExp3Set(fivePercentList)
    testingExp3Set(tenPercentList)
    testingExp3Set(fifteenPercentList)
    testingExp3Set(twentyPercentList)
    testingExp3Set(twentyfivePercentList)
    plt.figure('2')
    plt.plot(vocabList, performanceList)
    plt.title('removed higher percentage words')
    plt.xlabel('number of words')
    plt.ylabel('accuracy')
    plt.show()

# EXIT PRINT
# -------------------
print('The program is now finished!!')

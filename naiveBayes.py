# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 23:48:13 2018

@author: lakshay
"""

import sys
from os import listdir
from os.path import join, isfile
import codecs
import re
import math

if(len(sys.argv) != 4):      # also includes sys.argv[0] i.e. file of name
    sys.exit("Please give the required amount of arguments - \
              <path-to-training-dataset-up-till-folder-name-that-contains-ham-and-spam-folders> \
              <path-to-test-dataset-up-till-folder-name-that-contains-ham-and-spam-folders> \
              <path-to-stop-words-text-file-including-file-name> \n\n \
               Please refer to the Read Me document for further details.")
#if (False):
#    pass
else:
    pathToTrainingDataset = sys.argv[1]
    pathToTestDataset = sys.argv[2]
    pathToStopWordsTextFile = sys.argv[3]
#    pathToTrainingDataset = r'C:\Users\laksh\Downloads\ML\Assignment 2\Submission\hw2_train\train'
#    pathToTestDataset = r'C:\Users\laksh\Downloads\ML\Assignment 2\Submission\hw2_test\test'
#    pathToStopWordsTextFile = r'C:\Users\laksh\Downloads\ML\Assignment 2\Submission\stopWords.txt'
    # python naiveBayes.py "C:\Users\laksh\Downloads\ML\Assignment 2\Submission\hw2_train\train" "C:\Users\laksh\Downloads\ML\Assignment 2\Submission\hw2_test\test" "C:\Users\laksh\Downloads\ML\Assignment 2\Submission\stopWords.txt"

# Creating list of 174 stopWords
with open(pathToStopWordsTextFile, 'r') as stopWordsTextFile:
    listOfStopWords = stopWordsTextFile.read().split('\n')

# Defining classification class structure
class ClassificationClass:
    def __init__(self, passedClassName = None, 
                 passedListOfFiles = None, passedNumberOfFiles = None, 
                 passedDictionaryOfWords = dict(), passedNumberOfWords = None, 
                 passedPriorProbability = None, passedConditionalPriorDictionary = None, 
                 passedPathToFiles = None, passedCountOfWords = None,
                 passedWCafterPseudoCounter = None, passedReqProbLHS = None):
        self.className = passedClassName
        self.listOfFiles = passedListOfFiles
        self.numberOfFiles = passedNumberOfFiles
        self.dictionaryOfWords = passedDictionaryOfWords
        self.numberOfWords = passedNumberOfWords
        self.priorProbability = passedPriorProbability
        self.conditionalPriorDictionary = passedConditionalPriorDictionary
        self.pathToFiles = passedPathToFiles
        self.wordCountAfterPseudoCounter = passedWCafterPseudoCounter
        self.requiredProbabilityLHS = passedReqProbLHS






# WITH STOP WORDS
print("*** Executing Naive Bayes with Stop Words intact in files...***")
        
# Creating list of classification classes
listOfClassificationClasses = []
hamClass = ClassificationClass(passedClassName = 'Ham',
            passedListOfFiles = [file for file in listdir(pathToTrainingDataset + '\ham') if 
                              isfile(join(pathToTrainingDataset + '\ham', file))], 
            passedPathToFiles = pathToTrainingDataset + '\ham\\')
spamClass = ClassificationClass(passedClassName = 'Spam',
            passedListOfFiles = [file for file in listdir(pathToTrainingDataset + '\spam') if 
                              isfile(join(pathToTrainingDataset + '\spam', file))], 
            passedPathToFiles = pathToTrainingDataset + '\spam\\')

listOfClassificationClasses.append(hamClass)
listOfClassificationClasses.append(spamClass)

# Populating dictionaryOfWords and numberOfWords for each classificationClass
def wordsDictAndCount(listOfFiles, pathToFile):
    countOfWords=0
    dictionaryOfWords = dict()

    for fileName in listOfFiles:
        fileHandler = codecs.open(pathToFile + fileName,'rU','latin-1') # codecs handles -> UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d in position 1651: character maps to <undefined>
        extractedWords = [word.lower() for word in re.findall('[A-Za-z0-9\']+', fileHandler.read())]
        fileHandler.close()
        countOfWords += len(extractedWords)
        for extractedWord in extractedWords:
            if extractedWord not in dictionaryOfWords:
                dictionaryOfWords[extractedWord] = 1
            else:
                dictionaryOfWords[extractedWord] += 1
    return dictionaryOfWords, countOfWords

for classificationClass in listOfClassificationClasses:
        classificationClass.dictionaryOfWords, classificationClass.numberOfWords = wordsDictAndCount(classificationClass.listOfFiles, classificationClass.pathToFiles)
        
#  Calculation of dictionaryOfWords prior to removing stopWords and after removing them
#for classificationClass in listOfClassificationClasses:
#    wordCountBeforeRemovingSW = len(classificationClass.dictionaryOfWords)
#    for stopWord in listOfStopWords:
#        if stopWord in classificationClass.dictionaryOfWords:
#            del classificationClass.dictionaryOfWords[stopWord]
#    classificationClass.numberOfWords = len(classificationClass.dictionaryOfWords)
#    wordCountAfterRemovingSW = classificationClass.numberOfWords
#    print("Number of stop words removed from {}: {}".format(str(classificationClass.className), 
#                                                                  wordCountBeforeRemovingSW - wordCountAfterRemovingSW))
 
# Populating overall dictionaryOfWords
overallDictionaryOfWords = dict()

for classificationClass in listOfClassificationClasses:
    # Combining all dictionaries of classification classes
    for word in classificationClass.dictionaryOfWords:
        if (word not in overallDictionaryOfWords):
            overallDictionaryOfWords[word] = classificationClass.dictionaryOfWords[word]
        else:
            overallDictionaryOfWords[word] += classificationClass.dictionaryOfWords[word]
    
# Initialize missing words in classificationClasses to 0 by synchronizing
# overallDictionaryOfWords with individual dictionaryOfWords
for classificationClass in listOfClassificationClasses:
    for word in overallDictionaryOfWords:
        if word not in classificationClass.dictionaryOfWords:
            classificationClass.dictionaryOfWords[word] = 0

# Setting updated numberOfWords for individual classificationClasses, 
# setting numberOfFiles for individual classificationClasses,
# calculating overallNumberOfFiles   
overallNumberOfFiles = 0
for classificationClass in listOfClassificationClasses:
    classificationClass.numberOfWords = len(classificationClass.dictionaryOfWords)
    classificationClass.numberOfFiles = len(classificationClass.listOfFiles)
    overallNumberOfFiles += classificationClass.numberOfFiles

## CLASSIFICATION ON TRAINING DATASET
# Calculating prior probability of classificationClasses
# Adding laplace smoothing / pseudo counter on conditional probabilities
for classificationClass in listOfClassificationClasses:
    classificationClass.priorProbability = classificationClass.numberOfFiles / overallNumberOfFiles
    classificationClass.conditionalPriorDictionary = dict()
    classificationClass.wordCountAfterPseudoCounter = 0
    for word in classificationClass.dictionaryOfWords:
        classificationClass.wordCountAfterPseudoCounter += (classificationClass.dictionaryOfWords[word] + 1)
    for word in classificationClass.dictionaryOfWords:
        classificationClass.conditionalPriorDictionary[word] = (classificationClass.dictionaryOfWords[word] + 1) / classificationClass.wordCountAfterPseudoCounter

## TESTING ON TEST DATASET
def predictClassificationLabel(fileHandler, listOfClassificationClasses):
    extractedWords = [word.lower() for word in re.findall('[A-Za-z0-9\']+', fileHandler.read())]
    
    for classificationClass in listOfClassificationClasses:
        classificationClass.requiredProbabilityLHS = math.log(classificationClass.priorProbability)
        for word in extractedWords:
            if word in classificationClass.conditionalPriorDictionary:
                classificationClass.requiredProbabilityLHS += math.log(classificationClass.conditionalPriorDictionary[word])
    
    highestLHSprobability = listOfClassificationClasses[1].requiredProbabilityLHS   # initialzing to any arbitrary values
    classWithHighestLHSprobability = listOfClassificationClasses[1].className.lower()

    for classificationClass in listOfClassificationClasses:
        if highestLHSprobability < classificationClass.requiredProbabilityLHS:
            highestLHSprobability = classificationClass.requiredProbabilityLHS
            classWithHighestLHSprobability = classificationClass.className.lower()
            
    return classWithHighestLHSprobability        
        
def calculateAccuracy(listOfFiles, pathToFiles, knownClassificationLabel, listOfClassificationClasses):
    totalNumberOfFiles = len(listOfFiles)
    accuracy = 0
    predictedLabel = ""
    numberOfFilesClassifiedAsHam, numberOfFilesClassifiedAsSpam = 0, 0
    
    for fileName in listOfFiles:
        fileHandler = codecs.open(pathToFiles + fileName,'rU','latin-1')
        predictedClassificationLabel = predictClassificationLabel(fileHandler, listOfClassificationClasses)
        if predictedClassificationLabel == 'ham':
            numberOfFilesClassifiedAsHam += 1
        if predictedClassificationLabel == 'spam':
            numberOfFilesClassifiedAsSpam += 1
    print("- Total number of files tested: {}".format(totalNumberOfFiles))
    print("- Number of files classified as Ham: {}".format(numberOfFilesClassifiedAsHam))
    print("- Number of files classified as Spam: {}".format(numberOfFilesClassifiedAsSpam))

    if (knownClassificationLabel == 'ham'):
        predictedLabel = 'Ham'
        accuracy = round((numberOfFilesClassifiedAsHam * 100) / (totalNumberOfFiles), 2)
    elif (knownClassificationLabel == 'spam'):
        predictedLabel = 'Spam'
        accuracy = round((numberOfFilesClassifiedAsSpam * 100) / (totalNumberOfFiles), 2)

    print('- Accuracy on {} files in Training Dataset: {}%\n'.format(predictedLabel, accuracy))

listOfFilesForHam = [file for file in listdir(pathToTestDataset + '\ham') if 
                              isfile(join(pathToTestDataset + '\ham', file))]
listOfFilesForSpam = [file for file in listdir(pathToTestDataset + '\spam') if 
                              isfile(join(pathToTestDataset + '\spam', file))]

print("*Calculating accuracy over HAM files in testing dataset...")
calculateAccuracy(listOfFilesForHam, pathToTestDataset + '\ham\\', 'ham', listOfClassificationClasses)
print("*Calculating accuracy over SPAM files in testing dataset...")
calculateAccuracy(listOfFilesForSpam, pathToTestDataset + '\spam\\', 'spam', listOfClassificationClasses)























# AFTER REMOVING STOP WORDS
print("*** Executing Naive Bayes after removing Stop Words from files...***")

# Creating list of classification classes
listOfClassificationClasses = []
hamClass = ClassificationClass(passedClassName = 'Ham',
            passedListOfFiles = [file for file in listdir(pathToTrainingDataset + '\ham') if 
                              isfile(join(pathToTrainingDataset + '\ham', file))], 
            passedPathToFiles = pathToTrainingDataset + '\ham\\')
spamClass = ClassificationClass(passedClassName = 'Spam',
            passedListOfFiles = [file for file in listdir(pathToTrainingDataset + '\spam') if 
                              isfile(join(pathToTrainingDataset + '\spam', file))], 
            passedPathToFiles = pathToTrainingDataset + '\spam\\')

listOfClassificationClasses.append(hamClass)
listOfClassificationClasses.append(spamClass)

# Populating dictionaryOfWords and numberOfWords for each classification class
def wordsDictAndCount(listOfFiles, pathToFile):
    countOfWords=0
    dictionaryOfWords = dict()

    for fileName in listOfFiles:
        fileHandler = codecs.open(pathToFile + fileName,'rU','latin-1') # codecs handles -> UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d in position 1651: character maps to <undefined>
        extractedWords = [word.lower() for word in re.findall('[A-Za-z0-9\']+', fileHandler.read())]
        countOfWords += len(extractedWords)
        for extractedWord in extractedWords:
            if (extractedWord not in dictionaryOfWords):
                dictionaryOfWords[extractedWord] = 1
            else:
                dictionaryOfWords[extractedWord] += 1
    return dictionaryOfWords, countOfWords

for classificationClass in listOfClassificationClasses:
        classificationClass.dictionaryOfWords, classificationClass.numberOfWords = wordsDictAndCount(classificationClass.listOfFiles, classificationClass.pathToFiles)
        
#  Calculation of dictionaryOfWords prior to removing stopWords and after removing them
for classificationClass in listOfClassificationClasses:
    wordCountBeforeRemovingSW = len(classificationClass.dictionaryOfWords)
    for stopWord in listOfStopWords:
        if stopWord in classificationClass.dictionaryOfWords:
            del classificationClass.dictionaryOfWords[stopWord]
    classificationClass.numberOfWords = len(classificationClass.dictionaryOfWords)
    wordCountAfterRemovingSW = classificationClass.numberOfWords
    print("Number of stop words removed from {}: {}".format(str(classificationClass.className), 
              wordCountBeforeRemovingSW - wordCountAfterRemovingSW))
 
# Populating overall dictionaryOfWords
overallDictionaryOfWords = dict()

for classificationClass in listOfClassificationClasses:
    # Combining all dictionaries of classification classes
    for word in classificationClass.dictionaryOfWords:
        if (word not in overallDictionaryOfWords):
            overallDictionaryOfWords[word] = classificationClass.dictionaryOfWords[word]
        else:
            overallDictionaryOfWords[word] += classificationClass.dictionaryOfWords[word]
    
# Initialize missing words in classificationClasses to 0 by synchronizing
# overallDictionaryOfWords with individual dictionaryOfWords
for classificationClass in listOfClassificationClasses:
    for word in overallDictionaryOfWords:
        if word not in classificationClass.dictionaryOfWords:
            classificationClass.dictionaryOfWords[word] = 0

# Setting updated numberOfWords for individual classificationClasses, 
# setting numberOfFiles for individual classificationClasses,
# calculating overallNumberOfFiles   
overallNumberOfFiles = 0
for classificationClass in listOfClassificationClasses:
    classificationClass.numberOfWords = len(classificationClass.dictionaryOfWords)
    classificationClass.numberOfFiles = len(classificationClass.listOfFiles)
    overallNumberOfFiles += classificationClass.numberOfFiles

## CLASSIFICATION ON TRAINING DATASET
# Calculating prior probability of classificationClasses
# Adding laplace smoothing / pseudo counter on conditional probabilities
for classificationClass in listOfClassificationClasses:
    classificationClass.priorProbability = classificationClass.numberOfFiles / overallNumberOfFiles
    classificationClass.conditionalPriorDictionary = dict()
    classificationClass.wordCountAfterPseudoCounter = 0
    for word in classificationClass.dictionaryOfWords:
        classificationClass.wordCountAfterPseudoCounter += (classificationClass.dictionaryOfWords[word] + 1)
    for word in classificationClass.dictionaryOfWords:
        classificationClass.conditionalPriorDictionary[word] = (classificationClass.dictionaryOfWords[word] + 1) / classificationClass.wordCountAfterPseudoCounter

## TESTING ON TEST DATASET
def predictClassificationLabel(fileHandler, listOfClassificationClasses):
    extractedWords = [word.lower() for word in re.findall('[A-Za-z0-9\']+', fileHandler.read())]
    
    for classificationClass in listOfClassificationClasses:
        classificationClass.requiredProbabilityLHS = math.log(classificationClass.priorProbability)
        for word in extractedWords:
            if word in classificationClass.conditionalPriorDictionary:
                classificationClass.requiredProbabilityLHS += math.log(classificationClass.conditionalPriorDictionary[word])
    
    highestLHSprobability = listOfClassificationClasses[1].requiredProbabilityLHS   # initialzing to any arbitrary values
    classWithHighestLHSprobability = listOfClassificationClasses[1].className.lower()

    for classificationClass in listOfClassificationClasses:
        if highestLHSprobability < classificationClass.requiredProbabilityLHS:
            highestLHSprobability = classificationClass.requiredProbabilityLHS
            classWithHighestLHSprobability = classificationClass.className.lower()
            
    return classWithHighestLHSprobability        
        
def calculateAccuracy(listOfFiles, pathToFiles, knownClassificationLabel, listOfClassificationClasses):
    totalNumberOfFiles = len(listOfFiles)
    accuracy = 0
    predictedLabel = ""
    numberOfFilesClassifiedAsHam, numberOfFilesClassifiedAsSpam = 0, 0
    
    for fileName in listOfFiles:
        fileHandler = codecs.open(pathToFiles + fileName,'rU','latin-1')
        predictedClassificationLabel = predictClassificationLabel(fileHandler, listOfClassificationClasses)
        if predictedClassificationLabel == 'ham':
            numberOfFilesClassifiedAsHam += 1
        if predictedClassificationLabel == 'spam':
            numberOfFilesClassifiedAsSpam += 1
    print("- Total number of files tested: {}".format(totalNumberOfFiles))
    print("- Number of files classified as Ham: {}".format(numberOfFilesClassifiedAsHam))
    print("- Number of files classified as Spam: {}".format(numberOfFilesClassifiedAsSpam))

    if (knownClassificationLabel == 'ham'):
        predictedLabel = 'Ham'
        accuracy = round((numberOfFilesClassifiedAsHam * 100) / (totalNumberOfFiles), 2)
    elif (knownClassificationLabel == 'spam'):
        predictedLabel = 'Spam'
        accuracy = round((numberOfFilesClassifiedAsSpam * 100) / (totalNumberOfFiles), 2)

    print('- Accuracy on {} files in Training Dataset: {}%\n'.format(predictedLabel, accuracy))

listOfFilesForHam = [file for file in listdir(pathToTestDataset + '\ham') if 
                              isfile(join(pathToTestDataset + '\ham', file))]
listOfFilesForSpam = [file for file in listdir(pathToTestDataset + '\spam') if 
                              isfile(join(pathToTestDataset + '\spam', file))]

print("*Calculating accuracy over HAM files in testing dataset...")
calculateAccuracy(listOfFilesForHam, pathToTestDataset + '\ham\\', 'ham', listOfClassificationClasses)
print("*Calculating accuracy over SPAM files in testing dataset...")
calculateAccuracy(listOfFilesForSpam, pathToTestDataset + '\spam\\', 'spam', listOfClassificationClasses)

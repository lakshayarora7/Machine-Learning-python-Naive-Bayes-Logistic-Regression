# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 12:40:07 2018

@author: laksh
"""

## IMPORTS
import numpy as np
import time
import sys
from os import listdir
from os.path import isfile, join
import codecs
import re


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
#    pathToTrainingDataset = r'C:\Users\laksh\Downloads\ML\Assignment 2\Submission\hw2_train\train\'
#    pathToTestDataset = r'C:\Users\laksh\Downloads\ML\Assignment 2\Submission\hw2_test\test\'
#    pathToStopWordsTextFile = r'C:\Users\laksh\Downloads\ML\Assignment 2\Submission\stopWords.txt'
#   spyder command line arguments "C:\Users\laksh\Downloads\ML\Assignment 2\Submission\hw2_train\train\\" "C:\Users\laksh\Downloads\ML\Assignment 2\Submission\hw2_test\test\\" "C:\Users\laksh\Downloads\ML\Assignment 2\Submission\stopWords.txt"
# NON-INPUT PARAMETERS
regularizationLambda = 3
learningRateAlpha = 0.001
numberOfIterations = 100

print("VALUES OF PARAMETERS: ")
print("Regularization Lambda: {}".format(str(regularizationLambda)))
print("Learning Rate Alpha: {}".format(str(learningRateAlpha)))
print("Number of Iterations: {}\n".format(str(numberOfIterations)))

# CREATING LIST OF 174 STOPWORDS
with open(pathToStopWordsTextFile, 'r') as stopWordsTextFile:
    listOfStopWords = stopWordsTextFile.read().split('\n')

# DEFINITIONS of ClassificationClass and featureMatrix
class ClassificationClass:
    def __init__(self, className = None, classValue = None, listOfFiles = list(), 
                 dictionaryOfWords = dict(), priorProbability = 0, 
                 conditionalPriorDictionary = dict(), pathToFiles = None, numberOfWords = 0,
                 countOfWords = 0):
        self.className = className            
        self.classValue = classValue
        self.listOfFiles = listOfFiles     
        self.dictionaryOfWords = dictionaryOfWords
        self.prior = priorProbability
        self.condPriorDict = conditionalPriorDictionary            
        self.pathToFiles = pathToFiles
        self.numberOfWords = numberOfWords

class featureMatrix:
    def __init__(self, matrix = None, numberOfColumns = None, classificationClass = None,
                 wordsList = None, countOfFiles = None, valueOfSigmoidFunction = None, weights = None):
        self.matrix  = matrix                       # fileNames for rows, words for columns
        self.numberOfColumns = numberOfColumns
        self.classificationClass = classificationClass
        self.wordsList= wordsList
        self.countOfFiles= countOfFiles
        self.valueOfSigmoidFunction = valueOfSigmoidFunction
        self.weights = weights
    
def createFeatureMatrixFrom(pathToTrainingDataset, stopWordsRemoved):
    # Creating listOfClassificationClasses
    listOfClassificationClasses = []
    hamClass = ClassificationClass(className = 'ham', classValue = 0,
                listOfFiles = [file for file in listdir(pathToTrainingDataset + 'ham') if 
                                  isfile(join(pathToTrainingDataset + 'ham', file))], 
                pathToFiles = pathToTrainingDataset + 'ham\\')
    spamClass = ClassificationClass(className = 'spam', classValue = 1,
                listOfFiles = [file for file in listdir(pathToTrainingDataset + 'spam') if 
                                  isfile(join(pathToTrainingDataset + 'spam', file))],
                               pathToFiles = pathToTrainingDataset + 'spam\\')
    listOfClassificationClasses.append(hamClass)
    listOfClassificationClasses.append(spamClass)
    
    # Populating dictionaryOfWords and numberOfWords for each classificationClass
    # doing this to get overallDictionaryOfWords, in order to get numberOfColumns in matrix
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

    # Calculating overallNumberOfFiles: to determine numberOfRows in matrix
    overallNumberOfFiles = 0
    for classificationClass in listOfClassificationClasses:
        overallNumberOfFiles += len(classificationClass.listOfFiles)
    
    # Populating overall dictionaryOfWords: to determine numberOfColumns in matrix
    overallDictionaryOfWords = dict()
    for classificationClass in listOfClassificationClasses:
        # Combining all dictionaries of classification classes
        for word in classificationClass.dictionaryOfWords:
            if (word not in overallDictionaryOfWords):
                overallDictionaryOfWords[word] = classificationClass.dictionaryOfWords[word]
            else:
                overallDictionaryOfWords[word] += classificationClass.dictionaryOfWords[word]

    numberOfColumns = len(overallDictionaryOfWords)
    numberOfRows = overallNumberOfFiles                           # 463
    matrix = [[0 for columnNumber in range(numberOfColumns)] for rowNumber in range(numberOfRows)]
    targetFunction = [0 for rowNumber in range(numberOfRows)]
    wordsList = []     # for representing each word in numerical form (using indexes)
    countOfFiles = 0   # for populating each row one by one
    
    # Populating Matrix
    for classificationClass in listOfClassificationClasses:
        for fileName in classificationClass.listOfFiles:
            fileHandler = codecs.open(classificationClass.pathToFiles + fileName,'rU','latin-1')
            extractedWords = [word.lower() for word in re.findall('[A-Za-z0-9\']+', fileHandler.read())]
            fileHandler.close()
            
            dictionaryOfWords = {}
            for extractedWord in extractedWords:
                if extractedWord not in dictionaryOfWords:
                    dictionaryOfWords[extractedWord] = 1
                else:
                    dictionaryOfWords[extractedWord] += 1
                
            # WITHOUT STOP WORDS i.e. stopWordsRemoved = True
            if stopWordsRemoved:
                for stopWord in listOfStopWords:
                    if stopWord in dictionaryOfWords:
                        del dictionaryOfWords[stopWord]
            
            for word in dictionaryOfWords:
                if word in wordsList:
                    index = wordsList.index(word)
                    matrix[countOfFiles][index] = dictionaryOfWords[word]
                elif word not in wordsList:
                    wordsList.append(word)
                    index = wordsList.index(word)
                    matrix[countOfFiles][index] = dictionaryOfWords[word]

            targetFunction[countOfFiles] = classificationClass.classValue
            countOfFiles += 1
            
    #print("Number of files: {}".format(countOfFiles)) # 463
    return featureMatrix(matrix = matrix, numberOfColumns = numberOfColumns, 
                         classificationClass = targetFunction, wordsList = wordsList, 
                         countOfFiles = countOfFiles, weights = [0.0 for eachWord in range(len(wordsList))],
                         valueOfSigmoidFunction = [0 for index in range(countOfFiles)])

def populateSigmoid(featureMatrix):
    for fileNumber in range(featureMatrix.countOfFiles):
        summationOfWeightAndFrequency = 1.0
        valueOfSigmoidFunction = 0.0

        #summation of weightOfWord*frequencyOfWord
        for wordNumber in range(len(featureMatrix.wordsList)):
            #print(featureMatrix.weights[wordNumber], end = " ")     # 0.0 for all words in 1st file
            summationOfWeightAndFrequency += featureMatrix.weights[wordNumber]  * featureMatrix.matrix[fileNumber][wordNumber]
        # sigmoid function
        try:
            valueOfSigmoidFunction = ( 1 / (1 + np.exp(-summationOfWeightAndFrequency)))
        except Exception:
            print("Exception while executing sigmoid function. Denominator: {}; Summation: {}".format(str((1 + np.exp(-summationOfWeightAndFrequency))), str(summationOfWeightAndFrequency)))
        featureMatrix.valueOfSigmoidFunction[fileNumber] = valueOfSigmoidFunction
    return featureMatrix

def calculateWeightUpdate(featureMatrix):
    for weightNumber in range(len(featureMatrix.weights)):
        termOne = 0                 # bias term
        for fileNumber in range(featureMatrix.countOfFiles):
            frequencyOfWord = featureMatrix.matrix[fileNumber][weightNumber]
            valueForTargetFunction = featureMatrix.classificationClass[fileNumber]
            valueOfSigmoidFunction = featureMatrix.valueOfSigmoidFunction[fileNumber]

            termOne += frequencyOfWord * (valueForTargetFunction - valueOfSigmoidFunction)
        weightValueBeforeUpdate = featureMatrix.weights[weightNumber]
        # weight update formula as given on slide 26 of 'Logistic Regression.pdf;
        featureMatrix.weights[weightNumber] += ((termOne * learningRateAlpha) - (learningRateAlpha * regularizationLambda * weightValueBeforeUpdate))
    return featureMatrix

def trainingFunction(featureMatrix):
    print("Iterations: ", end = "")
    for iterationNumber in range(1, numberOfIterations + 1):
        print(iterationNumber, end = " ")
        featureMatrix = populateSigmoid(featureMatrix)
        featureMatrix = calculateWeightUpdate(featureMatrix)
    return featureMatrix

def classificationFunction(trainingFeatureMatrix, pathToTestDataset, stopWordsRemoved):
    testFeatureMatrix = createFeatureMatrixFrom(pathToTestDataset, stopWordsRemoved)

    numberOfHamsCorrectlyClassified, numberOfHamsIncorrectlyClassified = 0, 0
    numberOfSpamsCorrectlyClassified, numberOfSpamIncorrectlyClassified = 0, 0

    for fileNumber in range(testFeatureMatrix.countOfFiles):
        summationOfWeightAndFrequency = 1.0
        valueOfSigmoidFunction = 0.0
        # calculate summationOfWeightAndFrequency using weight in training matrix 
        # and frequency in test matrix
        for wordIndex in range(len(testFeatureMatrix.wordsList)-1):
            word =  testFeatureMatrix.wordsList[wordIndex]
            if word in trainingFeatureMatrix.wordsList:
                indexOfWord = trainingFeatureMatrix.wordsList.index(word)
                weightInTrainingMatrix = trainingFeatureMatrix.weights[indexOfWord]
                frequencyOfWordInTestMatrix = testFeatureMatrix.matrix[fileNumber][wordIndex]
                summationOfWeightAndFrequency +=  weightInTrainingMatrix * frequencyOfWordInTestMatrix
        # sigmoid function
        try:
            valueOfSigmoidFunction = ( 1 / (1 + np.exp(-summationOfWeightAndFrequency)))
        except Exception:
            print("Exception while executing sigmoid function. Denominator: {}; Summation: {}".format(str((1 + np.exp(-summationOfWeightAndFrequency))), str(summationOfWeightAndFrequency)))
        if(testFeatureMatrix.classificationClass[fileNumber] == 0):
            if valueOfSigmoidFunction < 0.5:
                numberOfHamsCorrectlyClassified += 1.0
            else:
                numberOfHamsIncorrectlyClassified += 1.0
        else:
            if valueOfSigmoidFunction >= 0.5:
                numberOfSpamsCorrectlyClassified += 1.0
            else:
                numberOfSpamIncorrectlyClassified += 1.0
    
    accuracyOnHamFiles = round((numberOfHamsCorrectlyClassified * 100)/(numberOfHamsCorrectlyClassified + numberOfHamsIncorrectlyClassified) , 2)            
    accuracyOnSpamFiles = round((numberOfSpamsCorrectlyClassified * 100) / (numberOfSpamsCorrectlyClassified + numberOfSpamIncorrectlyClassified), 2)
    print("Accuracy over Ham files: {}%".format(str(accuracyOnHamFiles)))
    print("Accuracy over Spam files: {}%".format(str(accuracyOnSpamFiles)))

def main():
    stopWordsRemoved = False
    print("***Executing Logistic Regression with Stop Words intact in files...***")
    timeOne = time.time()
    trainingFeatureMatrix = createFeatureMatrixFrom(pathToTrainingDataset, stopWordsRemoved)
    print("Training Feature Matrix constructed in {} seconds".format(round(time.time() - timeOne, 2)))
    
    # Training on Training dataset
    timeTwo = time.time()
    print("\nStarting training on training dataset...")
    trainingFeatureMatrix = trainingFunction(trainingFeatureMatrix)
    print("\nTraining completed on training dataset, took {} seconds.".format(round(time.time() - timeTwo, 2)))    
    # Classification of test dataset
    timeThree = time.time()
    print("\nStarting classification of test dataset...")
    classificationFunction(trainingFeatureMatrix, pathToTestDataset, stopWordsRemoved)
    print("Classification of test dataset completed, took {} seconds.".format(round(time.time() - timeThree, 2)))
    print("Total time taken for Logistic Regression with Stop Words intact in files: {} seconds.".format(round(time.time() - timeOne, 2)))






    stopWordsRemoved = True
    print("\n\n*** Executing Logistic Regression after removing Stop Words from files...***")
    timeFour = time.time()
    testFeatureMatrix = createFeatureMatrixFrom(pathToTrainingDataset, stopWordsRemoved)
    print("\nTraining Feature Matrix constructed in {} seconds".format(round(time.time() - timeFour, 2)))
    
    # Training on Training dataset
    timeFive = time.time()
    print("\nStarting training on training dataset...")
    testFeatureMatrix = trainingFunction(testFeatureMatrix)
    print("\nTraining completed on training dataset, took {} seconds.".format(round(time.time() - timeFive, 2))) 
    # Classificiation of test dataset
    timeSix = time.time()
    print("\nStarting classification of test dataset...")
    classificationFunction(testFeatureMatrix,pathToTestDataset, stopWordsRemoved)
    print("Classification of test dataset completed, took {} seconds.".format(round(time.time() - timeSix, 2)))
    print("Total time taken for Logistic Regression after removing Stop Words: {} seconds.".format(round(time.time() - timeFour, 2)))

if __name__ == "__main__": main()

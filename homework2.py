#Tacy Bechtel


import random
import numpy
import csv
from timeit import default_timer as timer


start = timer() #used to figure out how long the trial took
trainingData = list(list())   #variable to hold training data
testData = list(list())   #variable to hold test data
learningRate = 0.01   #changed manually for each trial to 1.0, 0.1, and 0.01 respectively
hiddenUnits = 10    #hidden units can be changed to determine peak performance

#read in the preprocessed training data. Add an extra row, all 1, to interact with the bias weights
with open('processedTrain.csv') as csv_file:
  read = csv.reader(csv_file, delimiter = ',')
  for row in list(read):
    trainingData.append(numpy.array([row[0]] + [1] + row[1:]).astype(float))    #adding a column after the true value, all 1s


#read in the preprocessed test data. Add an extra row, all 1, to interact with the bias weights
with open('processedTest.csv') as csv_file:
  read = csv.reader(csv_file, delimiter = ',')
  for row in list(read):
    testData.append(numpy.array([row[0]] + [1] + row[1:]).astype(float))    #adding a column after the true value, all 1s



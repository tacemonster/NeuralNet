#Tacy Bechtel

import random
import numpy
import csv
import math
from timeit import default_timer as timer


trainingData = list(list())   #variable to hold training data
testData = list(list())   #variable to hold test data
n = 12      #number of hidden neurons
learningRate = 1

#read in the preprocessed training data. Add an extra row, all 1, to interact with the bias weights
def readTraining():
  with open('processedTrain.csv') as csv_file:
    read = csv.reader(csv_file, delimiter = ',')
    for row in list(read):
      trainingData.append(numpy.array([row[0]] + [1] + row[1:]).astype(float))    #adding a column after the true value, all 1s

def readTemp():
  with open('temp.csv') as csv_file:
    read = csv.reader(csv_file, delimiter = ',')
    for row in list(read):
      trainingData.append(numpy.array([row[0]] + [1] + row[1:]).astype(float))    #adding a column after the true value, all 1s
"""
#read in the preprocessed test data. Add an extra row, all 1, to interact with the bias weights
with open('processedTest.csv') as csv_file:
  read = csv.reader(csv_file, delimiter = ',')
  for row in list(read):
    testData.append(numpy.array([row[0]] + [1] + row[1:]).astype(float))    #adding a column after the true value, all 1s
"""

def main():
  readTemp()
  weightsji = numpy.array(setWeights(785, n))
  weightskj = numpy.array(setWeights(n, 10))
  forwardPass = passForward(trainingData[0][1:], weightsji)
  forwardPass = sigmoid(forwardPass)
  print 'post sigmoid'
  print forwardPass
  secondPass = passForward(forwardPass, weightskj)
  secondPass = numpy.array(sigmoid(secondPass))
  print 'output'
  print secondPass
  targets = targetArray(trainingData[0][0])
  print 'targets:'
  print targets
  print 'Actual Value:', trainingData[0][0]
  outputDelta = numpy.array(outputError(targets, secondPass))
  print 'output error:'
  print outputDelta
  hiddenDelta = hiddenError(forwardPass, outputDelta)
  print 'hidden error:'
  for x in range(0, n):
    print hiddenDelta[x]


def setWeights(columns, rows):
  weights = [[(float(random.randint(-5, 5))/100) for i in range(0, rows)] for j in range(0, columns)]
  return weights

def targetArray(correctDigit):
  targets = [0.1 for i in range(0, 10)]
  targets[int(correctDigit)] = 0.9
  return targets

def passForward(data, weights):
  forwardPass = numpy.matmul(data, weights)
  return forwardPass

def sigmoid(x):
  return 1 / (1 - math.e ** -x)

def outputError(targets, results):
  deltak = [results[i]*(1-results[i])*(targets[i]-results[i]) for i in range(0, 10)]
  return deltak

def hiddenError(hiddenResults, deltak):
  tempMatrix = numpy.matmul(numpy.array([hiddenResults]).T, numpy.array([deltak]))
  deltaj = [hiddenResults[i]*(1-hiddenResults[i])*(tempMatrix[i]) for i in range(0, n)]
  return deltaj

if __name__ == '__main__':
    main()

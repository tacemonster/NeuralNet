#Tacy Bechtel

import random
import numpy
import csv
import math
from timeit import default_timer as timer


trainingData = list(list())   #variable to hold training data
testData = list(list())   #variable to hold test data
n = 12      #number of hidden neurons
learningRate = 0.1
momentum = 0.9

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

#read in the preprocessed test data. Add an extra row, all 1, to interact with the bias weights
def readTesting():
  with open('processedTest.csv') as csv_file:
    read = csv.reader(csv_file, delimiter = ',')
    for row in list(read):
      testData.append(numpy.array([row[0]] + [1] + row[1:]).astype(float))    #adding a column after the true value, all 1s

def main():
  readTemp()
  weightsji = setWeights(785, n)
  oldWeightsji = [[0.0 for i in range(0, n)] for j in range(0, 785)]
  weightskj = setWeights(n, 10)
  oldWeightskj = [[0.0 for i in range(0, 10)] for j in range(0, n)]
  weightsji, weightskj = epochTrain(weightsji, weightskj, oldWeightsji, oldWeightskj)

def epochTrain(weightsji, weightskj, oldWeightsji, oldWeightskj):
  for i in range(0, len(trainingData)):
    weightsji, weightskj, oldWeightsji, oldWeightskj = trainOnce(trainingData[i][1:], weightsji, weightskj, oldWeightsji, oldWeightskj, trainingData[i][0])
  return weightsji, weightskj

def trainOnce(inputs, weightsji, weightskj, oldWeightsji, oldWeightskj, targetVal):
  hjs = []
  hjs.append(numpy.array(sigmoid(passForward(inputs, weightsji))))
  oks = sigmoid(passForward(hjs, weightskj))
  targets = numpy.array([targetArray(targetVal)])
  deltak = numpy.array([errork(targets, oks)])
  deltaj = numpy.array(errorj(weightskj, deltak, hjs))
  kjchange = (learningRate * numpy.transpose(hjs) @ deltak - numpy.multiply(momentum, oldWeightskj))
  oldWeightskj = weightskj
  weightskj += kjchange
  holdData = numpy.array([inputs]).T
  jichange = learningRate * holdData @ deltaj.T - numpy.multiply(momentum, oldWeightsji)
  oldWeightsji = weightsji
  weightsji += jichange
  return weightsji, weightskj, oldWeightsji, oldWeightskj


def epochTest():
  return 0

def targetArray(correctDigit):
  targets = [0.1 for i in range(0, 10)]
  targets[int(correctDigit)] = 0.9
  return targets

def passForward(data, weights):
  forwardPass = numpy.matmul(data, weights)
  return forwardPass

def sigmoid(x):
  return 1 / (1 - math.e ** -x)

def errork(targets, results):
  deltak = [results[0][i]*(1-results[0][i])*(targets[0][i]-results[0][i]) for i in range(0, 10)]
  return deltak

def errorj(weights, deltaks, hresults):
  weightsxdeltas = numpy.matmul(weights, deltaks.T)
  deltaj = [hresults[0][i]*(1-hresults[0][i])*weightsxdeltas[i] for i in range(0, n)]
  return deltaj


def setWeights(columns, rows):
  weights = [[(float(random.randint(-5, 5))/100) for i in range(0, rows)] for j in range(0, columns)]
  return weights

if __name__ == '__main__':
    main()

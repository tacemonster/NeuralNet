#Tacy Bechtel


import random
import numpy
import csv
from timeit import default_timer as timer

#TODO get jupyter notebook stuff going????

trainingData = list(list())   #variable to hold training data
testData = list(list())   #variable to hold test data
n = 10      #number of hidden neurons

#read in the preprocessed training data. Add an extra row, all 1, to interact with the bias weights
with open('processedTrain.csv') as csv_file:
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
  start = timer() #used to figure out how long the trial took
  learningRate = 0.01   #changed manually for each trial to 1.0, 0.1, and 0.01 respectively

  #begin the trial
  trainLen = len(trainingData)  #saved training data length in an attempt to minimize performance time
  inputWeights = numpy.array(setWeights(785, n)) #weights are now randomized
  hiddenWeights = numpy.array(setWeights(n, 10))
  for i in range(0, 50):
    correctCount = runAllTraining(trainingData, inputWeights, hiddenWeights)    #run all with constant weights
    correctTest = runAllTraining(testData, inputWeights, hiddenWeights)         #run all with constant weights
    inputWeights, hiddenWeights = runEpoch(trainingData, inputWeights, hiddenWeights, learningRate) #run the epoch, where weights are changed
    print 'Epoch', i, ':'     #display for each epoch
    print '\t Training Data:', correctCount, "/", trainLen
    print '\t Testing Data:', correctTest, "/", len(testData)
  #the final post-epoch run:
  correctCount = runAllTraining(trainingData, inputWeights)    #run all with constant weights
  correctTest, confusion = finalRun(testData, inputWeights)    #run all with consant weights, getting confusion matrix back
  print 'Epoch', 50, ':'    #display to command line for last epoch
  print '\t Training Data:', correctCount, "/", trainLen
  print '\t Testing Data:', correctTest, "/", len(testData)
  print confusion   #display confusion matrix, though not well-formatted
  end = timer()

  print(end - start)    #how long did it take to run?! n = 10    #hidden units can be changed to determine peak performance

#randomly assign weights between -0.05 and 0.05 for each pixel in each perceptron
def setWeights(columns, rows):
  weights = [[(float(random.randint(-5, 5))/100) for i in range(0, rows)] for j in range(0, columns)]
  return weights


#TODO update train. Maybe train 1 and train 2?
#function runs one 'image' through the ten perceptrons and returns the resulting matrix and its largest value's element
def train(greyVals, weights):
  usable = [greyVals]  #make it a matrix soI can use matmul
  result = numpy.matmul(usable, weights)    #matrix multiplication to find each perceptron's result
  return (result.argmax(), result)    #returns the element with the largest value and the resulting matrix of values

#TODO fix this for new functionality
#function updates all weights for all perceptrons based on the results from a train() call
def updateWeights(weights, allOutputs, learningRate, expectedResult, greyVals):
  results = []
  for i in range(0, 10):
    if i == expectedResult:   #find t(k)
      t = 1   #expected true result
    else:
      t = 0   #expected false result
    actual = allOutputs[0][i] > 0   #did the perceptron 'fire'
    results = numpy.append(results, t - actual).astype(float)   #append expected minus output to results list
  greyVector = numpy.array([greyVals])    #turn it into a matrix for matrix multiplication
  delta = numpy.matmul(greyVector.T, [results]).astype(float) * learningRate    #multiply t(k) - y(k) and x(i)s, then multipy by learning rate
  newWeights = weights + delta    #make new matrix with updated weights
  return newWeights



#funtion runs one fully epoch of all training data, calling updateWeights() after each call to train()
#TODO update this function for new levels
def runEpoch(greyScale, weights, learningRate):
  numpy.random.shuffle(greyScale)   #randomize the data at the beginning of the epoch
  for i in range(0, len(greyScale)):
    result, allOutputs = train(greyScale[i][1:], weights) #call train()
    updated = updateWeights(weights, allOutputs, learningRate, greyScale[i][0], greyScale[i][1:]) #catch new weights
  return updated

#TODO fix this for new funcitonality
#poorly named, this function is useable with either training or testing data to run all without updating the weights
def runAllTraining(greyScale, weights):
  correctCount = 0
  numpy.random.shuffle(greyScale)   #randomize the data
  for i in range(0, len(greyScale)):
    result, allOutputs = train(greyScale[i][1:], weights)   #call train function
    if result == greyScale[i][0]:   #if the largest value from the resulting outputs is also the correct output,
      correctCount += 1             #increment the count of correctly identified images
  return correctCount

#TODO fix this for new functionality
#very similar to the runAllTraining() function, with special behavior for the confusion matrix
def finalRun(greyScale, weights):
  correctCount = 0
  numpy.random.shuffle(greyScale) #randomize the data
  confusion = [[0 for i in range(0, 10)] for j in range(0, 10)]   #allocate the confusion matrix
  for i in range(0, len(greyScale)):
    result, allOutputs = train(greyScale[i][1:], weights)   #call train function
    if result == greyScale[i][0]:   #if the largest value from the resulting outputs is also the correct output,
      correctCount += 1             #increment the count of correctly identified images
    confusion[int(greyScale[i][0])][result] += 1    #add the image to the confusion matrix in the correct location
  return correctCount, confusion    #return both the number of correctly identified images and the confusion matrix



if __name__ == '__main__':
    main()

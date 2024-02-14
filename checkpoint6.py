# Load the data file
import csv
with open('Workspace/GoMyCode/iris.data.txt') as csvfile:
    lines = csv.reader(csvfile)
    for row in lines:
        print(','.join(row))

# Split the data into training and test dataset
import random
def loadDataset(filename, split, trainingSet=[],testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

# test the function
trainingSet = []
testSet = []
loadDataset('Workspace/GoMyCode/iris.data.txt', 0.66, trainingSet, testSet)
# print('Train: ' + repr(len(trainingSet)))
# print('Test: ' + repr(len(testSet)))

# Similarity
# Euclidean distance

import math
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

# # Testing with sample data
# data1 = [2,2,2,'a']
# data2 = [4,4,4,'b']
# distance = euclideanDistance(data1,data2, 3)
# print('Distance: ' + repr(distance))

# Neigbours
# Defining a function to use to collect the k most similar instances
import operator
def getNeigbours(trainingSet, testInstance, k):
    distance = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance,trainingSet[x],length)
        distance.append((trainingSet[x], dist))
    distance.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distance[x][0])
    return neighbors

# # Testing with sample data
# trainSet = [[2,2,2,'a'],[4,4,4,'b']]
# testInstance = [5,5,5]
# k = 1
# neighbours = getNeigbours(trainSet,testInstance,1)
# print(neighbours)

# Response
# Defining a fucntion to devise a predicted response based on those neighbors.

def getResponse(neigbours):
    classVotes = {}
    for x in range(len(neigbours)):
        response = neigbours[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]

# # Testing the function
# neigbours = [[1,1,1,'a'],[2,2,2,'a'],[3,3,3,'b']]
# response = getResponse(neigbours)
# print(response)

# Accuracy
# Define a function to get the accuracy of the prediction

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def main():
    # Prepare data
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset('Workspace/GoMyCode/iris.data.txt', split, trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    
    # Predictiion
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeigbours(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual = ' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
if __name__ == "__main__":
    main()
    
    # Manhattan Distance Function
    def manhattanDistance(instance1, instance2, length):
        distnace = 0
        for x in range(length):
            distnace += abs(instance1[x] - instance2[x])
        return distnace
    
    # Using the manhattan distance function with getneighbors function
    def getNeighborsManhattan(trainingSet, testInstance, k):
        distance = []
        length = len(testInstance) - 1
        for x in range(len(trainingSet)):
            dist = manhattanDistance(testInstance, trainingSet[x], length)
            distance.append((trainingSet[x], dist))
        distance.sort(key=operator.itemgetter(1))
        neighbours = []
        for x in range(k):
            neighbours.append(distance[x][0])
        return neighbours
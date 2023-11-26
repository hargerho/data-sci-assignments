"""CA Assignment 1 Data Classification

Implementation of The Perceptron Algorithm

Classes
-------
Perceptron
     Perceptron class
Methods
-------
main()
    The main function of the script
"""

import numpy as np


class Perceptron:

    """
    A Perceptron class
    ...

    Attributes
    ----------
    maxEpoch : int
            Number of iterations
    classA : float
        One of the classes in comparison
    classB : float
        The opposing class in comparison.
        If there are specific class to compare, classB is None
    trainSet : Array
        Containing the features and class label of each data sample

    Methods
    -------
    train(l2Reg=float)
        Trains and updates the weights and bias of the perceptron
    test(testSet=Array, restFlag=Boolean)
        Takes in the weights and bias from train function, returns prediction
        given from the features in the testSet
    accuracy()
        Returns accuracy of prediction vs true label of the specific perceptron

    """

    def __init__(self, maxEpoch, classA, classB, trainSet):

        """
        Constructs the attributes for Perceptron object

        Parameters
        ----------
        maxEpoch : int
            Number of iterations
        classA : float
            One of the classes in comparison
        classB : float
            The opposing class in comparison.
            If there are specific class to compare, classB is None
        trainSet : Array
            Containing the features and class label of each data sample
        """

        self.maxEpoch = maxEpoch
        self.trainSet = trainSet

        self.classA = classA
        self.classB = classB

        self.numInstances, self.numFeatures = self.trainSet.shape
        self.weights = np.zeros(self.numFeatures-1, np.float16)
        self.b = 0

    def train(self, l2Reg):

        """
        Trains and updates the weights and bias of the perceptron

        Parameters
        ----------
        l2Reg : float
            The regularisation term for weight updates

        Returns
        -------
        Nothing
        """

        # Split feature and labels
        classLabel = self.trainSet[:,4]
        features = self.trainSet[:,:4]

        # Binary Classification
        y_true = [1 if label == self.classA else -1 for label in classLabel]

        # Intermediary step for the regularised update
        a = float(1-(2*l2Reg))

        for epoch in range(self.maxEpoch):
            for j in range(self.numInstances):
                activation = np.dot(self.weights, features[j]) + self.b
                # Misclassification
                if (y_true[j]*activation) <= 0:
                    # Updating the weights
                    for k in range(len(features[j])):
                        self.weights[k] = a*self.weights[k] + features[j][k]*y_true[j]
                    self.b += y_true[j]
                # Correct classification
                else:
                    # Update the weights
                    for k in range(len(features[j])):
                        self.weights[k] = (self.weights[k]*a)

    def test(self, testSet, restFlag):

        """
        Trains and updates the weights and bias of the perceptron

        Parameters
        ----------
        testSet : Array
            Containing the features and class label of each data sample
            of the desired dataset for testing
        restFlag : Boolean
            A flag that determines if the test function is for 1vs1 application
            or 1vsrest application

        Returns
        -------
        y_pred : Array
            Array of predicted values
        activation : float
            The activation score of the perceptron
        """

        # For 1vs1 appraoch
        if not restFlag:
            # Extracting only the features
            features = testSet[:,:4]

            activation = np.dot(features, self.weights) + self.b

            # Obtain the predicted labels
            y_pred = np.sign(activation)
            return y_pred
        # For 1vsrest approach
        else:
            features = testSet
            activation = np.dot(self.weights, features) + self.b
            return activation

    def accuracy(self, dataset, y_pred):

        """
        Returns accuracy of prediction vs true label of the specific perceptron

        Parameters
        ----------
        dataset : Array
            Containing the features and class label of each data sample
        y_pred : Array
            Array of predicted values

        Returns
        -------
        accuracy : float
            The accuracy value of the predicted labels vs true labels
        """

        # Extracting the class label
        classlabel = dataset[:,4]

        y_true = np.array([1 if label == self.classA else -1 for label in classlabel])

        # Number of correctly classified data samples
        positive = np.sum(y_pred == y_true)

        # Calculation of accuracy
        accuracy = (positive / len(y_true)) *100

        return accuracy

# Helper Functions
def masking1v1(dataset, classCombinations):

    """
    Separated the classes 1 vs 2, 2 vs 3 and 1 vs 3 accordingly

    Parameters
    ----------
    dataset : Array
        Containing the features and class label of each data sample
    classCombinations : List
        List of class combinations that is used in this experiment

    Returns
    -------
    classDatasets : List
        List of features and labels of the separated dataset according to
        the class combinations
    """

    # Extracting the class label
    mask = dataset[:,4]

    classDatasets = []

    # Separating the data set for each of the classes for 1vs1 approach
    for combination in classCombinations:
        classA = combination[0]
        classB = combination[1]
        maskClass = ((mask == classA) | (mask == classB))
        dataSplit = dataset[maskClass]
        classDatasets.append(dataSplit)

    return classDatasets

def overallAccuracy(dataset, y_pred):

    """
    Returns accuracy of prediction vs true label of 1vsrest approach

    Parameters
    ----------
    dataset : Array
        Containing the features and class label of each data sample
    y_pred : Array
        Array of predicted values

    Returns
    -------
    accuracy : float
        The accuracy value of the predicted labels vs true labels
    """

    # Extracting the class label
    y_true = dataset[:,4]

    # Number of correctly classified data samples
    positive = np.sum(y_pred == y_true)

    accuracy = (positive / len(y_true)) *100

    return accuracy

def q3Run(classCombinations, idx, trainDataList, testDataList, maxEpoch):

    """
    Executes question 3
    1vs1 approach to classification

    Parameters
    ----------
    classCombinations : List
        CList of features and labels of the separated dataset according to
    the class combinations
    idx : int
        A counter to keep track of the class index
    trainDataList : List
        List of training dataset
    testDataList : List
        List of testing dataset
    maxEpoch : int
        Number of iterations

    Returns
    -------
    Nothing
    """

    # It is not a 1vsrest approach
    restFlag = False
    l2Reg = 0

    # For the different comparison combinations
    for combination in classCombinations:
        classA = combination[0]
        classB = combination[1]
        print("Class {} vs Class {}". format(classA, classB))

        trainSet = trainDataList[idx]
        testSet = testDataList[idx]

        perceptron3 = Perceptron(maxEpoch=maxEpoch, classA=classA, classB=classB,
                                trainSet=trainSet)

        #Training Set
        perceptron3.train(l2Reg=l2Reg)
        y_predTrain = perceptron3.test(testSet=trainSet, restFlag=restFlag)
        trainAccuracy = perceptron3.accuracy(dataset=trainSet, y_pred=y_predTrain)
        print("Train Accuracy: %.1f" % trainAccuracy)

        # Testing Set
        y_predTest = perceptron3.test(testSet=testSet, restFlag=restFlag)
        testAccuracy = perceptron3.accuracy(dataset=testSet, y_pred=y_predTest)
        print("Test Accuracy: %.1f" % testAccuracy)
        print("----------")
        idx += 1

def multiClassification(perceptronList, data, restFlag):

    """
    Multiclassification 1vsrest approach

    Parameters
    ----------
    perceptronList : List of Perceptron classes
        List of Perceptron classes for each of the classes 1, 2 and 3
    data : Array
        Dataset in consideration for the classification
    restFlag : Boolean
            A flag that determines if the test function is for 1vs1 application
            or 1vsrest application

    Returns
    -------
    predictions : List of floats
        Class label predictions using the 1vsrest approach
    """

    predictions = []
    features = data[:,:4]
    numData = data.shape[0]
    for i in range(numData):
        activationScores = [perceptronList[0].test(features[i], restFlag=restFlag),
                        perceptronList[1].test(features[i], restFlag=restFlag),
                        perceptronList[2].test(features[i], restFlag=restFlag)]
        # Storing the class label with the max activation score
        predictions.append(float(np.argmax(activationScores)+1))

    return predictions

def onevsRest(categories, idx, trainSet, testSet, maxEpoch, l2Reg):

    """
    Executing 1vsrest approach

    Parameters
    ----------
    categories : List of classes
        List of classes for each of the classes 1, 2 and 3
    idx : int
        A counter to keep track of the class index
    restFlag : Boolean
            A flag that determines if the test function is for 1vs1 application
            or 1vsrest application
    trainSet : Array
            Containing the features and class label of each data sample
            in the training dataset
    testSet : Array
            Containing the features and class label of each data sample
            of the desired dataset for testing
    maxEpoch : int
        Number of iterations
    l2Reg : float
            The regularisation term for weight updates

    Returns
    -------
    Nothing
    """

    # It is a 1vsrest approach
    restFlag = True

    # List of perceptron classes
    perceptronList = [i for i in range(len(categories))]

    # Training 3 separate perceptrons
    for category in categories:
        idx = int(category-1)

        perceptronList[idx] = Perceptron(maxEpoch=maxEpoch, classA=category, classB=None,
                                trainSet=trainSet)

        # Train perceptron with regularization
        perceptronList[idx].train(l2Reg=l2Reg)


    # Getting prediction from each perceptron
    # Training set
    trainPredictions = multiClassification(perceptronList=perceptronList, data=trainSet, restFlag=restFlag)

    # Getting prediction from each perceptron
    # Test set
    testPredictions = multiClassification(perceptronList=perceptronList, data=testSet, restFlag=restFlag)

    generaltrainAccuracy = overallAccuracy(dataset=trainSet, y_pred=trainPredictions)
    print("Multiclass-Training Accuracy: %.1f" % generaltrainAccuracy)

    generaltestAccuracy = overallAccuracy(dataset=testSet, y_pred=testPredictions)
    print("Multiclass-Testing Accuracy: %.1f" % generaltestAccuracy)

def main():
    # Reading in data files
    with open('CA1data/train.data', 'r') as file:
        trainData = [[float(value.split('-')[1]) if value.startswith('class-') else float(value) for value in line.strip().split(',')] for line in file]

    with open('CA1data/test.data', 'r') as file:
        testData = [[float(value.split('-')[1]) if value.startswith('class-') else float(value) for value in line.strip().split(',')] for line in file]

    # Convert the list of lists to a numpy array
    trainingData = np.array(trainData)

    # Shuffle the dataset
    np.random.seed(2)
    np.random.shuffle(trainingData)

    # Convert the list of lists to a numpy array
    testData = np.array(testData)

    # Shuffle the dataset
    np.random.seed(2)
    np.random.shuffle(testData)

    classCombinations = [(1.0,2.0), (2.0,3.0), (1.0,3.0)]
    categories = [1.0, 2.0, 3.0]
    regList = [0.01, 0.1, 1.0, 10.0, 100.0]

    # List of data-List of each classCombinations
    trainDataList = masking1v1(trainingData, classCombinations)
    testDataList = masking1v1(testData, classCombinations)

    maxEpoch = 20

    # Executing the assignment questions
    print("Question 3")
    print("----------")
    q3Run(classCombinations=classCombinations, idx=0,
          trainDataList=trainDataList, testDataList=testDataList,
          maxEpoch=maxEpoch)
    print("==========")

    print("Question 4")
    print("----------")
    onevsRest(categories=categories, idx=0, trainSet=trainingData, testSet=testData,
              maxEpoch=maxEpoch, l2Reg=0)
    print("==========")

    print("Question 5")
    print("----------")
    for regTerm in regList:
        print("Regularisation term =", regTerm)
        onevsRest(categories=categories, idx=0, trainSet=trainingData, testSet=testData,
                  maxEpoch=maxEpoch, l2Reg=regTerm)
        print("----------")
    print("==========")

if __name__ == "__main__":
    main()
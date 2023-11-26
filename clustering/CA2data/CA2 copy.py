# Li Yang Alphonsus Ho 201669440

"""CA Assignment 2 Data Clustering

Implementation of clustering algorithms

Methods
-------
sqDistance(x1, x2)
    Computes the squared euclidean distances between 2 features

Euclidean(x1, x2)
    Computes the euclidean distances between 2 features

distanceMatrix(dataset, dist)
    Computes distances matrix for a given dataset and a distance function

silhouetteCoefficient(data, clusterDict, distMatrix)
    Computes mean silhouette coefficient of all clusters
    Referenced from Week 7 lab solutions

silhouetteVisualizer(lst, title)
    Plots the mean sihouette score across various k values

assignClusters(cent, datapoint)
    Assigns each datapoint to its closest centroid

updateCentroids(labelledClusters, previousMean, constantFlag, maxIter, i)
    Updates the centroid based on cluster mean

initializeCentroids(data, k)
    Centroid selection for the K-means++ approach
    Selection of centroid where the next datapoint furthest away to the centroids is next centroid

KMeans(data, k, maxIter, plusFlag)
    K-Means clustering approach

intraCalculator(data, distFunction, clustering)
    Computes intra distance of datapoints within a cluster

bisectKMeans(data, s, maxIter, plusFlag)
    Bisecting K-Means clustering approach

plotSihouette(data, distMatrix, maxK, maxIter, plusFlag, kFunction, title)
    Presenting the results from the 3 approaches

main()
    The main function of the script
"""

import numpy as np
import math
import pandas as pd
import random
import matplotlib.pyplot as plt

def sqDistance(x1, x2):

    """
    Computes the squared euclidean distances between 2 features

    Parameters
    ----------
    x1 : List of floats
        List of features of object x1
    x2 : List of floats
        List of features of object x2

    Returns
    -------
    sq_distance : Float
        The computed squared euclidean distance
    """

    sq_distance = np.sum((x1-x2)**2)

    return sq_distance

def Euclidean(x1, x2):

    """
    Computes the euclidean distances between 2 features

    Parameters
    ----------
    x1 : List of floats
        List of features of object x1
    x2 : List of floats
        List of features of object x2

    Returns
    -------
    dist : Float
        The computed euclidean distance
    """

    dist = np.linalg.norm(x1 - x2)

    return dist

def distanceMatrix(dataset, dist):

    """
    Computes distances matrix for a given dataset and a distance function

    Parameters
    ----------
    dataset : Nested list of floats
        List of datapoints with a list of features
    dist : Chosen distance function
        Distance function of Euclidean

    Returns
    -------
    distMatrix : Array of floats
        Array of floats of the distance computed between objects
    """

    # Compute the number of objects in the dataset
    N = len(dataset)

    # Initialize distance matrix
    distMatrix = np.zeros((N, N))

    # Compute distances between the objects
    for i in range(N):
        for j in range (N):
            # Distance is symmetric
            if i < j:
                distMatrix[i][j] = dist(x1=dataset[i], x2=dataset[j])
                distMatrix[j][i] = distMatrix[i][j]

    return distMatrix

def silhouetteCoefficient(data, clusterDict, distMatrix):

    """
    Computes mean silhouette coefficient of all clusters
    Referenced from Week 7 lab solutions

    Parameters
    ----------
    data : Nested list of floats
        List of datapoints with a list of features
    clusterDict : Dictionary of key: clusterNumber, value: List of datapoints
        Contains the cluster information
    distMatrix : Array of floats
        Array of floats of the distance computed between objects

    Returns
    -------
    meanSilhouettes : Float
        The mean value of the silhouette coefficient for all clusters
    """

    numData = len(data)

    # Initialization
    silhouettes = [0 for i in range(numData)]
    a = [0 for i in range(numData)]
    b = [math.inf for i in range(numData)]

    # Indexing the clusterDict objects
    clusterIdxDict = {}
    for clusterNum, clusterObj in clusterDict.items():
        indexing = []
        for obj in clusterObj:
            for i in range(len(data)):
                if (obj == data[i]).all():
                    indexing.append(i)
        clusterIdxDict[clusterNum] = indexing

    # Iterating over all the datapoints
    for i in range(numData):
        # For each of the clusters
        for clusterNum, clusterIdx in clusterIdxDict.items():

            clusterSize = len(clusterIdx)

            # If the datapoint is in the iterated cluster
            if i in clusterIdx:

                # if there is more than 1 datapoint in the cluster
                if clusterSize > 1:

                    # a = average intra-cluster distance
                    # distMatrix[i] -> the first point
                    # [cluster] -> list of 2nd points and summing up the distances
                    a[i] = np.sum(distMatrix[i][clusterIdx])/(clusterSize-1)
                else:
                    a[i] = 0

            else:
                # b = average inter-cluster distance
                tempb = np.sum(distMatrix[i][clusterIdx])/clusterSize
                if tempb < b[i]:
                    b[i] = tempb

    # Computing the silhouettes scores
    for i in range(numData):
        if a[i] == 0:
            silhouettes[i] = 0
        else:
            silhouettes[i] = (b[i]-a[i])/np.max([a[i], b[i]])

    # Computing the mean silhouettes scores
    meanSilhouettes = np.mean(silhouettes)

    return meanSilhouettes

def silhouetteVisualizer(lst, title):

    """
    Plots the mean sihouette score across various k values

    Parameters
    ----------
    lst : List of sihouette scores
        List of sihouette scores from each k values
    title : String
        Title of the plot

    Returns
    -------
    Nothing
    """
    legend = ['K-Means', 'K-Means++', ]
    # Initializing the figure
    plt.figure()
    x = []
    y = []

    # Creating the axis
    for i, val in enumerate(lst):
        if not np.isnan(val):
            x.append(i+2)
            y.append(val)

    # Plotting the figure
    plt.plot(x, y)
    plt.xticks(np.arange(min(x), max(x)+1, 1))
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Coefficient')
    plt.title(title="Combined Plot", fontsize=16)

    plt.show()

def assignClusters(cent, datapoint):

    """
    Assigns each datapoint to its closest centroid

    Parameters
    ----------
    cent : Nested list of floats
        List of centroids containing its features
    datapoint : Nested list of floats
        List of datapoints with a list of features

    Returns
    -------
    labelledClusters : Array
        Array of datapoints where a labelled is added to the last column of each object
    """

    # Create a list of the index of the assigned centroids
    centIdx = []

    # For each datapoint
    for dataFeature in datapoint:
        # The distance of each datapoint feature to each centroid feature
        distanceList = []
        for centFeature in cent:
            distanceList.append(sqDistance(x1=dataFeature, x2=centFeature))

        # Get first centroid index with the smallest distance to the datapoint feature
        centIdx.append(np.argmin(distanceList))

    # Appending the cluster label to each datapoint
    labelledClusters = np.concatenate((datapoint, np.array(centIdx)[:, np.newaxis]), axis=1)

    return labelledClusters

def updateCentroids(labelledClusters, previousMean, constantFlag, maxIter, i):

    """
    Updates the centroid based on cluster mean

    Parameters
    ----------
    labelledClusters : Nested list of floats
        List of centroids containing its features
    previousMean : Array of floats
        The cluster mean of the previous iteration
    constantFlag : bool
        Termination flag when the mean are constant
    maxIter : int
        Maximum iteration before termination
    i : int
        Iteration counter

    Returns
    -------
    newCent : Nested list of floats
        List of updated centroids
    constantFlag : Bool
        Termination flag when the mean are constant
    previousMean : Array of floats
        The cluster mean of the previous iteration
    """

    # Creating a new centroid list tracker
    newCent = []

    # For each of the cluster label in the labbelledCluster
    for labels in np.unique(labelledClusters[:, -1]):

        # Get the current cluster of the datapoint
        currentCluster = labelledClusters[labelledClusters[:, -1] == labels][:, :-1]

        # Computing the mean of all datapoints that are assigned to their clusters
        clusterMean = np.mean(currentCluster, axis=0)

        # Cluster mean is the new centroid
        newCent.append(clusterMean)

    # If no change in centroids
    if ((clusterMean == previousMean).all()) or (maxIter == i+1):
        constantFlag = True
    else:
        previousMean = clusterMean

    return newCent, constantFlag, previousMean

def initializeCentroids(data, k):

    """
    Centroid selection for the K-means++ approach
    Selection of centroid where the next datapoint furthest away to the centroids is next centroid

    Parameters
    ----------
    data : Nested list of floats
        List of datapoints with a list of features
    k : int
        Number of clusters

    Returns
    -------
    centroids : Nested list of floats
        List of selected centroids
    """

    # initialize the centroids list and add
    centroids = []
    np.random.seed(42)

    # Randomly selecting the first centroid
    x = data[np.random.randint(0, len(data))]
    centroids.append(x)

    # For the remaining centroids
    for iterator in range(k - 1):

        # Stores a list of distances of centroid to datapoints
        dist = []

        # For each datapoint
        for idx, datapoint in enumerate(data):

            # Creating arbitary infinite distance
            maxDist = math.inf

            # For each previously selected centroid
            for j in range(len(centroids)):

                # Compute the distance of datapoint to the centroid
                tmpDist = sqDistance(x1=datapoint, x2=centroids[j])

                # Get the smaller distance
                maxDist = min(maxDist, tmpDist)

            dist.append(maxDist)

        # Next datapoint furthest away to the centroids is next centroid
        nextCent = data[np.array(dist).argmax(), :]
        centroids.append(nextCent)

        # Reset the distance list
        dist = []

    return centroids

def KMeans(data, k, maxIter, plusFlag):

    """
    K-Means clustering approach

    Parameters
    ----------
    data : Nested list of floats
        List of datapoints with a list of features
    k : int
        Number of clusters
    maxIter : int
        Maximum iteration before termination
    plusFlag : bool
        Flag to switch between random or k-means++ centroid selection

    Returns
    -------
    clusterDict : Dictionary of key: clusterNumber, value: List of datapoints
        Contains the cluster information
    """

    # Initialize the constantFlag used in updateCentroids
    constantFlag = False
    random.seed(42)

    # If K-means method
    if not plusFlag:
        # Pick k random data objects for initial centroid set
        centroids = []
        selectedIndex = random.sample(range(0, len(data)), k=k)
        for index in selectedIndex:
            centroids.append(data[index])

    # If K-means++ method
    else:
        centroids = initializeCentroids(data=data, k=k)

    # Creating an empty array
    previousMean = np.empty((len(data), len(data[0])))

    for i in range(maxIter):
        clusters = assignClusters(cent=centroids, datapoint=data)
        # Loop iterations if the mean is not constant
        if not constantFlag:
            centroids, constantFlag, previousMean = updateCentroids(labelledClusters=clusters, previousMean=previousMean,
                                                                    constantFlag=constantFlag, maxIter=maxIter, i=i)
        # Terminates
        else:
            break

    # Re-arranging into a dictionary data structure
    clusterDict = {}

    # iterate through the list of lists
    for sublist in clusters:

        # extract the last element as the key
        key = sublist[-1]

        # extract the sublist without the last element as the value
        value = sublist[:-1]

        # append the value to the corresponding key in the dictionary
        clusterDict.setdefault(int(key), []).append(value)


    return clusterDict

def intraCalculator(data, distFunction, clustering):

    """
    Computes intra distance of datapoints within a cluster

    Parameters
    ----------
    data : Nested list of floats
        List of datapoints with a list of features
    distFunction : Chosen distance function
        Distance function chosen for intraCalcuator
    clustering : List of floats
        The features of an data object in the cluster

    Returns
    -------
    totalSum : Float
        The total sum of the intra distance between datapoints within a cluster
    """

    distMatrix = distanceMatrix(dataset=data, dist=distFunction)

    # Initializing variables
    numData = len(data)
    a = [0 for i in range(numData)]

    # Converting clustering to get the index in main dataset
    clusterIdx = []
    for cluster in clustering:
        indexing = [i for i, x in enumerate(data) if x in cluster]
        clusterIdx.append(indexing)

    # Re-formating the clusterIdx list
    clusterIdx = [idx for sublist in clusterIdx for idx in sublist]

    # Iterate over all datapoints
    for i in range(numData):
        clusterSize = len(clusterIdx)
        if clusterSize > 1:
            a[i] = np.sum(distMatrix[i][clusterIdx])
        else:
            a[i] = 0

    totalSum = sum(a)

    return totalSum

def bisectKMeans(data, s, maxIter, plusFlag):

    """
    Bisecting K-Means clustering approach

    Parameters
    ----------
    data : Nested list of floats
        List of datapoints with a list of features
    s : int
        Number of clusters
    maxIter : int
        Maximum iteration before termination
    plusFlag : bool
        Flag to switch between random or k-means++ centroid selection

    Returns
    -------
    clusterDict : Dictionary of key: clusterNumber, value: List of datapoints
        Contains the cluster information
    """

    # Assigning all datapoints into the same root cluster
    clusterDict = KMeans(data, k=1, maxIter=maxIter, plusFlag=plusFlag)

    if s == 1:
        return clusterDict
    elif s == 2:
        clusterDict = KMeans(clusterDict.pop(0), k=2, maxIter=maxIter, plusFlag=plusFlag)
        return clusterDict

    # If want more than 2 clusters
    else:
        # Initial Split
        clusterDict = KMeans(data=clusterDict.pop(0), k=2, maxIter=maxIter, plusFlag=plusFlag)

        # Continuing splitting the clusters
        while len(clusterDict) != s:
            # Selecting the cluster with a higher distance metric
            intraDict = {}
            for clusterNum, clusterObj in clusterDict.items():
                intraDist = intraCalculator(data=data, distFunction=sqDistance, clustering=clusterObj)
                intraDict[clusterNum] = intraDist

            # Get the index of the cluster with a higher distance metric
            maxClusterIdx = max(intraDict, key=intraDict.get)

            # The next cluster to split is the cluster with higher distance metric
            tmpDict = KMeans(data=clusterDict.pop(maxClusterIdx), k=2, maxIter=maxIter, plusFlag=plusFlag)

            ## clusterDict housekeeping
            # Find the highest key number in dict1
            highestKey = max(clusterDict.keys())

            # Create a new dictionary with renamed keys from dict2
            tmpDict = {i+highestKey+1: v for i, (k, v) in enumerate(tmpDict.items())}

            # Merge all the dictionaries
            clusterDict = {**clusterDict, **tmpDict}

        return clusterDict

def plotSihouette(data, distMatrix, maxK, maxIter, plusFlag, kFunction, title):

    """
    Presenting the results from the 3 approaches

    Parameters
    ----------
    data : Nested list of floats
        List of datapoints with a list of features
    distMatrix : Array of floats
        Array of floats of the distance computed between objects
    maxK : int
        Maximum number of clusters
    maxIter : int
        Maximum iteration before termination
    plusFlag : bool
        Flag to switch between random or k-means++ centroid selection
    kFunction : Function
        Desired clustering function to execute
    title : String
        The title of the plot

    Returns
    -------
    Nothing
    """

    # Initializing sihouette tracker
    silhouetteList = []

    for k in range(1,maxK):

        # Executing the clustering algorithm
        clusters = kFunction(data, k, maxIter, plusFlag)

        # Silhouette Coeff not defined for k = 1
        if k > 1:
            silhouetteScore = silhouetteCoefficient(data=data, clusterDict=clusters, distMatrix=distMatrix)
            silhouetteList.append(silhouetteScore)

    # Visualise the results
    silhouetteVisualizer(lst=silhouetteList, title=title)

def main():
    # Reading in the data
    data = pd.read_csv('Data Mining CA2\CA2data\dataset', delimiter=' ', header=None)

    # Dropping the first column containing the string text
    data = data.iloc[:, 1:]

    # Data is list of list, where each element in an object with 300 features
    data = data.to_numpy()

    # Initializing parameters
    maxK = 3
    maxIter = 10

    # Compute distance matrix
    distMatrix = distanceMatrix(dataset=data, dist=Euclidean)

    #K-means
    plotSihouette(data=data, distMatrix=distMatrix, maxK=maxK, maxIter=maxIter,
                plusFlag=False, kFunction=KMeans, title='K-Means')

    #K-means++
    plotSihouette(data=data, distMatrix=distMatrix, maxK=maxK, maxIter=maxIter,
                plusFlag=True, kFunction=KMeans, title='K-Means++')

    #Bisecting K-means
    plotSihouette(data=data, distMatrix=distMatrix, maxK=maxK, maxIter=maxIter,
                plusFlag=False, kFunction=bisectKMeans, title='Bisecting K-Means')

if __name__ == "__main__":
    main()
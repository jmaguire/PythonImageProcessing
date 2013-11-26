import numpy as np


def kmeans(data,k,maxIter):
    if data.shape[0] == 1:
        np.reshape(b, (len(b),1))
    """
    Runs K-means to learn k centroids, for maxIter iterations.
    
    Args:
      k - number of centroids.
      data - rows are data vectors
      maxIter - number of iterations to run K-means for

    Returns:
      centroids - 2D numpy array of size k x dimension
    """
    # This line starts you out with randomly initialized centroids in a matrix 
    # with patchSize rows and k columns. Each column is a centroid.
    
    centroids = data[np.random.choice(data.shape[0], k, replace=False),:]
    
    numPoints = data.shape[0]
    
    print centroids
    for i in range(maxIter):
        oldCentroids = centroids
        # BEGIN_YOUR_CODE (around 19 lines of code expected)
        ## centroid each patch is closest too
        categories = np.array([1]*numPoints) 
        ## categorize patches by closest centroid 
        for j in range(numPoints):
            ## iterate over each centroid record the distance
            distances = np.array([np.linalg.norm(data[j,:] - centroids[n,:]) for n in range(k)])
            categories[j] = distances.argmin() ## returns index of minimum, index of closest centroid
        ## find new means
        for n in range(k):
            centroid = np.mean(data[categories == n, :],0);
            centroids[n,:] = centroid
        if np.array_equal(centroids,oldCentroids): break
    return centroids
    
def tictactoeMeans(data,k,maxIter):
    x = data[:,0:1]
    y = data[:,1:]
    
    xmin = min(x[:,0])
    xmax = max(x[:,0])
    ymin = min(y[:,0])
    ymax = max(y[:,0])
    
    ## create 4 centroids
    centroids = []
    centroids.append([(xmin+xmax)/2,ymin])
    centroids.append([(xmin+xmax)/2,ymax])
    centroids.append([xmin,(ymin+ymax)/2])
    centroids.append([xmax,(ymin+ymax)/2])
    centroids = np.array(centroids);
    numPoints = data.shape[0]
    categories = np.array([1]*numPoints)
    distances  = np.array([1]*numPoints)
    
    numPoints = data.shape[0]
    
    print centroids
    for i in range(maxIter):
        oldCentroids = centroids
        # BEGIN_YOUR_CODE (around 19 lines of code expected)
        ## centroid each patch is closest too
        categories = np.array([1]*numPoints) 
        ## categorize patches by closest centroid 
        for j in range(numPoints):
            ## iterate over each centroid record the distance
            distances = np.array([np.linalg.norm(data[j,:] - centroids[n,:]) for n in range(k)])
            categories[j] = distances.argmin() ## returns index of minimum, index of closest centroid
        ## find new means
        for n in range(k):
            centroid = np.mean(data[categories == n, :],0);
            centroids[n,:] = centroid
        if np.array_equal(centroids,oldCentroids): break
    return centroids

a = np.zeros((4,2))
a[0,:] = [1,2.5] 
a[1,:] = [1,2]
a[2,:] = [-4,3]
a[3,:] = [-4,4]
# print a
# print 'centroids', kmeans(a,2,10)
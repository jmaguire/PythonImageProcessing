import numpy as np


def kmeans(data,k,maxIter):
    """
    Runs K-means to learn k centroids, for maxIter iterations.
    
    Args:
      k - number of centroids.
      data - 2D numpy array of size numPoints x dimension
      maxIter - number of iterations to run K-means for

    Returns:
      centroids - 2D numpy array of size k x dimension
    """
    # This line starts you out with randomly initialized centroids in a matrix 
    # with patchSize rows and k columns. Each column is a centroid.
    if k == 4:
        centroids = []
        xmin = min(data[:,0])
        xmax = max(data[:,0])
        ymin = min(data[:,1])
        ymax = max(data[:,1])
        
        temp = data[data[:,0] == xmin, :]
        temp = np.mean(temp,0);
        centroids.append(temp)
        
        temp = data[data[:,0] == xmax, :]
        temp = np.mean(temp,0);
        centroids.append(temp)
        
        temp = data[data[:,1] == ymin, :]
        temp = np.mean(temp,0);
        centroids.append(temp)
        
        temp = data[data[:,1] == ymax, :]
        temp = np.mean(temp,0);
        centroids.append(temp)
        centroids = np.array(centroids)
        print centroids
    else: 
        centroids = data[np.random.choice(data.shape[0], k, replace=False),:]
    numPoints = data.shape[0]
    
    for i in range(maxIter):
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
         
    return centroids
a = np.zeros((4,2))
a[0,:] = [1,2.5] 
a[1,:] = [1,2]
a[2,:] = [-4,3]
a[3,:] = [-4,4]
# print a
# print 'centroids', kmeans(a,2,10)